"""Детерминированное извлечение диалогов из поддерживаемых схем AEF.

Модуль намеренно не ищет произвольные JSON-ключи рекурсивно. Каждый
публикуемый turn подтверждён correlation key FIPA либо явной внешней границей
request/response AEF. Неподдерживаемые и неполные данные возвращаются как
структурированная диагностика, а не превращаются в правдоподобный текст.
"""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
import logging
import re
from typing import Any, Iterable

import pandas as pd

logger = logging.getLogger(__name__)

EXTRACTION_CONTRACT = "laim-trace-turn-extraction.v2"
FIPA_SCHEMA = "fipa_acl_v1"
AEF_BOUNDARY_SCHEMA = "aef_boundary_v1"
AEF_START_AGENT_SCHEMA = "aef_start_agent_v1"
AEF_PARENT_BOUNDARY_SCHEMA = "aef_parent_boundary_v1"
AEF_SEMANTIC_TERMINAL_SCHEMA = "aef_semantic_terminal_v1"
STATE_JSON_SCHEMA = "state_json_v1"

_BOUNDARY_KINDS = {"input_request", "start_agent"}
_OUTER_KINDS = {"input_request"}
_HTTP_BOUNDARY = re.compile(
    r"^(?:get|post|put|patch|delete)\s+\S+|(?:^|[./])invoke$", re.I
)
_FIPA_MARKER = re.compile(r"[\"'](?:conversation_id|reply_with|in_reply_to)[\"']")
# Метка маршрута/класса в content.message: короткий токен без пробелов.
_LABEL_TOKEN = re.compile(r"^[\w.:-]{1,64}$")
_TECHNICAL_ACKS = {"ok", "success", "done", "accepted", "processing", "true", "false"}
_MESSAGE_PART_SEPARATOR = "\n\n"
_TERMINAL_ANSWER_KEYS = (
    "final_response",
    "message_to_user",
    "assistant_message",
    "generated_answer",
    "agent_answer",
    "agent_response",
    "full_answer",
)

TURN_COLUMNS = [
    "turn_id",
    "session_id",
    "input_query",
    "agent_response",
    "route_label",
    "schema_family",
    "schema_version",
    "entry_trace_id",
    "exit_trace_id",
    "entry_span_id",
    "exit_span_id",
    "entry_time_ns",
    "exit_time_ns",
    "same_trace",
    "same_session",
    "route_chain_complete",
    "turn_latency_ms",
    "query_source_path",
    "response_source_path",
    "route_source_path",
]

_ISSUE_COLUMNS = [
    "issue_code",
    "severity",
    "schema_version",
    "turn_id",
    "trace_id",
    "span_id",
    "details",
]


class TraceExtractionError(ValueError):
    """Таблица трейсов или конфигурация извлечения некорректна."""


@dataclass(frozen=True)
class ExtractionConfig:
    """Стабильная конфигурация ядра извлечения."""

    agent_id: str = ""
    max_issue_examples: int = 100

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_issue_examples, bool)
            or not isinstance(self.max_issue_examples, int)
            or self.max_issue_examples < 1
        ):
            raise ValueError("max_issue_examples должен быть целым >= 1")


@dataclass(frozen=True)
class ExtractionResult:
    """Полные turn, причины непубликации и агрегированные доказательства."""

    turns: pd.DataFrame
    issues: pd.DataFrame
    report: dict[str, Any]


@dataclass
class _FipaEvents:
    """Входы и выходы пользователя, сгруппированные по protocol key."""

    entries: dict[tuple[str, str], list[dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    exits: dict[tuple[str, str], list[dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    traces: set[str] = field(default_factory=set)
    non_fipa_records: list[dict[str, object]] = field(default_factory=list)
    counterparts: dict[str, int] = field(default_factory=dict)
    rows: int = 0
    malformed_entries: int = 0
    malformed_exits: int = 0
    failures_without_text: int = 0


@dataclass(frozen=True)
class _FipaRow:
    """Граничный span с разобранными FIPA-конвертами."""

    record: dict[str, object]
    incoming: dict[str, Any] | None
    incoming_path: str
    outgoing: dict[str, Any] | None
    outgoing_path: str

    @property
    def conversation_id(self) -> str:
        return _identifier(
            (self.incoming or {}).get("conversation_id")
            or (self.outgoing or {}).get("conversation_id")
        )

    @property
    def time_ns(self) -> int | None:
        return _integer(self.record.get("start_time_ns"))


def _identifier(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _integer(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(str(value).strip())
    except (TypeError, ValueError, OverflowError):
        return None


def _json_value(value: object) -> object:
    """Разобрать JSON или legacy Python repr без выполнения входного кода."""
    current = value
    for _ in range(3):
        if not isinstance(current, str):
            return current
        text = current.strip()
        if not text or text[:1] not in "[{\"'":
            return current
        try:
            decoded = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            try:
                decoded = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return current
        if decoded == current:
            return current
        current = decoded
    return current


def _mapping(value: object) -> dict[str, Any] | None:
    value = _json_value(value)
    return value if isinstance(value, dict) else None


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _semantic(value: object) -> str:
    return re.sub(r"\s+", " ", _identifier(value)).casefold()


def _string_series(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip()


def _looks_like_envelope(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    keys = set(value)
    return bool(
        "content" in keys
        and {"sender", "receiver"} & keys
        and {"conversation_id", "reply_with", "in_reply_to", "message_id"} & keys
    )


def _incoming_envelope(
    payload: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str]:
    if payload is None:
        return None, ""
    for key in ("message", "incoming"):
        candidate = _mapping(payload.get(key))
        if _looks_like_envelope(candidate):
            return candidate, f"input_text.{key}"
    if _looks_like_envelope(payload):
        return payload, "input_text"
    return None, ""


def _outgoing_envelope(
    payload: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str]:
    if payload is None:
        return None, ""
    candidate = _mapping(payload.get("outgoing"))
    if _looks_like_envelope(candidate):
        return candidate, "output_text.outgoing"
    if _looks_like_envelope(payload):
        return payload, "output_text"
    return None, ""


def _message_parts(envelope: dict[str, Any] | None) -> list[tuple[int, str]]:
    """Непустые текстовые части content.message с их индексами.

    Виджеты и другие нетекстовые элементы (value=None) пропускаются: ответ
    пользователю — это все текстовые части, а не только первая.
    """
    if envelope is None:
        return []
    content = _mapping(envelope.get("content"))
    if content is None:
        return []
    messages = _json_value(content.get("message"))
    if not isinstance(messages, list):
        return []
    parts = []
    for index, raw in enumerate(messages):
        item = _mapping(raw)
        text = _text(item.get("value")) if item is not None else ""
        if text:
            parts.append((index, text))
    return parts


def _message_text(parts: list[tuple[int, str]]) -> str:
    return _MESSAGE_PART_SEPARATOR.join(text for _, text in parts)


def _is_request(envelope: dict[str, Any] | None) -> bool:
    """FIPA request: перформатив request либо, без перформатива, reply_with без in_reply_to."""
    if envelope is None:
        return False
    performative = _identifier(envelope.get("performative")).casefold()
    if performative:
        return performative == "request"
    return bool(_identifier(envelope.get("reply_with"))) and not _identifier(
        envelope.get("in_reply_to")
    )


def _is_reply(envelope: dict[str, Any] | None) -> bool:
    """FIPA-ответ (inform/failure/…): не request и с in_reply_to."""
    if envelope is None:
        return False
    performative = _identifier(envelope.get("performative")).casefold()
    return performative != "request" and bool(_identifier(envelope.get("in_reply_to")))


def _split_label(
    parts: list[tuple[int, str]],
) -> tuple[str, int, list[tuple[int, str]]]:
    """Отделить метку вида [label, текст...] от текста; метка — токен без пробелов."""
    if (
        len(parts) >= 2
        and _LABEL_TOKEN.match(parts[0][1])
        and not _LABEL_TOKEN.match(parts[-1][1])
    ):
        index, label = parts[0]
        return label, index, parts[1:]
    return "", -1, parts


def _route_label(
    outgoing: dict[str, Any] | None,
    outgoing_path: str,
    query: str,
    counterparts: dict[str, int],
) -> tuple[str, str]:
    """Наблюдаемое решение маршрутизации на исходящем событии входа.

    Роутер вида [label, echo вопроса] в content.message кладёт метку класса
    в текстовую часть, а receiver у него постоянный. Без такого эха меткой
    остаётся receiver, если это не контрагент-инициатор.
    """
    if outgoing is None:
        return "", ""
    parts = _message_parts(outgoing)
    query_key = _semantic(query)
    if len(parts) >= 2 and any(_semantic(text) == query_key for _, text in parts):
        for index, text in parts:
            if _semantic(text) != query_key:
                return text, f"{outgoing_path}.content.message[{index}].value"
    receiver = _identifier(outgoing.get("receiver"))
    if receiver and receiver not in counterparts:
        return receiver, f"{outgoing_path}.receiver"
    return "", ""


def _row_record(row: tuple[object, ...], columns: list[str]) -> dict[str, object]:
    return dict(zip(columns, row, strict=True))


def _candidate_mask(frame: pd.DataFrame) -> pd.Series:
    kinds = _string_series(frame["aef_kind"]).str.casefold()
    boundary = kinds.isin(_BOUNDARY_KINDS) | frame["span_name"].str.contains(
        _HTTP_BOUNDARY, na=False
    )
    input_marker = frame["input_text"].str.contains(_FIPA_MARKER, na=False)
    output_marker = frame["output_text"].str.contains(_FIPA_MARKER, na=False)
    return boundary | input_marker | output_marker


def _ordered(candidate: dict[str, Any], *, latest: bool) -> tuple[int, str, str]:
    timestamp = candidate.get("time_ns")
    order_time = timestamp if isinstance(timestamp, int) else 0
    if latest:
        order_time = -order_time
    return (
        order_time,
        _identifier(candidate.get("trace_id")),
        _identifier(candidate.get("span_id")),
    )


def _replica_values(
    candidates: Iterable[dict[str, Any]], field_name: str
) -> dict[str, object]:
    values: dict[str, object] = {}
    for candidate in candidates:
        value = candidate.get(field_name)
        key = _semantic(value)
        if key:
            values.setdefault(key, value)
    return values


def _merge_replicas(
    candidates: list[dict[str, Any]],
    fields: tuple[str, ...],
    *,
    latest: bool,
) -> tuple[dict[str, Any] | None, list[str]]:
    conflicts = [
        field_name
        for field_name in fields
        if len(_replica_values(candidates, field_name)) > 1
    ]
    if conflicts:
        return None, conflicts
    selected = dict(
        sorted(candidates, key=lambda item: _ordered(item, latest=latest))[0]
    )
    for field_name in fields:
        values = _replica_values(candidates, field_name)
        if values:
            selected[field_name] = next(iter(values.values()))
    selected["replica_count"] = len(candidates)
    return selected, []


def _issue(
    code: str,
    *,
    severity: str = "error",
    schema_version: str = "",
    turn_id: str = "",
    trace_id: str = "",
    span_id: str = "",
    details: str = "",
) -> dict[str, str]:
    return {
        "issue_code": code,
        "severity": severity,
        "schema_version": schema_version,
        "turn_id": turn_id,
        "trace_id": trace_id,
        "span_id": span_id,
        "details": details,
    }


def _typed_message(payload: dict[str, Any], roles: set[str]) -> str:
    messages = _json_value(payload.get("messages"))
    if not isinstance(messages, list):
        return ""
    for raw in reversed(messages):
        message = _mapping(raw)
        if message is None:
            continue
        role = _identifier(message.get("role") or message.get("type")).casefold()
        content = _text(message.get("content"))
        if role in roles and content:
            return content
    return ""


def _request_text(value: object) -> tuple[str, str]:
    parsed = _json_value(value)
    if isinstance(parsed, str):
        text = parsed.strip()
        if text[:1] in "[{":
            return "", ""
        return (text, "input_text") if text else ("", "")
    if not isinstance(parsed, dict):
        return "", ""
    for key in (
        "message",
        "query",
        "question",
        "user_query",
        "input_query",
        "text",
    ):
        text = _text(parsed.get(key))
        if text:
            return text, f"input_text.{key}"
    typed = _typed_message(parsed, {"human", "user"})
    if typed:
        return typed, "input_text.messages[last human].content"
    for key in ("body", "request"):
        raw_nested = parsed.get(key)
        direct_text = _text(raw_nested)
        if direct_text and direct_text[:1] not in "[{":
            return direct_text, f"input_text.{key}"
        nested = _mapping(raw_nested)
        if nested is None:
            continue
        for nested_key in ("message", "query", "question", "user_query", "input_query"):
            text = _text(nested.get(nested_key))
            if text:
                return text, f"input_text.{key}.{nested_key}"
    return "", ""


def _response_text(value: object) -> tuple[str, str]:
    parsed = _json_value(value)
    if isinstance(parsed, str):
        text = parsed.strip()
        if text[:1] in "[{":
            return "", ""
        if text and _semantic(text) not in _TECHNICAL_ACKS:
            return text, "output_text"
        return "", ""
    if not isinstance(parsed, dict):
        return "", ""
    answer_keys = (
        "answer",
        "agent_response",
        "output_answer",
        "final_response",
        "message_to_user",
    )
    for key in answer_keys:
        text = _text(parsed.get(key))
        if text:
            return text, f"output_text.{key}"
    typed = _typed_message(parsed, {"assistant", "ai", "agent"})
    if typed:
        return typed, "output_text.messages[last assistant].content"
    body = _mapping(parsed.get("body"))
    if body is not None:
        for key in answer_keys:
            text = _text(body.get(key))
            if text:
                return text, f"output_text.body.{key}"
    direct_body = _text(parsed.get("body"))
    if direct_body and _semantic(direct_body) not in _TECHNICAL_ACKS:
        return direct_body, "output_text.body"
    response = _mapping(parsed.get("response"))
    if response is not None:
        raw_response_body = response.get("body")
        response_body = _mapping(raw_response_body) or response
        for key in answer_keys:
            text = _text(response_body.get(key))
            if text:
                suffix = f"body.{key}" if response_body is not response else key
                return text, f"output_text.response.{suffix}"
        direct_response_body = _text(raw_response_body)
        if (
            direct_response_body
            and _semantic(direct_response_body) not in _TECHNICAL_ACKS
        ):
            return direct_response_body, "output_text.response.body"
    return "", ""


def _aef_turn(record: dict[str, object]) -> dict[str, Any] | None:
    query, query_path = _request_text(record.get("input_text"))
    response, response_path = _response_text(record.get("output_text"))
    schema_version = AEF_BOUNDARY_SCHEMA
    if _identifier(record.get("aef_kind")).casefold() == "start_agent":
        if query_path == "input_text.text":
            schema_version = AEF_START_AGENT_SCHEMA
        input_payload = _mapping(record.get("input_text"))
        output_payload = _mapping(record.get("output_text"))
        for key in ("goal", "task", "text"):
            fallback = _text((input_payload or {}).get(key))
            if not query and fallback:
                query, query_path = fallback, f"input_text.{key}"
                schema_version = AEF_START_AGENT_SCHEMA
                break
        for key in ("summary", "error"):
            fallback = _text((output_payload or {}).get(key))
            if not response and fallback:
                response, response_path = fallback, f"output_text.{key}"
                schema_version = AEF_START_AGENT_SCHEMA
                break
    if not query or not response or _semantic(query) == _semantic(response):
        return None
    trace_id = _identifier(record.get("trace_id"))
    span_id = _identifier(record.get("span_id"))
    time_ns = _integer(record.get("start_time_ns"))
    if not trace_id or not span_id or time_ns is None:
        return None
    return {
        "turn_id": f"aef:{trace_id}:{span_id}",
        "session_id": _identifier(record.get("session_id")),
        "input_query": query,
        "agent_response": response,
        "route_label": "",
        "schema_family": "aef_boundary",
        "schema_version": schema_version,
        "entry_trace_id": trace_id,
        "exit_trace_id": trace_id,
        "entry_span_id": span_id,
        "exit_span_id": span_id,
        "entry_time_ns": time_ns,
        "exit_time_ns": time_ns,
        "same_trace": True,
        "same_session": True,
        "route_chain_complete": False,
        "turn_latency_ms": 0.0,
        "query_source_path": query_path,
        "response_source_path": response_path,
        "route_source_path": "",
    }


def _filter_agent(
    frame: pd.DataFrame, requested_agent: str
) -> tuple[pd.DataFrame, str, dict[str, int]]:
    """Строки агента и учёт отброшенных: чужой agent_id и пустой agent_id
    не попадают в знаменатель, и отчёт обязан это показать."""
    requested = requested_agent.strip()
    agent_values = _string_series(frame["agent_id"])
    blank = agent_values.eq("")
    observed = sorted(set(agent_values[~blank]))
    if requested:
        selected = agent_values.str.upper().eq(requested.upper())
        if not selected.any():
            raise TraceExtractionError(f"В выгрузке нет спанов агента {requested}")
        agent_id = requested.upper()
    else:
        if len(observed) != 1:
            raise TraceExtractionError(
                "Без agent_id выгрузка должна содержать ровно одного агента; "
                f"найдено: {observed}"
            )
        selected = ~blank
        agent_id = observed[0].upper()
    dropped = {
        "dropped_foreign_agent_rows": int((~selected & ~blank).sum()),
        "dropped_blank_agent_rows": int(blank.sum()),
    }
    return frame.loc[selected], agent_id, dropped


def _parse_fipa_rows(candidates: pd.DataFrame, events: _FipaEvents) -> list[_FipaRow]:
    columns = list(candidates.columns)
    rows: list[_FipaRow] = []
    for raw in candidates.itertuples(index=False, name=None):
        record = _row_record(raw, columns)
        incoming, incoming_path = _incoming_envelope(_mapping(record.get("input_text")))
        outgoing, outgoing_path = _outgoing_envelope(
            _mapping(record.get("output_text"))
        )
        if incoming is None and outgoing is None:
            events.non_fipa_records.append(record)
            continue
        events.rows += 1
        events.traces.add(_identifier(record.get("trace_id")))
        rows.append(_FipaRow(record, incoming, incoming_path, outgoing, outgoing_path))
    return rows


def _conversation_initiators(rows: list[_FipaRow]) -> dict[str, int]:
    """Контрагенты агента — отправители, чей request в conversation приходит раньше,
    чем агент сам отправляет им request.

    Человек или вышестоящий агент открывает разговор своим request; нижестоящий
    агент получает request от этого агента первым и шлёт свои (уточнения) уже
    внутри разговора. Правило считается по строгому большинству conversation отправителя,
    поэтому обрезка окна выгрузки на отдельных разговорах его не меняет. Имена
    endpoint не участвуют.
    """
    incoming_first: dict[tuple[str, str], tuple[int, str, str]] = {}
    outgoing_first: dict[tuple[str, str], tuple[int, str, str]] = {}
    self_names: Counter[str] = Counter()
    for row in rows:
        conversation_id = row.conversation_id
        if not conversation_id:
            continue
        if row.incoming is not None:
            self_names[_identifier(row.incoming.get("receiver"))] += 1
        order = (
            row.time_ns if row.time_ns is not None else 0,
            _identifier(row.record.get("trace_id")),
            _identifier(row.record.get("span_id")),
        )
        if _is_request(row.incoming):
            sender = _identifier((row.incoming or {}).get("sender"))
            if sender:
                key = (conversation_id, sender)
                incoming_first[key] = min(incoming_first.get(key, order), order)
        if _is_request(row.outgoing):
            receiver = _identifier((row.outgoing or {}).get("receiver"))
            if receiver:
                key = (conversation_id, receiver)
                outgoing_first[key] = min(outgoing_first.get(key, order), order)
    # Имя самого агента — получатель его входящих; оно не может быть контрагентом.
    self_name = self_names.most_common(1)[0][0] if self_names else ""
    conversations: Counter[str] = Counter()
    leading: Counter[str] = Counter()
    for (conversation_id, sender), first_in in incoming_first.items():
        if sender == self_name:
            continue
        conversations[sender] += 1
        first_out = outgoing_first.get((conversation_id, sender))
        if first_out is None or first_in <= first_out:
            leading[sender] += 1
    return {
        sender: count
        for sender, count in leading.most_common()
        if 2 * count > conversations[sender]
    }


def _collect_fipa_events(candidates: pd.DataFrame) -> _FipaEvents:
    """Разложить кандидатов на входы/выходы контрагентов и non-FIPA записи."""
    events = _FipaEvents()
    rows = _parse_fipa_rows(candidates, events)
    events.counterparts = _conversation_initiators(rows)
    for row in rows:
        record = row.record
        trace_id = _identifier(record.get("trace_id"))
        span_id = _identifier(record.get("span_id"))
        conversation_id = row.conversation_id
        base = {
            "session_id": _identifier(record.get("session_id")),
            "trace_id": trace_id,
            "span_id": span_id,
            "time_ns": row.time_ns,
        }
        incoming = row.incoming or {}
        outgoing = row.outgoing or {}
        if (
            _is_request(row.incoming)
            and _identifier(incoming.get("sender")) in events.counterparts
        ):
            request_id = _identifier(incoming.get("reply_with"))
            label, label_index, query_parts = _split_label(_message_parts(row.incoming))
            query = _message_text(query_parts)
            if conversation_id and request_id and query and trace_id and span_id:
                if label:
                    route, route_path = (
                        label,
                        f"{row.incoming_path}.content.message[{label_index}].value",
                    )
                else:
                    route, route_path = _route_label(
                        row.outgoing, row.outgoing_path, query, events.counterparts
                    )
                events.entries[(conversation_id, request_id)].append(
                    {
                        **base,
                        "input_query": query,
                        "route_label": route,
                        "downstream_request_id": _identifier(
                            outgoing.get("reply_with")
                        ),
                        "query_source_path": f"{row.incoming_path}.content.message[*].value",
                        "route_source_path": route_path,
                    }
                )
            else:
                events.malformed_entries += 1
        if (
            _is_reply(row.outgoing)
            and _identifier(outgoing.get("receiver")) in events.counterparts
        ):
            request_id = _identifier(outgoing.get("in_reply_to"))
            response = _message_text(_message_parts(row.outgoing))
            if conversation_id and request_id and response and trace_id and span_id:
                events.exits[(conversation_id, request_id)].append(
                    {
                        **base,
                        "agent_response": response,
                        "returned_downstream_request_id": _identifier(
                            incoming.get("in_reply_to")
                        ),
                        "response_source_path": f"{row.outgoing_path}.content.message[*].value",
                    }
                )
            elif (
                _identifier(outgoing.get("performative")).casefold() == "failure"
                and not response
            ):
                events.failures_without_text += 1
            else:
                events.malformed_exits += 1
    return events


def _join_fipa_turns(
    events: _FipaEvents, issues: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Соединить вход и финальный выход по (conversation_id, request_id)."""
    stats: Counter[str] = Counter()
    turns: list[dict[str, Any]] = []
    for key in sorted(set(events.entries) | set(events.exits)):
        conversation_id, request_id = key
        turn_id = f"{conversation_id}|{request_id}"
        entry: dict[str, Any] | None = None
        entry_conflicts: list[str] = []
        if key in events.entries:
            entry, entry_conflicts = _merge_replicas(
                events.entries[key],
                ("input_query", "route_label", "downstream_request_id"),
                latest=False,
            )
        exit_event: dict[str, Any] | None = None
        exit_conflicts: list[str] = []
        if key in events.exits:
            exit_event, exit_conflicts = _merge_replicas(
                events.exits[key],
                ("agent_response", "returned_downstream_request_id"),
                latest=True,
            )
        if entry_conflicts:
            stats["conflicting_entry_keys"] += 1
            issues.append(
                _issue(
                    "conflicting_entry_replicas",
                    schema_version=FIPA_SCHEMA,
                    turn_id=turn_id,
                    details="conflicting_fields=" + ",".join(entry_conflicts),
                )
            )
        if exit_conflicts:
            stats["conflicting_exit_keys"] += 1
            issues.append(
                _issue(
                    "conflicting_exit_replicas",
                    schema_version=FIPA_SCHEMA,
                    turn_id=turn_id,
                    details="conflicting_fields=" + ",".join(exit_conflicts),
                )
            )
        if entry_conflicts or exit_conflicts:
            continue
        if entry is None:
            stats["exit_without_entry"] += 1
            issues.append(
                _issue(
                    "exit_without_entry", schema_version=FIPA_SCHEMA, turn_id=turn_id
                )
            )
            continue
        if exit_event is None:
            stats["entry_without_exit"] += 1
            issues.append(
                _issue(
                    "entry_without_exit",
                    schema_version=FIPA_SCHEMA,
                    turn_id=turn_id,
                    trace_id=entry["trace_id"],
                    span_id=entry["span_id"],
                )
            )
            continue
        turn = _fipa_turn(turn_id, entry, exit_event, issues)
        if turn is None:
            continue
        stats["cross_trace_turns"] += int(not turn["same_trace"])
        stats["route_chain_complete_turns"] += int(turn["route_chain_complete"])
        turns.append(turn)
    stats["session_mismatches"] = sum(
        issue["issue_code"] == "session_id_mismatch" for issue in issues
    )
    return turns, stats


def _fipa_turn(
    turn_id: str,
    entry: dict[str, Any],
    exit_event: dict[str, Any],
    issues: list[dict[str, str]],
) -> dict[str, Any] | None:
    """Собрать turn из пары событий; None, если пара не публикуема."""
    session_id = entry["session_id"] or exit_event["session_id"]
    if not session_id:
        issues.append(
            _issue("session_id_missing", schema_version=FIPA_SCHEMA, turn_id=turn_id)
        )
        return None
    both_sessions = bool(entry["session_id"] and exit_event["session_id"])
    same_session = both_sessions and entry["session_id"] == exit_event["session_id"]
    if both_sessions and not same_session:
        issues.append(
            _issue("session_id_mismatch", schema_version=FIPA_SCHEMA, turn_id=turn_id)
        )
        return None
    if not both_sessions:
        issues.append(
            _issue(
                "session_id_partial",
                severity="warning",
                schema_version=FIPA_SCHEMA,
                turn_id=turn_id,
            )
        )
    entry_time = entry["time_ns"]
    exit_time = exit_event["time_ns"]
    if entry_time is None or exit_time is None:
        issues.append(
            _issue(
                "turn_timestamp_missing", schema_version=FIPA_SCHEMA, turn_id=turn_id
            )
        )
        return None
    latency_ms = (exit_time - entry_time) / 1_000_000.0
    if latency_ms < 0:
        issues.append(
            _issue("negative_turn_latency", schema_version=FIPA_SCHEMA, turn_id=turn_id)
        )
        return None
    return {
        "turn_id": turn_id,
        "session_id": session_id,
        "input_query": entry["input_query"],
        "agent_response": exit_event["agent_response"],
        "route_label": entry["route_label"],
        "schema_family": "fipa_acl",
        "schema_version": FIPA_SCHEMA,
        "entry_trace_id": entry["trace_id"],
        "exit_trace_id": exit_event["trace_id"],
        "entry_span_id": entry["span_id"],
        "exit_span_id": exit_event["span_id"],
        "entry_time_ns": entry_time,
        "exit_time_ns": exit_time,
        "same_trace": entry["trace_id"] == exit_event["trace_id"],
        "same_session": same_session,
        "route_chain_complete": bool(
            entry["downstream_request_id"]
            and entry["downstream_request_id"]
            == exit_event["returned_downstream_request_id"]
        ),
        "turn_latency_ms": latency_ms,
        "query_source_path": entry["query_source_path"],
        "response_source_path": exit_event["response_source_path"],
        "route_source_path": entry["route_source_path"],
    }


def _is_outer_boundary(record: dict[str, object]) -> bool:
    kind = _identifier(record.get("aef_kind")).casefold()
    return kind in _OUTER_KINDS or bool(
        _HTTP_BOUNDARY.search(_identifier(record.get("span_name")))
    )


def _collect_aef_turns(
    events: _FipaEvents, issues: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], int]:
    """Явные пары request/response на внешней границе non-FIPA trace.

    start_agent считается внешней границей только при отсутствии input_request
    или HTTP boundary в trace: внутренний агент не должен подменять внешний ответ.
    """

    outer_traces = set(events.traces) | {
        _identifier(record.get("trace_id"))
        for record in events.non_fipa_records
        if _is_outer_boundary(record)
    }
    turns: list[dict[str, Any]] = []
    incomplete = 0
    for record in events.non_fipa_records:
        trace_id = _identifier(record.get("trace_id"))
        kind = _identifier(record.get("aef_kind")).casefold()
        is_start_fallback = kind == "start_agent" and trace_id not in outer_traces
        if not (_is_outer_boundary(record) or is_start_fallback):
            continue
        events.traces.add(trace_id)
        turn = _aef_turn(record)
        if turn is None:
            incomplete += 1
            issues.append(
                _issue(
                    "boundary_pair_incomplete",
                    schema_version=AEF_BOUNDARY_SCHEMA,
                    trace_id=trace_id,
                    span_id=_identifier(record.get("span_id")),
                )
            )
            continue
        if not turn["session_id"]:
            incomplete += 1
            issues.append(
                _issue(
                    "session_id_missing",
                    schema_version=AEF_BOUNDARY_SCHEMA,
                    turn_id=turn["turn_id"],
                    trace_id=trace_id,
                    span_id=turn["entry_span_id"],
                )
            )
            continue
        turns.append(turn)
    return turns, incomplete


def _parent_boundary_turns(
    frame: pd.DataFrame,
    covered_traces: set[str],
    issues: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], int]:
    """Turn из start_agent и его непосредственной внешней AEF-границы."""
    turns: list[dict[str, Any]] = []
    incomplete = 0
    columns = list(frame.columns)
    for trace_id, group in frame.groupby("trace_id", sort=False):
        trace_id = _identifier(trace_id)
        if not trace_id or trace_id in covered_traces:
            continue
        records = [
            _row_record(row, columns)
            for row in group.itertuples(index=False, name=None)
        ]
        by_span_id = {
            _identifier(record.get("span_id")): record
            for record in records
            if _identifier(record.get("span_id"))
        }
        candidates = []
        has_topology = False
        for start in records:
            if _identifier(start.get("aef_kind")).casefold() != "start_agent":
                continue
            parent = by_span_id.get(_identifier(start.get("parent_span_id")))
            if parent is None or not _is_outer_boundary(parent):
                continue
            has_topology = True
            carrier_query, _ = _request_text(start.get("input_text"))
            incoming, incoming_path = _incoming_envelope(
                _mapping(parent.get("input_text"))
            )
            outgoing, outgoing_path = _outgoing_envelope(
                _mapping(parent.get("output_text"))
            )
            incoming_parts = _message_parts(incoming)
            boundary_query = _message_text(incoming_parts)
            label, label_index, query_parts = _split_label(incoming_parts)
            query = _message_text(query_parts)
            response = _message_text(_message_parts(outgoing))
            start_session = _identifier(start.get("session_id"))
            parent_session = _identifier(parent.get("session_id"))
            time_ns = _integer(parent.get("start_time_ns"))
            if not (
                carrier_query
                and _semantic(carrier_query) == _semantic(boundary_query)
                and query
                and response
                and _semantic(query) != _semantic(response)
                and start_session
                and start_session == parent_session
                and time_ns is not None
            ):
                continue
            span_id = _identifier(start.get("span_id"))
            candidates.append(
                {
                    "turn_id": f"parent:{trace_id}:{span_id}",
                    "session_id": start_session,
                    "input_query": query,
                    "agent_response": response,
                    "route_label": label,
                    "schema_family": "aef_parent_boundary",
                    "schema_version": AEF_PARENT_BOUNDARY_SCHEMA,
                    "entry_trace_id": trace_id,
                    "exit_trace_id": trace_id,
                    "entry_span_id": _identifier(parent.get("span_id")),
                    "exit_span_id": _identifier(parent.get("span_id")),
                    "entry_time_ns": time_ns,
                    "exit_time_ns": _integer(parent.get("end_time_ns")) or time_ns,
                    "same_trace": True,
                    "same_session": True,
                    "route_chain_complete": False,
                    "turn_latency_ms": 0.0,
                    "query_source_path": (
                        f"{incoming_path}.content.message[*].value"
                    ),
                    "response_source_path": (
                        f"{outgoing_path}.content.message[*].value"
                    ),
                    "route_source_path": (
                        f"{incoming_path}.content.message[{label_index}].value"
                        if label
                        else ""
                    ),
                }
            )
        variants = {
            (
                _semantic(candidate["input_query"]),
                _semantic(candidate["agent_response"]),
            )
            for candidate in candidates
        }
        if len(variants) == 1:
            turns.append(candidates[0])
        elif has_topology:
            incomplete += 1
            issues.append(
                _issue(
                    "parent_boundary_incomplete",
                    schema_version=AEF_PARENT_BOUNDARY_SCHEMA,
                    trace_id=trace_id,
                    details=f"candidate_variants={len(variants)}",
                )
            )
    return turns, incomplete


def _semantic_terminal_turns(
    frame: pd.DataFrame,
    covered_traces: set[str],
    route_labels: set[str],
) -> list[dict[str, Any]]:
    """Финальный ответ из явного top-level terminal-поля дочернего span."""
    turns = []
    for trace_id, group in frame.groupby("trace_id", sort=False):
        trace_id = _identifier(trace_id)
        if not trace_id or trace_id in covered_traces:
            continue
        starts = group.loc[
            _string_series(group["aef_kind"]).str.casefold().eq("start_agent")
        ].sort_values(["start_time_ns", "span_id"], kind="stable")
        carriers = []
        for record in starts.to_dict(orient="records"):
            query, query_path = _request_text(record.get("input_text"))
            if query:
                carriers.append((record, query, query_path))
        query_variants = {_semantic(query) for _, query, _ in carriers}
        if len(query_variants) != 1:
            continue
        start, carrier_query, query_path = carriers[0]
        session_id = _identifier(start.get("session_id"))
        if not session_id:
            continue
        route = ""
        query = carrier_query
        prefix, separator, remainder = carrier_query.partition(" ")
        if separator and prefix in route_labels and remainder.strip():
            route = prefix
            query = remainder.strip()

        answers: dict[str, tuple[str, str]] = {}
        for record in group.to_dict(orient="records"):
            if _identifier(record.get("session_id")) != session_id:
                continue
            payload = _mapping(record.get("output_text"))
            if payload is None:
                continue
            for key in _TERMINAL_ANSWER_KEYS:
                raw = _json_value(payload.get(key))
                answer = _text(raw)
                if not answer and isinstance(raw, list):
                    answer = next(
                        (_text(item) for item in raw if _text(item)), ""
                    )
                semantic = _semantic(answer)
                if semantic and semantic not in {
                    _semantic(carrier_query),
                    _semantic(query),
                }:
                    answers.setdefault(
                        semantic,
                        (
                            answer,
                            f"output_text.{key}",
                        ),
                    )
        if len(answers) != 1:
            continue
        answer, response_path = next(iter(answers.values()))
        span_id = _identifier(start.get("span_id"))
        time_ns = _integer(start.get("start_time_ns"))
        if not session_id or not span_id or time_ns is None:
            continue
        turns.append(
            {
                "turn_id": f"terminal:{trace_id}:{span_id}",
                "session_id": session_id,
                "input_query": query,
                "agent_response": answer,
                "route_label": route,
                "schema_family": "aef_semantic_terminal",
                "schema_version": AEF_SEMANTIC_TERMINAL_SCHEMA,
                "entry_trace_id": trace_id,
                "exit_trace_id": trace_id,
                "entry_span_id": span_id,
                "exit_span_id": span_id,
                "entry_time_ns": time_ns,
                "exit_time_ns": time_ns,
                "same_trace": True,
                "same_session": True,
                "route_chain_complete": False,
                "turn_latency_ms": 0.0,
                "query_source_path": query_path,
                "response_source_path": response_path,
                "route_source_path": query_path if route else "",
            }
        )
    return turns


def _state_json_turns(
    frame: pd.DataFrame, covered_traces: set[str]
) -> tuple[list[dict[str, Any]], int]:
    """Turn'ы langgraph-агентов: диалог лежит в state_json внутри input_text.

    Финальное состояние — спан со stage exit/__end__: вопрос — последняя
    user-реплика messages, ответ — message_to_user, маршрут — product_agent.
    Дополняет только трейсы, не покрытые FIPA/AEF, чтобы не дублировать turn'ы.
    """
    exits: dict[str, dict[str, Any]] = {}
    inputs = frame["input_text"].astype(str)
    outputs = frame["output_text"].astype(str)
    likely = (
        inputs.str.contains('"message_to_user"', regex=False)
        & inputs.str.contains('"stage"', regex=False)
    ) | (
        outputs.str.contains('"message_to_user"', regex=False)
        & outputs.str.contains('"stage"', regex=False)
    )
    for record in frame.loc[likely].to_dict(orient="records"):
        trace_id = _identifier(record.get("trace_id"))
        if not trace_id or trace_id in covered_traces:
            continue
        state = None
        source = ""
        for candidate_source in ("output_text", "input_text"):
            candidate = _mapping(_json_value(record.get(candidate_source)))
            if candidate is not None and _identifier(candidate.get("stage")).casefold() in {
                "exit",
                "__end__",
            }:
                state = candidate
                source = candidate_source
                break
        if state is None:
            continue
        answer = _text(state.get("message_to_user"))
        query = _typed_message(state, {"user", "human"})
        if not answer or not query or _semantic(answer) == _semantic(query):
            continue
        time_ns = _integer(record.get("start_time_ns")) or 0
        current = exits.get(trace_id)
        if current is not None and current["entry_time_ns"] >= time_ns:
            continue
        route = _text(state.get("product_agent"))
        span_id = _identifier(record.get("span_id"))
        exits[trace_id] = {
            "turn_id": f"state:{trace_id}:{span_id}",
            "session_id": _identifier(record.get("session_id")),
            "input_query": query,
            "agent_response": answer,
            "route_label": route,
            "schema_family": "state_json",
            "schema_version": STATE_JSON_SCHEMA,
            "entry_trace_id": trace_id,
            "exit_trace_id": trace_id,
            "entry_span_id": span_id,
            "exit_span_id": span_id,
            "entry_time_ns": time_ns,
            "exit_time_ns": time_ns,
            "same_trace": True,
            "same_session": True,
            "route_chain_complete": False,
            "turn_latency_ms": 0.0,
            "query_source_path": f"{source}.messages[last user].content",
            "response_source_path": f"{source}.message_to_user",
            "route_source_path": f"{source}.product_agent" if route else "",
        }
    turns = list(exits.values())
    missing_sessions = sum(not turn["session_id"] for turn in turns)
    return [turn for turn in turns if turn["session_id"]], missing_sessions


def _ordered_turns(turns: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(turns, columns=TURN_COLUMNS)
    if frame.empty:
        return frame
    order_time = pd.to_numeric(frame["entry_time_ns"], errors="coerce").fillna(0)
    return (
        frame.assign(_order_time=order_time)
        .sort_values(["session_id", "_order_time", "turn_id"], kind="stable")
        .drop(columns="_order_time")
        .reset_index(drop=True)
    )


def extract_turns(
    spans: pd.DataFrame,
    config: ExtractionConfig | None = None,
) -> ExtractionResult:
    """Извлечь полные диалоговые turn и доказательства из таблицы spans."""
    if not isinstance(spans, pd.DataFrame):
        raise TypeError("spans должен быть pandas.DataFrame")
    if spans.empty:
        raise TraceExtractionError("Выгрузка трейсов пуста")
    config = config or ExtractionConfig()
    required = {
        "trace_id",
        "span_id",
        "aef_kind",
        "span_name",
        "input_text",
        "output_text",
        "agent_id",
        "session_id",
        "start_time_ns",
    }
    missing = sorted(required - set(spans.columns))
    if missing:
        raise TraceExtractionError(
            f"raw spans не содержит обязательные колонки: {missing}"
        )

    frame, agent_id, dropped_rows = _filter_agent(spans, config.agent_id)
    trace_values = _string_series(frame["trace_id"])
    candidates = frame.loc[_candidate_mask(frame)]
    issues: list[dict[str, str]] = []

    events = _collect_fipa_events(candidates)
    fipa_keys = len(set(events.entries) | set(events.exits))
    fipa_turns, fipa_stats = _join_fipa_turns(events, issues)
    aef_turns, incomplete_boundaries = _collect_aef_turns(events, issues)

    covered_traces = {
        _identifier(turn["entry_trace_id"]) for turn in fipa_turns + aef_turns
    } | {_identifier(turn["exit_trace_id"]) for turn in fipa_turns + aef_turns}
    parent_turns, incomplete_parent_boundaries = _parent_boundary_turns(
        frame, covered_traces, issues
    )
    covered_traces |= {
        _identifier(turn["entry_trace_id"]) for turn in parent_turns
    }
    route_labels = {
        _identifier(turn["route_label"])
        for turn in fipa_turns + parent_turns
        if _identifier(turn["route_label"])
    }
    terminal_turns = _semantic_terminal_turns(
        frame, covered_traces, route_labels
    )
    covered_traces |= {
        _identifier(turn["entry_trace_id"]) for turn in terminal_turns
    }
    state_turns, state_session_missing = _state_json_turns(frame, covered_traces)
    state_traces = {turn["entry_trace_id"] for turn in state_turns}

    fallback_turns = parent_turns + terminal_turns + state_turns
    resolved_boundary_traces = {
        _identifier(turn["entry_trace_id"]) for turn in fallback_turns
    }
    resolved_boundary_issues = [
        issue
        for issue in issues
        if issue["issue_code"] == "boundary_pair_incomplete"
        and issue["trace_id"] in resolved_boundary_traces
    ]
    if resolved_boundary_issues:
        issues = [
            issue for issue in issues if issue not in resolved_boundary_issues
        ]
        incomplete_boundaries -= len(resolved_boundary_issues)
    fallback_queries = {
        (turn["entry_trace_id"], _semantic(turn["input_query"]))
        for turn in fallback_turns
    }
    fallback_answers = {
        (turn["exit_trace_id"], _semantic(turn["agent_response"]))
        for turn in fallback_turns
    }
    resolved_entry_ids = {
        f"{conversation_id}|{request_id}"
        for (conversation_id, request_id), entries in events.entries.items()
        if (conversation_id, request_id) not in events.exits
        and any(
            (entry["trace_id"], _semantic(entry["input_query"]))
            in fallback_queries
            for entry in entries
        )
    }
    resolved_exit_ids = {
        f"{conversation_id}|{request_id}"
        for (conversation_id, request_id), exits in events.exits.items()
        if (conversation_id, request_id) not in events.entries
        and any(
            (exit_event["trace_id"], _semantic(exit_event["agent_response"]))
            in fallback_answers
            for exit_event in exits
        )
    }
    resolved_fipa_ids = resolved_entry_ids | resolved_exit_ids
    if resolved_fipa_ids:
        issues = [
            issue
            for issue in issues
            if not (
                issue["issue_code"]
                in {"entry_without_exit", "exit_without_entry"}
                and issue["turn_id"] in resolved_fipa_ids
            )
        ]
        fipa_keys -= len(resolved_fipa_ids)
        fipa_stats["entry_without_exit"] -= len(resolved_entry_ids)
        fipa_stats["exit_without_entry"] -= len(resolved_exit_ids)

    candidate_traces = set(_string_series(candidates["trace_id"]))
    unrecognized = set(trace_values) - events.traces - state_traces
    no_boundary_trace_ids = sorted(unrecognized - candidate_traces)
    unsupported_trace_ids = sorted(unrecognized & candidate_traces)
    for trace_id in no_boundary_trace_ids[: config.max_issue_examples]:
        issues.append(
            _issue(
                "no_boundary_span",
                severity="warning",
                trace_id=trace_id,
                details="trace has no input_request/start_agent/HTTP boundary span",
            )
        )
    for trace_id in unsupported_trace_ids[: config.max_issue_examples]:
        issues.append(
            _issue(
                "unsupported_trace_schema",
                trace_id=trace_id,
                details="boundary span present but no supported FIPA or AEF pair",
            )
        )
    turns = _ordered_turns(
        fipa_turns + aef_turns + parent_turns + terminal_turns + state_turns
    )
    issue_frame = pd.DataFrame(issues, columns=_ISSUE_COLUMNS)
    candidate_turn_keys = (
        fipa_keys
        + events.malformed_entries
        + events.malformed_exits
        + events.failures_without_text
        + len(aef_turns)
        + incomplete_boundaries
        + len(parent_turns)
        + len(terminal_turns)
        + len(state_turns)
        + state_session_missing
    )
    complete_turns = len(turns)
    report = {
        "contract_version": EXTRACTION_CONTRACT,
        "agent_id": agent_id,
        "input_rows": int(len(spans)),
        **dropped_rows,
        "input_trace_count": int(trace_values.nunique()),
        "candidate_rows_scanned": int(len(candidates)),
        "fipa_rows": int(events.rows),
        "fipa_turn_keys": int(fipa_keys),
        "fipa_complete_turns": int(len(fipa_turns)),
        "fipa_extraction_coverage": len(fipa_turns) / fipa_keys if fipa_keys else 0.0,
        "aef_candidate_boundaries": int(len(aef_turns) + incomplete_boundaries),
        "aef_complete_turns": int(len(aef_turns)),
        "parent_boundary_candidate_turns": int(
            len(parent_turns) + incomplete_parent_boundaries
        ),
        "parent_boundary_complete_turns": int(len(parent_turns)),
        "semantic_terminal_complete_turns": int(len(terminal_turns)),
        "state_json_candidate_turns": int(len(state_turns) + state_session_missing),
        "state_json_complete_turns": int(len(state_turns)),
        "state_json_session_missing": int(state_session_missing),
        "candidate_turn_keys": int(candidate_turn_keys),
        "complete_turns": int(complete_turns),
        "extraction_coverage": (
            complete_turns / candidate_turn_keys if candidate_turn_keys else 0.0
        ),
        "entry_without_exit": int(fipa_stats["entry_without_exit"]),
        "exit_without_entry": int(fipa_stats["exit_without_entry"]),
        "duplicate_entry_keys": int(sum(len(v) > 1 for v in events.entries.values())),
        "duplicate_exit_keys": int(sum(len(v) > 1 for v in events.exits.values())),
        "conflicting_entry_keys": int(fipa_stats["conflicting_entry_keys"]),
        "conflicting_exit_keys": int(fipa_stats["conflicting_exit_keys"]),
        "malformed_entry_rows": int(events.malformed_entries),
        "malformed_exit_rows": int(events.malformed_exits),
        "failures_without_text": int(events.failures_without_text),
        "counterparts": list(events.counterparts),
        "counterpart_conversations": dict(events.counterparts),
        "incomplete_boundary_rows": int(incomplete_boundaries),
        "cross_trace_turns": int(fipa_stats["cross_trace_turns"]),
        "session_mismatches": int(fipa_stats["session_mismatches"]),
        "route_chain_complete_turns": int(fipa_stats["route_chain_complete_turns"]),
        "supported_trace_count": int(len(events.traces)),
        "no_boundary_trace_count": int(len(no_boundary_trace_ids)),
        "unsupported_trace_count": int(len(unsupported_trace_ids)),
        "unsupported_rows": int(trace_values.isin(unsupported_trace_ids).sum()),
        "strategies": dict(sorted(Counter(turns["schema_version"].tolist()).items())),
        "turn_key_policy": "FIPA conversation_id+reply_with/in_reply_to; AEF trace_id+span_id",
        "order_policy": "session_id,entry_time_ns,turn_id",
        "issue_rows": int(len(issue_frame)),
        "issue_examples_truncated": max(
            len(unsupported_trace_ids), len(no_boundary_trace_ids)
        )
        > config.max_issue_examples,
    }
    logger.info(
        "Dialogue extraction complete: turns=%d, candidates=%d, FIPA=%d, AEF=%d",
        complete_turns,
        candidate_turn_keys,
        len(fipa_turns),
        len(aef_turns),
    )
    return ExtractionResult(turns=turns, issues=issue_frame, report=report)
