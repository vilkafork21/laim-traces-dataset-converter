"""Извлечение turn из spans AEF-трейсинга по стандартам, без схем под агента.

Граница обращения — внешний граничный спан по `aef_kind` (input_request,
kafka_consume, start_agent) или спан с FIPA-конвертом; текст берётся проекцией
тела (`bodies`), асинхронные ответы соединяются по correlation key (`fipa`).
Всё, что не доказано, считается и объясняется в отчёте, а не угадывается.
"""

from __future__ import annotations

import logging
import re
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from urllib.parse import quote

import pandas as pd

import fipa
from bodies import MALFORMED, Leaf, Projection, envelopes, parse_body, project_body
from fipa import FIPA_SCHEMA, ISSUE_COLUMNS, issue
from shadow_extraction import ShadowCollector

logger = logging.getLogger(__name__)

EXTRACTION_CONTRACT = "laim-trace-turn-extraction.v3"
AEF_BOUNDARY_SCHEMA = "aef_boundary_v1"
BOUNDARY_KINDS = ("input_request", "kafka_consume", "start_agent")
_FIPA_MARKER = re.compile(r"\b(?:conversation_id|reply_with|in_reply_to|message_id)\b")
_INDEX = re.compile(r"\[\d+\]")

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

_REQUIRED_COLUMNS = {
    "trace_id",
    "span_id",
    "aef_kind",
    "input_text",
    "output_text",
    "agent_id",
    "session_id",
    "start_time_ns",
}


class TraceExtractionError(ValueError):
    """Таблица трейсов или конфигурация извлечения некорректна."""


@dataclass(frozen=True)
class ExtractionConfig:
    agent_id: str = ""
    max_issue_examples: int = 100

    def __post_init__(self) -> None:
        if type(self.max_issue_examples) is not int or self.max_issue_examples < 1:
            raise TraceExtractionError("max_issue_examples должен быть целым >= 1")


@dataclass(frozen=True)
class ExtractionResult:
    """Полные turn, причины непубликации и агрегированные доказательства."""

    turns: pd.DataFrame
    issues: pd.DataFrame
    report: dict[str, object]


@dataclass(frozen=True)
class ParsedSpan:
    """Кандидат: спан с разобранными телами и найденными конвертами."""

    trace_id: str
    span_id: str
    parent_span_id: str
    session_id: str
    time_ns: int | None
    aef_kind: str
    request: object
    response: object
    incoming: tuple[str, dict[str, object]] | None
    outgoing: tuple[str, dict[str, object]] | None
    envelopes_ambiguous: bool = False

    @property
    def has_envelope(self) -> bool:
        return self.incoming is not None or self.outgoing is not None


@dataclass(frozen=True)
class _Pair:
    """Синхронная пара запрос/ответ внешней границы."""

    span: ParsedSpan
    query: Projection
    answer: Projection
    query_prefix: str
    request_body: object


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
    if isinstance(value, pd.Timestamp):
        return value.value if not pd.isna(value) else None
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if isinstance(value, float) and (not value.is_integer() or abs(value) > 2**53):
        return None
    return number


def _string_series(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip()


def candidate_mask(frame: pd.DataFrame) -> pd.Series:
    """Граничные спаны по aef_kind и строки с признаком FIPA-конверта (предфильтр)."""
    kinds = _string_series(frame["aef_kind"]).str.casefold()
    input_marker = (
        frame["input_text"].astype("string").str.contains(_FIPA_MARKER, na=False)
    )
    output_marker = (
        frame["output_text"].astype("string").str.contains(_FIPA_MARKER, na=False)
    )
    return kinds.isin(BOUNDARY_KINDS) | input_marker | output_marker


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


def _single_envelope(
    body: object, column: str
) -> tuple[tuple[str, dict[str, object]] | None, bool]:
    if body is None or body is MALFORMED:
        return None, False
    found = envelopes(body)
    if len(found) == 1:
        return (f"{column}{found[0][0]}", found[0][1]), False
    return None, len(found) > 1


def _parse_spans(candidates: pd.DataFrame) -> list[ParsedSpan]:
    spans = []
    columns = list(candidates.columns)
    has_parent = "parent_span_id" in columns
    for raw in candidates.itertuples(index=False, name=None):
        record = dict(zip(columns, raw, strict=True))
        request = parse_body(record.get("input_text"))
        response = parse_body(record.get("output_text"))
        if isinstance(request, str):
            request = record["input_text"]
        if isinstance(response, str):
            response = record["output_text"]
        incoming, incoming_ambiguous = _single_envelope(request, "input_text")
        outgoing, outgoing_ambiguous = _single_envelope(response, "output_text")
        kind = _identifier(record.get("aef_kind")).casefold()
        if (
            kind not in BOUNDARY_KINDS
            and incoming is None
            and outgoing is None
            and not (incoming_ambiguous or outgoing_ambiguous)
        ):
            continue
        spans.append(
            ParsedSpan(
                trace_id=_identifier(record.get("trace_id")),
                span_id=_identifier(record.get("span_id")),
                parent_span_id=_identifier(record.get("parent_span_id"))
                if has_parent
                else "",
                session_id=_identifier(record.get("session_id")),
                time_ns=_integer(record.get("start_time_ns")),
                aef_kind=kind,
                request=request,
                response=response,
                incoming=incoming,
                outgoing=outgoing,
                envelopes_ambiguous=incoming_ambiguous or outgoing_ambiguous,
            )
        )
    return spans


def _ancestor_candidate(
    span: ParsedSpan, own: dict[str, ParsedSpan], parents: dict[tuple[str, str], str]
) -> ParsedSpan | None:
    seen: set[str] = set()
    current = span.parent_span_id
    while current and current not in seen:
        seen.add(current)
        if current in own:
            return own[current]
        current = parents.get((span.trace_id, current), "")
    return None


def _split_boundaries(
    spans: list[ParsedSpan], parents: dict[tuple[str, str], str]
) -> tuple[dict[str, list[ParsedSpan]], dict[str, list[ParsedSpan]]]:
    """Внешние кандидаты каждого trace и их внутренние потомки.

    Внутренним считается потомок другого кандидата. При неполной вложенности
    транспортная граница имеет приоритет над start_agent.
    """
    outer: dict[str, list[ParsedSpan]] = defaultdict(list)
    inner: dict[str, list[ParsedSpan]] = defaultdict(list)
    by_trace: dict[str, list[ParsedSpan]] = defaultdict(list)
    for span in spans:
        by_trace[span.trace_id].append(span)
    for trace_id, items in by_trace.items():
        own = {span.span_id: span for span in items}
        tops = [
            span for span in items if _ancestor_candidate(span, own, parents) is None
        ]
        if any(span.has_envelope for span in tops):
            tops = [
                span
                for span in tops
                if span.has_envelope or span.aef_kind != "start_agent"
            ]
        kinds = {span.aef_kind for span in tops if span.aef_kind in BOUNDARY_KINDS}
        if len(kinds) > 1:
            best = min(kinds, key=BOUNDARY_KINDS.index)
            tops = [
                span
                for span in tops
                if span.aef_kind not in BOUNDARY_KINDS or span.aef_kind == best
            ]
        top_ids = {span.span_id for span in tops}
        outer[trace_id] = tops
        inner[trace_id] = [span for span in items if span.span_id not in top_ids]
    return outer, inner


def _descendant_request(
    span: ParsedSpan, descendants: list[ParsedSpan], parents: dict[tuple[str, str], str]
) -> ParsedSpan | None:
    """Запрос ближайшего граничного потомка той же сессии, если он единственный."""
    ranked: dict[int, list[ParsedSpan]] = defaultdict(list)
    for candidate in descendants:
        if (
            candidate.aef_kind not in BOUNDARY_KINDS
            or candidate.session_id != span.session_id
            or project_body(candidate.request, side="request").status == "empty"
        ):
            continue
        depth = 0
        current = candidate.parent_span_id
        seen: set[str] = set()
        while current and current not in seen and current != span.span_id:
            seen.add(current)
            depth += 1
            current = parents.get((span.trace_id, current), "")
        if current == span.span_id:
            ranked[depth].append(candidate)
    if not ranked:
        return None
    nearest = ranked[min(ranked)]
    return nearest[0] if len(nearest) == 1 else None


def _route(span: ParsedSpan, span_with_request: ParsedSpan) -> str:
    """Путь синхронного кандидата: fipa / forward / sync."""
    incoming = span_with_request.incoming
    if incoming is not None:
        sender = _identifier(incoming[1].get("sender"))
        if not fipa.is_request(incoming[1]) or (
            sender and sender == _identifier(incoming[1].get("receiver"))
        ):
            return "fipa"
    if span.outgoing is not None:
        envelope = span.outgoing[1]
        if fipa.is_request(envelope) or fipa.is_agreement(envelope):
            return "forward"
        if incoming is not None or not fipa.is_reply(envelope):
            return "fipa"
    elif (
        incoming is not None
        and project_body(span.response, side="response").status == "empty"
    ):
        return "fipa"
    return "sync"


def _protocol_keys(span: ParsedSpan) -> set[tuple[str, str]]:
    incoming = span.incoming[1] if span.incoming else {}
    outgoing = span.outgoing[1] if span.outgoing else {}
    keys = set()
    for envelope, field, recognized in (
        (incoming, "reply_with", fipa.is_request(incoming)),
        (outgoing, "in_reply_to", fipa.is_reply(outgoing)),
    ):
        conversation = _identifier(envelope.get("conversation_id")) or _identifier(
            incoming.get("conversation_id")
        )
        request = _identifier(envelope.get(field))
        if recognized and conversation and request:
            keys.add((conversation, request))
    return keys


def _pair(span: ParsedSpan, source: ParsedSpan) -> _Pair:
    prefix = "" if source is span else f"descendant:{source.span_id}:"
    query = project_body(source.request, side="request")
    answer = project_body(span.response, side="response")
    return _Pair(span, query, answer, prefix, source.request)


def _pair_failure(pair: _Pair) -> str:
    if pair.query.status != "text":
        return {"malformed": "request_malformed", "ambiguous": "request_ambiguous"}.get(
            pair.query.status, "request_missing"
        )
    if pair.answer.status != "text":
        return {
            "malformed": "response_malformed",
            "ambiguous": "response_ambiguous",
        }.get(pair.answer.status, "response_missing")
    if not pair.span.session_id:
        return "session_id_missing"
    if pair.span.time_ns is None:
        return "timestamp_missing"
    return ""


def _sync_turn(pair: _Pair) -> dict[str, object]:
    span = pair.span
    return {
        "turn_id": f"aef:{quote(span.trace_id, safe='')}:{quote(span.span_id, safe='')}",
        "session_id": span.session_id,
        "input_query": pair.query.text,
        "agent_response": pair.answer.text,
        "route_label": "",
        "schema_family": "aef_boundary",
        "schema_version": AEF_BOUNDARY_SCHEMA,
        "entry_trace_id": span.trace_id,
        "exit_trace_id": span.trace_id,
        "entry_span_id": span.span_id,
        "exit_span_id": span.span_id,
        "entry_time_ns": span.time_ns,
        "exit_time_ns": span.time_ns,
        "same_trace": True,
        "same_session": True,
        "route_chain_complete": False,
        "turn_latency_ms": 0.0,
        "query_source_path": f"{pair.query_prefix}input_text{pair.query.path}",
        "response_source_path": f"output_text{pair.answer.path}",
        "route_source_path": "",
    }


def _general_path(column: str, leaf: Leaf) -> str:
    return _INDEX.sub("[*]", f"{column}{leaf.path}")


def _ordered_turns(turns: list[dict[str, object]]) -> pd.DataFrame:
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


def _top(counter: Counter[str], limit: int) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit])


_COLUMN_ALIASES = {
    "traceid": "trace_id",
    "spanid": "span_id",
    "parentspanid": "parent_span_id",
    "name": "span_name",
    "starttimeunixnano": "start_time_ns",
    "endtimeunixnano": "end_time_ns",
}


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if not frame.columns.is_unique:
        raise TraceExtractionError("raw spans содержит повторяющиеся имена колонок")
    for alias, column in _COLUMN_ALIASES.items():
        if alias not in frame:
            continue
        if column in frame:
            if not frame[alias].equals(frame[column]):
                raise TraceExtractionError(
                    f"Колонки {alias} и {column} противоречат друг другу"
                )
        else:
            frame = frame.rename(columns={alias: column})
    return frame


def _unique_spans(
    frame: pd.DataFrame, issues: list[dict[str, str]]
) -> tuple[pd.DataFrame, dict[str, int]]:
    identities = frame[["trace_id", "span_id"]].apply(_string_series)
    invalid = identities.eq("").any(axis=1)
    counts = {
        "invalid_span_rows": int(invalid.sum()),
        "duplicate_span_rows": 0,
        "conflicting_span_rows": 0,
    }
    for trace_id, span_id in identities.loc[invalid].itertuples(index=False, name=None):
        issues.append(
            issue(
                "span_identity_missing",
                trace_id=trace_id,
                span_id=span_id,
                details="trace_id и span_id должны быть непустыми",
            )
        )
    frame = frame.loc[~invalid].copy()
    frame[["trace_id", "span_id"]] = identities.loc[~invalid]
    repeated = frame.duplicated(["trace_id", "span_id"], keep=False)
    rejected = []
    fields = [
        column
        for column in (
            "session_id",
            "parent_span_id",
            "start_time_ns",
            "end_time_ns",
            "aef_kind",
            "input_text",
            "output_text",
        )
        if column in frame
    ]
    for (trace_id, span_id), group in frame.loc[repeated].groupby(
        ["trace_id", "span_id"], sort=False
    ):
        values = group[fields].map(_identifier)
        for column in ("input_text", "output_text"):
            values[column] = group[column].map(
                lambda value: json.dumps(
                    parse_body(value), ensure_ascii=False, sort_keys=True, default=str
                )
            )
        if len(values.drop_duplicates()) == 1:
            rejected.extend(group.index[1:])
            counts["duplicate_span_rows"] += len(group) - 1
        else:
            rejected.extend(group.index)
            counts["conflicting_span_rows"] += len(group)
            issues.append(
                issue(
                    "conflicting_span_replicas",
                    trace_id=trace_id,
                    span_id=span_id,
                    details=f"Несовместимые копии спана: {len(group)}",
                )
            )
    return frame.drop(index=rejected), counts


def extract_turns(
    spans: pd.DataFrame, config: ExtractionConfig, *, shadow: ShadowCollector | None = None,
) -> ExtractionResult:
    """Извлечь внешние обращения; неоднозначные и неполные события оставить в диагностике."""
    if not isinstance(spans, pd.DataFrame):
        raise TraceExtractionError("spans должен быть pandas.DataFrame")
    if spans.empty:
        raise TraceExtractionError("Выгрузка трейсов пуста")
    frame = _normalize_columns(spans).reset_index(drop=True)
    missing = sorted(_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise TraceExtractionError(
            f"raw spans не содержит обязательные колонки: {missing}"
        )
    frame, agent_id, dropped_rows = _filter_agent(frame, config.agent_id)
    selected_rows = len(frame)
    trace_values = set(_string_series(frame["trace_id"])) - {""}
    issues: list[dict[str, str]] = []
    parent_chain_available = "parent_span_id" in frame
    parents = (
        {
            (trace_id, span_id): parent
            for trace_id, span_id, parent in frame[
                ["trace_id", "span_id", "parent_span_id"]
            ]
            .apply(_string_series)
            .itertuples(index=False, name=None)
        }
        if parent_chain_available
        else {}
    )
    frame, span_counts = _unique_spans(frame, issues)
    blocked = {
        (item["trace_id"], item["span_id"]): "ancestor_span_conflict"
        for item in issues
        if item["issue_code"] == "conflicting_span_replicas"
    }
    parsed = _parse_spans(frame.loc[candidate_mask(frame)])
    blocked.update(
        {
            (span.trace_id, span.span_id): "ancestor_envelopes_ambiguous"
            for span in parsed
            if span.envelopes_ambiguous
        }
    )
    pairs: list[_Pair] = []
    fipa_spans: list[ParsedSpan] = []
    inner_rows = 0
    ambiguous_envelopes = 0
    blocked_boundaries = 0

    def rejected(span: ParsedSpan) -> bool:
        nonlocal ambiguous_envelopes, blocked_boundaries
        if span.envelopes_ambiguous:
            ambiguous_envelopes += 1
            issues.append(
                issue(
                    "envelopes_ambiguous", trace_id=span.trace_id, span_id=span.span_id
                )
            )
            return True
        current = span.parent_span_id
        seen = {span.span_id}
        while current:
            if (span.trace_id, current) in blocked or current in seen:
                blocked_boundaries += 1
                code = (
                    blocked[(span.trace_id, current)]
                    if current not in seen
                    else "parent_cycle"
                )
                issues.append(issue(code, trace_id=span.trace_id, span_id=span.span_id))
                return True
            seen.add(current)
            current = parents.get((span.trace_id, current), "")
        return False

    eligible = [span for span in parsed if not rejected(span)]
    outer, inner = _split_boundaries(eligible, parents)
    for trace_id, tops in outer.items():
        descendants = inner[trace_id]
        boundaries = {span.span_id: span for span in tops}
        synchronous: dict[str, ParsedSpan] = {}
        for span in tops:
            if (
                span.aef_kind == "start_agent"
                and not span.has_envelope
                and (span.parent_span_id or not parent_chain_available)
            ):
                blocked_boundaries += 1
                issues.append(
                    issue(
                        "start_agent_not_root",
                        trace_id=span.trace_id,
                        span_id=span.span_id,
                        details="Для start_agent не подтверждён корень: нужен явно пустой parent_span_id",
                    )
                )
                continue
            source = (
                (_descendant_request(span, descendants, parents) or span)
                if span.request is None
                else span
            )
            if _route(span, source) == "sync":
                pairs.append(_pair(span, source))
                synchronous[span.span_id] = span
            else:
                incoming = source.incoming
                if source is not span and incoming is not None:
                    incoming = (
                        f"descendant:{source.span_id}:{incoming[0]}",
                        incoming[1],
                    )
                fipa_spans.append(
                    replace(span, request=source.request, incoming=incoming)
                )
        for span in descendants:
            ancestor = _ancestor_candidate(span, boundaries, parents)
            if (
                span.has_envelope
                and ancestor is not None
                and _ancestor_candidate(span, synchronous, parents) is None
                and _protocol_keys(span) & _protocol_keys(ancestor)
            ):
                fipa_spans.append(span)
            else:
                inner_rows += 1

    counterparts = fipa.derive_counterparts(fipa_spans)
    shadow_events = fipa.Events() if shadow is not None else None
    events = fipa.collect_events(fipa_spans, counterparts, issues, shadow=shadow_events)
    fipa_turns, fipa_stats = fipa.join_turns(events, issues)
    if shadow is not None and shadow_events is not None:
        for key in sorted(set(shadow_events.entries) | set(shadow_events.exits)):
            entries = shadow_events.entries.get(key, [])
            exits = shadow_events.exits.get(key, [])
            if len(entries) != 1 or len(exits) != 1:
                shadow.counts["fipa_non_unique_or_missing_events"] += 1
                continue
            entry, exit_event = entries[0], exits[0]
            paired, _ = fipa.join_turns(
                fipa.Events(entries={key: entries}, exits={key: exits}), [],
            )
            if not paired:
                shadow.counts["fipa_invalid_boundary"] += 1
                continue
            shadow.observe(
                paired[0], entry["projection"], exit_event["projection"],
                entry["body"], exit_event["body"],
            )
    fipa_keys = len(set(events.entries) | set(events.exits))
    unregistered = [
        span
        for span in fipa_spans
        if (span.trace_id, span.span_id) not in events.registered
    ]
    forwarded_receivers: Counter[str] = Counter()
    for span in unregistered:
        receiver = fipa.route_label(span.outgoing)[0]
        forwarded_receivers[receiver] += 1
        code = "forwarded" if receiver else "fipa_boundary_unresolved"
        issues.append(
            issue(
                code,
                schema_version=FIPA_SCHEMA,
                trace_id=span.trace_id,
                span_id=span.span_id,
                details=f"Получатель: {receiver}"
                if receiver
                else "Не установлены внешняя сторона или конечный ответ",
            )
        )

    sync_turns: list[dict[str, object]] = []
    incomplete = ambiguous_rows = 0
    request_candidates: Counter[str] = Counter()
    response_candidates: Counter[str] = Counter()
    for pair in pairs:
        if shadow is not None:
            shadow.observe(
                _sync_turn(pair), pair.query, pair.answer,
                pair.request_body, pair.span.response,
            )
        reason = _pair_failure(pair)
        if not reason:
            sync_turns.append(_sync_turn(pair))
            continue
        incomplete += 1
        ambiguous_rows += reason in {"request_ambiguous", "response_ambiguous"}
        request_candidates.update(
            _general_path("input_text", leaf) for leaf in pair.query.candidates
        )
        response_candidates.update(
            _general_path("output_text", leaf) for leaf in pair.answer.candidates
        )
        code = (
            "session_id_missing"
            if reason == "session_id_missing"
            else "boundary_pair_incomplete"
        )
        issues.append(
            issue(
                code,
                schema_version=AEF_BOUNDARY_SCHEMA,
                trace_id=pair.span.trace_id,
                span_id=pair.span.span_id,
                details=reason,
            )
        )

    candidate_traces = {span.trace_id for span in parsed}
    no_boundary_trace_ids = sorted(trace_values - candidate_traces)
    for trace_id in no_boundary_trace_ids:
        issues.append(
            issue(
                "no_boundary_span",
                severity="warning",
                trace_id=trace_id,
                details="Нет внешнего граничного спана или FIPA-конверта",
            )
        )
    turns = _ordered_turns(fipa_turns + sync_turns)
    issue_counts = Counter(item["issue_code"] for item in issues)
    issue_frame = pd.DataFrame(
        issues[: config.max_issue_examples], columns=ISSUE_COLUMNS
    )
    candidate_turn_keys = (
        fipa_keys
        + events.malformed_entries
        + events.malformed_exits
        + events.failures_without_text
        + len(pairs)
        + len(unregistered)
        + ambiguous_envelopes
        + blocked_boundaries
    )
    complete_turns = len(turns)
    report = {
        "contract_version": EXTRACTION_CONTRACT,
        "agent_id": agent_id,
        "input_rows": len(spans),
        "selected_agent_rows": selected_rows,
        **dropped_rows,
        **span_counts,
        "input_trace_count": len(trace_values),
        "candidate_rows_scanned": len(parsed),
        "non_candidate_rows": len(frame) - len(parsed),
        "parent_chain_available": parent_chain_available,
        "inner_boundary_rows": inner_rows,
        "fipa_rows": len(fipa_spans),
        "fipa_turn_keys": fipa_keys,
        "fipa_complete_turns": len(fipa_turns),
        "fipa_extraction_coverage": len(fipa_turns) / fipa_keys if fipa_keys else 0.0,
        "aef_candidate_boundaries": len(pairs),
        "aef_complete_turns": len(sync_turns),
        "forwarded_rows": sum(
            bool(fipa.route_label(span.outgoing)[0]) for span in unregistered
        ),
        "unresolved_fipa_rows": len(unregistered),
        "ambiguous_envelope_rows": ambiguous_envelopes,
        "blocked_boundary_rows": blocked_boundaries,
        "ambiguous_rows": ambiguous_rows + ambiguous_envelopes,
        "candidate_turn_keys": candidate_turn_keys,
        "complete_turns": complete_turns,
        "extraction_coverage": complete_turns / candidate_turn_keys
        if candidate_turn_keys
        else 0.0,
        "entry_without_exit": fipa_stats["entry_without_exit"],
        "exit_without_entry": fipa_stats["exit_without_entry"],
        "duplicate_entry_keys": sum(len(v) > 1 for v in events.entries.values()),
        "duplicate_exit_keys": sum(len(v) > 1 for v in events.exits.values()),
        "conflicting_entry_keys": fipa_stats["conflicting_entry_keys"],
        "conflicting_exit_keys": fipa_stats["conflicting_exit_keys"],
        "malformed_entry_rows": events.malformed_entries,
        "malformed_exit_rows": events.malformed_exits,
        "failures_without_text": events.failures_without_text,
        "counterparts": list(events.counterparts),
        "counterpart_conversations": events.counterparts,
        "incomplete_boundary_rows": incomplete,
        "cross_trace_turns": fipa_stats["cross_trace_turns"],
        "session_mismatches": fipa_stats["session_mismatches"],
        "route_chain_complete_turns": fipa_stats["route_chain_complete_turns"],
        "supported_trace_count": len(candidate_traces),
        "no_boundary_trace_count": len(no_boundary_trace_ids),
        "unsupported_trace_count": 0,
        "unsupported_rows": 0,
        "strategies": dict(sorted(Counter(turns["schema_version"].tolist()).items())),
        "discovery": {
            "request_candidates": _top(request_candidates, config.max_issue_examples),
            "response_candidates": _top(response_candidates, config.max_issue_examples),
            "forwarded_receivers": _top(forwarded_receivers, config.max_issue_examples),
        },
        "turn_key_policy": "FIPA conversation_id+reply_with/in_reply_to; boundary trace_id+span_id",
        "order_policy": "session_id,entry_time_ns,turn_id",
        "issue_counts": dict(sorted(issue_counts.items())),
        "issue_rows": len(issues),
        "issue_examples_truncated": len(issues) > len(issue_frame),
    }
    logger.info(
        "Извлечение завершено: turns=%d, candidates=%d, FIPA=%d, boundary=%d",
        complete_turns,
        candidate_turn_keys,
        len(fipa_turns),
        len(sync_turns),
    )
    return ExtractionResult(turns=turns, issues=issue_frame, report=report)
