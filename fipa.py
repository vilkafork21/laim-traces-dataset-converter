"""FIPA ACL: события конвертов, контрагенты и соединение turn по correlation key."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import quote

from bodies import project_body

if TYPE_CHECKING:
    from trace_dialogue import ParsedSpan

FIPA_SCHEMA = "fipa_acl_v1"
_TERMINAL = frozenset({"inform", "failure", "refuse", "not-understood"})

ISSUE_COLUMNS = [
    "issue_code",
    "severity",
    "schema_version",
    "turn_id",
    "trace_id",
    "span_id",
    "details",
]


def issue(
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


def _identifier(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _performative(envelope: dict[str, object]) -> str:
    return _identifier(envelope.get("performative")).casefold()


def is_request(envelope: dict[str, object]) -> bool:
    return _performative(envelope) == "request"


def is_reply(envelope: dict[str, object]) -> bool:
    """Завершающее сообщение FIPA request; agree ещё не является результатом."""
    return _performative(envelope) in _TERMINAL and bool(
        _identifier(envelope.get("in_reply_to"))
    )


def is_agreement(envelope: dict[str, object]) -> bool:
    return _performative(envelope) == "agree"


def route_label(outgoing: tuple[str, dict[str, object]] | None) -> tuple[str, str]:
    """Маршрут известен только при наблюдаемой отправке request другому участнику."""
    if outgoing is None or not is_request(outgoing[1]):
        return "", ""
    return _identifier(outgoing[1].get("receiver")), f"{outgoing[0]}.receiver"


@dataclass
class Events:
    """Входы и выходы контрагентов, сгруппированные по protocol key."""

    entries: dict[tuple[str, str], list[dict[str, object]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    exits: dict[tuple[str, str], list[dict[str, object]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    registered: set[tuple[str, str]] = field(default_factory=set)
    counterparts: dict[str, int] = field(default_factory=dict)
    malformed_entries: int = 0
    malformed_exits: int = 0
    failures_without_text: int = 0


def derive_counterparts(spans: Iterable[ParsedSpan]) -> dict[str, set[str]]:
    """Инициаторы conversation: входящий request предшествует исходящему этой стороне.

    Порядок определяется внутри conversation. Свидетельство нижестоящей роли
    исключает участника и в неполных conversation. Самообращение исключается.
    """
    incoming_first: dict[tuple[str, str], int] = {}
    outgoing_first: dict[tuple[str, str], int] = {}
    same_span_requests: set[tuple[str, str, int]] = set()
    for span in spans:
        if span.time_ns is None:
            continue
        incoming = span.incoming[1] if span.incoming else {}
        outgoing = span.outgoing[1] if span.outgoing else {}
        conversation = _identifier(incoming.get("conversation_id")) or _identifier(
            outgoing.get("conversation_id")
        )
        if not conversation:
            continue
        sender = _identifier(incoming.get("sender"))
        receiver = _identifier(incoming.get("receiver"))
        if is_request(incoming) and sender and sender != receiver:
            incoming_conversation = _identifier(incoming.get("conversation_id"))
            if incoming_conversation:
                key = (incoming_conversation, sender)
                incoming_first[key] = min(
                    incoming_first.get(key, span.time_ns), span.time_ns
                )
        if is_request(outgoing):
            party = _identifier(outgoing.get("receiver"))
            key = (_identifier(outgoing.get("conversation_id")) or conversation, party)
            outgoing_first[key] = min(
                outgoing_first.get(key, span.time_ns), span.time_ns
            )
            if is_request(incoming) and key == (
                _identifier(incoming.get("conversation_id")),
                sender,
            ):
                same_span_requests.add((*key, span.time_ns))
    # Если сторона уже наблюдалась как нижестоящая, обрезанный conversation
    # не доказывает смену её роли. Такие входы остаются неразрешёнными.
    downstream = {
        party
        for (conversation, party), time_ns in incoming_first.items()
        if outgoing_first.get((conversation, party), time_ns + 1) < time_ns
    }
    counterparts: dict[str, set[str]] = defaultdict(set)
    for (conversation, party), time_ns in incoming_first.items():
        sent_at = outgoing_first.get((conversation, party))
        # При равном времени порядок известен только для входа и выхода одного спана.
        leads = (
            sent_at is None
            or time_ns < sent_at
            or (
                time_ns == sent_at
                and (conversation, party, time_ns) in same_span_requests
            )
        )
        if party not in downstream and leads:
            counterparts[party].add(conversation)
    return dict(sorted(counterparts.items()))


def collect_events(
    spans: Iterable[ParsedSpan],
    counterparts: dict[str, set[str]],
    issues: list[dict[str, str]],
    *,
    shadow: Events | None = None,
) -> Events:
    """Разложить конверты на входы/выходы контрагентов."""
    events = Events()
    events.counterparts = {
        party: len(conversations) for party, conversations in counterparts.items()
    }
    for span in spans:
        if span.incoming is None and span.outgoing is None:
            continue
        base = {
            "session_id": span.session_id,
            "trace_id": span.trace_id,
            "span_id": span.span_id,
            "time_ns": span.time_ns,
        }
        incoming = span.incoming[1] if span.incoming else {}
        outgoing = span.outgoing[1] if span.outgoing else {}
        conversation_id = _identifier(incoming.get("conversation_id"))
        if (
            span.incoming
            and is_request(incoming)
            and conversation_id
            in counterparts.get(_identifier(incoming.get("sender")), set())
        ):
            request_id = _identifier(incoming.get("reply_with"))
            query_projection = project_body(incoming, side="request")
            query = query_projection.text if query_projection.status == "text" else ""
            route, route_path = route_label(span.outgoing)
            events.registered.add((span.trace_id, span.span_id))
            entry = {
                **base,
                "input_query": query,
                "route_label": route,
                "downstream_request_id": _identifier(outgoing.get("reply_with")),
                "downstream_conversation_id": _identifier(outgoing.get("conversation_id"))
                or conversation_id,
                "query_source_path": f"{span.incoming[0]}{query_projection.path}",
                "sender": _identifier(incoming.get("sender")),
                "receiver": _identifier(incoming.get("receiver")),
                "route_source_path": route_path,
            }
            if shadow is not None and conversation_id and request_id:
                shadow.entries[(conversation_id, request_id)].append({
                    **entry, "projection": query_projection, "body": incoming,
                })
            if (
                conversation_id
                and request_id
                and query
                and span.trace_id
                and span.span_id
            ):
                events.entries[(conversation_id, request_id)].append(entry)
            else:
                events.malformed_entries += 1
                issues.append(
                    issue(
                        "fipa_entry_incomplete",
                        schema_version=FIPA_SCHEMA,
                        trace_id=span.trace_id,
                        span_id=span.span_id,
                        details=f"conversation_id={conversation_id!r}, reply_with={request_id!r}, query_status={query_projection.status}",
                    )
                )
        if (
            not span.outgoing
            or _identifier(outgoing.get("receiver")) not in counterparts
        ):
            continue
        if (
            outgoing.get("in_reply_to")
            and not is_reply(outgoing)
            and not is_agreement(outgoing)
        ):
            events.registered.add((span.trace_id, span.span_id))
            events.malformed_exits += 1
            issues.append(
                issue(
                    "fipa_exit_not_terminal",
                    schema_version=FIPA_SCHEMA,
                    trace_id=span.trace_id,
                    span_id=span.span_id,
                    details=f"performative={_performative(outgoing)!r}",
                )
            )
        if is_reply(outgoing):
            events.registered.add((span.trace_id, span.span_id))
            conversation_id = _identifier(
                outgoing.get("conversation_id")
            ) or _identifier(incoming.get("conversation_id"))
            request_id = _identifier(outgoing.get("in_reply_to"))
            response_projection = project_body(outgoing, side="response")
            response = (
                response_projection.text if response_projection.status == "text" else ""
            )
            exit_event = {
                **base,
                "agent_response": response,
                "returned_downstream_request_id": _identifier(incoming.get("in_reply_to")),
                "returned_downstream_conversation_id": _identifier(incoming.get("conversation_id")),
                "response_source_path": f"{span.outgoing[0]}{response_projection.path}",
                "sender": _identifier(outgoing.get("sender")),
                "receiver": _identifier(outgoing.get("receiver")),
            }
            if shadow is not None and conversation_id and request_id:
                shadow.exits[(conversation_id, request_id)].append({
                    **exit_event, "projection": response_projection, "body": outgoing,
                })
            if (
                conversation_id
                and request_id
                and response
                and span.trace_id
                and span.span_id
            ):
                events.exits[(conversation_id, request_id)].append(exit_event)
            elif _performative(outgoing) == "failure" and not response:
                events.failures_without_text += 1
                issues.append(
                    issue(
                        "fipa_failure_without_text",
                        schema_version=FIPA_SCHEMA,
                        trace_id=span.trace_id,
                        span_id=span.span_id,
                    )
                )
            else:
                events.malformed_exits += 1
                issues.append(
                    issue(
                        "fipa_exit_incomplete",
                        schema_version=FIPA_SCHEMA,
                        trace_id=span.trace_id,
                        span_id=span.span_id,
                        details=f"conversation_id={conversation_id!r}, in_reply_to={request_id!r}, response_status={response_projection.status}",
                    )
                )
    return events


def _ordered(candidate: dict[str, object], *, latest: bool) -> tuple[int, str, str]:
    timestamp = candidate.get("time_ns")
    order_time = timestamp if isinstance(timestamp, int) else 0
    if latest:
        order_time = -order_time
    return (
        order_time,
        _identifier(candidate.get("trace_id")),
        _identifier(candidate.get("span_id")),
    )


def _merge_replicas(
    candidates: list[dict[str, object]],
    fields: tuple[str, ...],
    *,
    latest: bool,
) -> tuple[dict[str, object] | None, list[str]]:
    """Одинаковые реплики одного ключа схлопываются; разные тексты — конфликт."""
    if not candidates:
        return None, []
    conflicts = [
        name
        for name in fields
        if len({_identifier(item.get(name)) for item in candidates}) > 1
    ]
    chosen = min(candidates, key=lambda item: _ordered(item, latest=latest))
    return chosen, conflicts


def join_turns(
    events: Events, issues: list[dict[str, str]]
) -> tuple[list[dict[str, object]], Counter[str]]:
    """Соединить вход и финальный выход по (conversation_id, request_id)."""
    stats: Counter[str] = Counter()
    turns: list[dict[str, object]] = []
    for key in sorted(set(events.entries) | set(events.exits)):
        conversation_id, request_id = key
        turn_id = f"{quote(conversation_id, safe='')}|{quote(request_id, safe='')}"
        entry, entry_conflicts = _merge_replicas(
            events.entries.get(key, []),
            (
                "input_query",
                "route_label",
                "downstream_request_id",
                "downstream_conversation_id",
                "session_id",
                "sender",
                "receiver",
            ),
            latest=False,
        )
        exit_event, exit_conflicts = _merge_replicas(
            events.exits.get(key, []),
            (
                "agent_response",
                "returned_downstream_request_id",
                "returned_downstream_conversation_id",
                "session_id",
                "sender",
                "receiver",
            ),
            latest=True,
        )
        if entry_conflicts:
            stats["conflicting_entry_keys"] += 1
            issues.append(
                issue(
                    "conflicting_entry_replicas",
                    schema_version=FIPA_SCHEMA,
                    turn_id=turn_id,
                    details="Противоречивые поля: " + ",".join(entry_conflicts),
                )
            )
        if exit_conflicts:
            stats["conflicting_exit_keys"] += 1
            issues.append(
                issue(
                    "conflicting_exit_replicas",
                    schema_version=FIPA_SCHEMA,
                    turn_id=turn_id,
                    details="Противоречивые поля: " + ",".join(exit_conflicts),
                )
            )
        if entry_conflicts or exit_conflicts:
            continue
        if entry is None:
            stats["exit_without_entry"] += 1
            issues.append(
                issue("exit_without_entry", schema_version=FIPA_SCHEMA, turn_id=turn_id)
            )
            continue
        if exit_event is None:
            stats["entry_without_exit"] += 1
            issues.append(
                issue(
                    "entry_without_exit",
                    schema_version=FIPA_SCHEMA,
                    turn_id=turn_id,
                    trace_id=entry["trace_id"],
                    span_id=entry["span_id"],
                )
            )
            continue
        turn = _turn(turn_id, entry, exit_event, issues)
        if turn is None:
            continue
        stats["cross_trace_turns"] += int(not turn["same_trace"])
        stats["route_chain_complete_turns"] += int(turn["route_chain_complete"])
        turns.append(turn)
    stats["session_mismatches"] = sum(
        item["issue_code"] == "session_id_mismatch" for item in issues
    )
    return turns, stats


def _turn(
    turn_id: str,
    entry: dict[str, object],
    exit_event: dict[str, object],
    issues: list[dict[str, str]],
) -> dict[str, object] | None:
    """Собрать turn из пары событий; None, если пара не публикуема."""
    if entry["sender"] != exit_event["receiver"] or (
        exit_event["sender"] and entry["receiver"] != exit_event["sender"]
    ):
        issues.append(
            issue("counterpart_mismatch", schema_version=FIPA_SCHEMA, turn_id=turn_id)
        )
        return None
    session_id = entry["session_id"] or exit_event["session_id"]
    if not session_id:
        issues.append(
            issue("session_id_missing", schema_version=FIPA_SCHEMA, turn_id=turn_id)
        )
        return None
    both_sessions = bool(entry["session_id"] and exit_event["session_id"])
    same_session = both_sessions and entry["session_id"] == exit_event["session_id"]
    if both_sessions and not same_session:
        issues.append(
            issue("session_id_mismatch", schema_version=FIPA_SCHEMA, turn_id=turn_id)
        )
        return None
    if not both_sessions:
        issues.append(
            issue(
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
            issue("turn_timestamp_missing", schema_version=FIPA_SCHEMA, turn_id=turn_id)
        )
        return None
    latency_ms = (exit_time - entry_time) / 1_000_000.0
    if latency_ms < 0:
        issues.append(
            issue("negative_turn_latency", schema_version=FIPA_SCHEMA, turn_id=turn_id)
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
            and entry["downstream_conversation_id"]
            == exit_event["returned_downstream_conversation_id"]
        ),
        "turn_latency_ms": latency_ms,
        "query_source_path": entry["query_source_path"],
        "response_source_path": exit_event["response_source_path"],
        "route_source_path": entry["route_source_path"],
    }
