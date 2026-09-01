from __future__ import annotations

import json

import pandas as pd
import pytest

from trace_dialogue import ExtractionConfig, TraceExtractionError, extract_turns


def _span(
    *,
    trace_id: str,
    span_id: str,
    input_payload: object,
    output_payload: object,
    start_time_ns: int,
    session_id: str = "session-1",
    aef_kind: str = "input_request",
    span_name: str = "POST /invoke",
) -> dict[str, object]:
    return {
        "agent_id": "CI00000001",
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": "",
        "session_id": session_id,
        "start_time_ns": start_time_ns,
        "aef_kind": aef_kind,
        "span_name": span_name,
        "input_text": json.dumps(input_payload, ensure_ascii=False),
        "output_text": json.dumps(output_payload, ensure_ascii=False),
    }


def _message(
    *,
    sender: str,
    receiver: str,
    conversation_id: str,
    value: str,
    reply_with: str | None = None,
    in_reply_to: str | None = None,
) -> dict[str, object]:
    message: dict[str, object] = {
        "performative": "request" if reply_with is not None else "inform",
        "sender": sender,
        "receiver": receiver,
        "conversation_id": conversation_id,
        "content": {"message": [{"type": "text", "value": value}]},
    }
    if reply_with is not None:
        message["reply_with"] = reply_with
    if in_reply_to is not None:
        message["in_reply_to"] = in_reply_to
    return message


def _outgoing(
    *,
    receiver: str,
    value: str,
    reply_with: str | None = None,
    in_reply_to: str | None = None,
) -> dict[str, object]:
    outgoing: dict[str, object] = {
        "performative": "request" if reply_with is not None else "inform",
        "receiver": receiver,
        "content": {"message": [{"type": "text", "value": value}]},
    }
    if reply_with is not None:
        outgoing["reply_with"] = reply_with
    if in_reply_to is not None:
        outgoing["in_reply_to"] = in_reply_to
    return outgoing


def _fipa_entry(
    *,
    trace_id: str = "trace-entry",
    span_id: str = "entry",
    query: str = "Где мой ответ?",
    start_time_ns: int = 10,
) -> dict[str, object]:
    incoming = _message(
        sender="agent_human",
        receiver="orchestrator",
        conversation_id="conversation-1",
        reply_with="user-request-1",
        value=query,
    )
    outgoing = _outgoing(
        receiver="domain-agent",
        reply_with="domain-request-1",
        value="route payload",
    )
    return _span(
        trace_id=trace_id,
        span_id=span_id,
        input_payload={"message": incoming},
        output_payload={"outgoing": outgoing},
        start_time_ns=start_time_ns,
    )


def _fipa_exit(
    *,
    trace_id: str = "trace-exit",
    span_id: str = "exit",
    answer: str = "Вот точный ответ агента.",
    start_time_ns: int = 20,
) -> dict[str, object]:
    incoming = _message(
        sender="domain-agent",
        receiver="orchestrator",
        conversation_id="conversation-1",
        in_reply_to="domain-request-1",
        value="downstream result",
    )
    outgoing = _outgoing(
        receiver="agent_human",
        in_reply_to="user-request-1",
        value=answer,
    )
    return _span(
        trace_id=trace_id,
        span_id=span_id,
        input_payload={"message": incoming},
        output_payload={"outgoing": outgoing},
        start_time_ns=start_time_ns,
    )


def test_fipa_turn_is_joined_by_protocol_key_across_traces() -> None:
    result = extract_turns(
        pd.DataFrame([_fipa_entry(), _fipa_exit()]),
        ExtractionConfig(agent_id="CI00000001"),
    )

    assert len(result.turns) == 1
    turn = result.turns.iloc[0]
    assert turn["turn_id"] == "conversation-1|user-request-1"
    assert turn["input_query"] == "Где мой ответ?"
    assert turn["agent_response"] == "Вот точный ответ агента."
    assert turn["route_label"] == "domain-agent"
    assert bool(turn["same_trace"]) is False
    assert bool(turn["route_chain_complete"]) is True
    assert result.report["complete_turns"] == 1
    assert result.report["cross_trace_turns"] == 1


def test_equivalent_fipa_replicas_are_deduplicated() -> None:
    replica = _fipa_entry(trace_id="trace-entry-copy", span_id="entry-copy")
    result = extract_turns(
        pd.DataFrame([_fipa_entry(), replica, _fipa_exit()]),
        ExtractionConfig(agent_id="CI00000001"),
    )

    assert len(result.turns) == 1
    assert result.report["duplicate_entry_keys"] == 1
    assert result.report["conflicting_entry_keys"] == 0


def test_conflicting_fipa_replicas_are_not_published() -> None:
    conflict = _fipa_entry(
        trace_id="trace-entry-copy",
        span_id="entry-copy",
        query="Другой запрос с тем же correlation key",
    )
    result = extract_turns(
        pd.DataFrame([_fipa_entry(), conflict, _fipa_exit()]),
        ExtractionConfig(agent_id="CI00000001"),
    )

    assert result.turns.empty
    assert result.report["conflicting_entry_keys"] == 1
    assert "conflicting_entry_replicas" in set(result.issues["issue_code"])


def test_incomplete_fipa_keys_are_reported_separately() -> None:
    orphan_exit = _fipa_exit()
    input_text = orphan_exit["input_text"]
    assert isinstance(input_text, str)
    orphan_exit["input_text"] = input_text.replace("conversation-1", "conversation-2")
    result = extract_turns(
        pd.DataFrame([_fipa_entry(), orphan_exit]),
        ExtractionConfig(agent_id="CI00000001"),
    )

    assert result.turns.empty
    assert result.report["entry_without_exit"] == 1
    assert result.report["exit_without_entry"] == 1
    assert set(result.issues["issue_code"]) == {
        "entry_without_exit",
        "exit_without_entry",
    }


def test_explicit_aef_boundary_pair_is_supported() -> None:
    row = _span(
        trace_id="trace-aef",
        span_id="boundary",
        input_payload={"message": "Вопрос пользователя"},
        output_payload={"answer": "Финальный ответ агента"},
        start_time_ns=100,
    )
    result = extract_turns(
        pd.DataFrame([row]),
        ExtractionConfig(agent_id="CI00000001"),
    )

    assert result.turns[["input_query", "agent_response"]].to_dict("records") == [
        {
            "input_query": "Вопрос пользователя",
            "agent_response": "Финальный ответ агента",
        }
    ]
    assert result.report["strategies"] == {"aef_boundary_v1": 1}


@pytest.mark.parametrize(
    ("input_key", "output_key"),
    [("goal", "summary"), ("task", "error"), ("text", "summary")],
)
def test_start_agent_upstream_fallback_is_versioned(
    input_key: str, output_key: str
) -> None:
    row = _span(
        trace_id="trace-start-agent",
        span_id="agent",
        input_payload={input_key: "Задача агента"},
        output_payload={output_key: "Результат агента"},
        start_time_ns=100,
        aef_kind="start_agent",
        span_name="Agent",
    )

    result = extract_turns(
        pd.DataFrame([row]),
        ExtractionConfig(agent_id="CI00000001"),
    )

    turn = result.turns.iloc[0]
    assert turn["input_query"] == "Задача агента"
    assert turn["agent_response"] == "Результат агента"
    assert turn["schema_version"] == "aef_start_agent_v1"


def test_direct_parent_boundary_recovers_mismatched_fipa_key() -> None:
    parent = _span(
        trace_id="trace-parent",
        span_id="boundary",
        input_payload={
            "message": _message(
                sender="caller",
                receiver="agent",
                conversation_id="conversation-1",
                reply_with="request-in",
                value="Вопрос пользователя",
            )
        },
        output_payload={
            "outgoing": _outgoing(
                receiver="caller",
                in_reply_to="request-out",
                value="Финальный ответ агента",
            )
        },
        start_time_ns=100,
        aef_kind="other",
        span_name="POST /invoke",
    )
    input_payload = json.loads(str(parent["input_text"]))
    input_payload["message"]["content"]["message"] = [
        {"type": "text", "value": "route"},
        {"type": "text", "value": "Вопрос пользователя"},
    ]
    parent["input_text"] = json.dumps(input_payload, ensure_ascii=False)
    start = _span(
        trace_id="trace-parent",
        span_id="agent",
        input_payload={"text": "route Вопрос пользователя"},
        output_payload={"answer": "route Вопрос пользователя"},
        start_time_ns=110,
        aef_kind="start_agent",
        span_name="agent_start",
    )
    start["parent_span_id"] = "boundary"

    result = extract_turns(
        pd.DataFrame([parent, start]),
        ExtractionConfig(agent_id="CI00000001"),
    )

    turn = result.turns.iloc[0]
    assert turn["input_query"] == "Вопрос пользователя"
    assert turn["agent_response"] == "Финальный ответ агента"
    assert turn["route_label"] == "route"
    assert turn["schema_version"] == "aef_parent_boundary_v1"
    assert result.report["candidate_turn_keys"] == 1
    assert result.report["complete_turns"] == 1
    assert result.report["entry_without_exit"] == 0
    assert result.report["exit_without_entry"] == 0


def test_start_agent_echo_uses_unique_explicit_terminal_answer() -> None:
    start = _span(
        trace_id="trace-terminal",
        span_id="agent",
        input_payload={"text": "Вопрос пользователя"},
        output_payload={"answer": "Вопрос пользователя"},
        start_time_ns=100,
        aef_kind="start_agent",
        span_name="agent_start",
    )
    start["parent_span_id"] = "missing-boundary"
    terminal = _span(
        trace_id="trace-terminal",
        span_id="terminal",
        input_payload={},
        output_payload={"full_answer": ["Финальный ответ", {"widget": {}}]},
        start_time_ns=110,
        aef_kind="chain",
        span_name="finalize",
    )
    terminal["parent_span_id"] = "agent"

    result = extract_turns(
        pd.DataFrame([start, terminal]),
        ExtractionConfig(agent_id="CI00000001"),
    )

    turn = result.turns.iloc[0]
    assert turn["input_query"] == "Вопрос пользователя"
    assert turn["agent_response"] == "Финальный ответ"
    assert turn["schema_version"] == "aef_semantic_terminal_v1"
    assert result.report["candidate_turn_keys"] == 1
    assert result.report["complete_turns"] == 1
    assert result.report["incomplete_boundary_rows"] == 0


def test_semantic_terminal_does_not_cross_session_boundary() -> None:
    start = _span(
        trace_id="trace-terminal",
        span_id="agent",
        input_payload={"text": "Вопрос пользователя"},
        output_payload={"answer": "Вопрос пользователя"},
        start_time_ns=100,
        aef_kind="start_agent",
        span_name="agent_start",
    )
    terminal = _span(
        trace_id="trace-terminal",
        span_id="terminal",
        input_payload={},
        output_payload={"full_answer": "Ответ другой сессии"},
        start_time_ns=110,
        aef_kind="chain",
        span_name="finalize",
    )
    terminal["session_id"] = "session-2"

    result = extract_turns(
        pd.DataFrame([start, terminal]),
        ExtractionConfig(agent_id="CI00000001"),
    )

    assert result.turns.empty
    assert result.report["complete_turns"] == 0
    assert result.report["candidate_turn_keys"] == 1
    assert result.report["incomplete_boundary_rows"] == 1


def test_explicit_string_response_body_is_supported() -> None:
    row = _span(
        trace_id="trace-aef",
        span_id="boundary",
        input_payload={"body": {"message": "Вопрос пользователя"}},
        output_payload={"response": {"body": "Финальный ответ агента"}},
        start_time_ns=100,
    )

    result = extract_turns(
        pd.DataFrame([row]),
        ExtractionConfig(agent_id="CI00000001"),
    )

    assert result.turns.loc[0, "agent_response"] == "Финальный ответ агента"
    assert result.turns.loc[0, "response_source_path"] == "output_text.response.body"


def test_nested_start_agent_is_not_used_as_outer_response() -> None:
    outer = _span(
        trace_id="trace-aef",
        span_id="outer",
        input_payload={"message": "Вопрос пользователя"},
        output_payload={"status": "processing"},
        start_time_ns=100,
        aef_kind="input_request",
    )
    nested = _span(
        trace_id="trace-aef",
        span_id="nested",
        input_payload={"goal": "internal task"},
        output_payload={"answer": "Ответ внутреннего агента"},
        start_time_ns=110,
        aef_kind="start_agent",
        span_name="DomainAgent",
    )
    result = extract_turns(
        pd.DataFrame([outer, nested]),
        ExtractionConfig(agent_id="CI00000001"),
    )

    assert result.turns.empty
    assert "boundary_pair_incomplete" in set(result.issues["issue_code"])


def test_malformed_structured_payload_is_not_published_as_plain_text() -> None:
    row = _span(
        trace_id="trace-aef",
        span_id="boundary",
        input_payload={"message": "Вопрос пользователя"},
        output_payload={"answer": "Финальный ответ агента"},
        start_time_ns=100,
    )
    row["input_text"] = '{"message": "повреждённый JSON"'

    result = extract_turns(
        pd.DataFrame([row]),
        ExtractionConfig(agent_id="CI00000001"),
    )

    assert result.turns.empty
    assert set(result.issues["issue_code"]) == {"boundary_pair_incomplete"}


def test_session_mismatch_is_not_published() -> None:
    exit_row = _fipa_exit()
    exit_row["session_id"] = "session-2"

    result = extract_turns(
        pd.DataFrame([_fipa_entry(), exit_row]),
        ExtractionConfig(agent_id="CI00000001"),
    )

    assert result.turns.empty
    assert result.report["session_mismatches"] == 1
    assert "session_id_mismatch" in set(result.issues["issue_code"])


def test_unknown_schema_is_visible_and_not_guessed() -> None:
    row = _span(
        trace_id="trace-unknown",
        span_id="unknown",
        input_payload={"arbitrary": {"text": "looks like a query"}},
        output_payload={"arbitrary": {"text": "looks like an answer"}},
        start_time_ns=100,
        aef_kind="chain",
        span_name="unknown",
    )
    result = extract_turns(
        pd.DataFrame([row]),
        ExtractionConfig(agent_id="CI00000001"),
    )

    assert result.turns.empty
    assert result.report["no_boundary_trace_count"] == 1
    assert result.report["complete_turns"] == 0
    assert set(result.issues["issue_code"]) == {"no_boundary_span"}


def test_missing_span_identity_fails_closed() -> None:
    row = _fipa_entry()
    row.pop("span_id")

    with pytest.raises(TraceExtractionError, match="span_id"):
        extract_turns(
            pd.DataFrame([row]),
            ExtractionConfig(agent_id="CI00000001"),
        )


def test_multipart_answer_keeps_every_text_part_and_skips_widgets() -> None:
    exit_row = _fipa_exit()
    output = json.loads(str(exit_row["output_text"]))
    output["outgoing"]["content"]["message"] = [
        {"type": "text", "value": "## Одобренные продукты\n\n"},
        {"type": "widget", "value": None},
        {"type": "text", "value": "\n\n## Отказанные продукты"},
    ]
    exit_row["output_text"] = json.dumps(output, ensure_ascii=False)

    result = extract_turns(
        pd.DataFrame([_fipa_entry(), exit_row]),
        ExtractionConfig(agent_id="CI00000001"),
    )

    assert result.turns.iloc[0]["agent_response"] == (
        "## Одобренные продукты\n\n## Отказанные продукты"
    )


def test_widget_first_answer_is_not_lost() -> None:
    exit_row = _fipa_exit()
    output = json.loads(str(exit_row["output_text"]))
    output["outgoing"]["content"]["message"] = [
        {"type": "widget", "value": None},
        {"type": "text", "value": "Текст после виджета"},
    ]
    exit_row["output_text"] = json.dumps(output, ensure_ascii=False)

    result = extract_turns(
        pd.DataFrame([_fipa_entry(), exit_row]),
        ExtractionConfig(agent_id="CI00000001"),
    )

    assert result.turns.iloc[0]["agent_response"] == "Текст после виджета"


def test_route_label_is_dispatched_label_when_dispatch_echoes_query() -> None:
    entry = _fipa_entry(query="хочу взять кредит")
    output = json.loads(str(entry["output_text"]))
    output["outgoing"]["content"]["message"] = [
        {"type": "text", "value": "issuance"},
        {"type": "text", "value": "хочу взять кредит"},
    ]
    entry["output_text"] = json.dumps(output, ensure_ascii=False)

    result = extract_turns(
        pd.DataFrame([entry, _fipa_exit()]),
        ExtractionConfig(agent_id="CI00000001"),
    )

    turn = result.turns.iloc[0]
    assert turn["route_label"] == "issuance"
    assert turn["route_source_path"] == "output_text.outgoing.content.message[0].value"
    assert turn["agent_response"] == "Вот точный ответ агента."


def test_route_label_falls_back_to_receiver_without_echo() -> None:
    result = extract_turns(
        pd.DataFrame([_fipa_entry(), _fipa_exit()]),
        ExtractionConfig(agent_id="CI00000001"),
    )

    turn = result.turns.iloc[0]
    assert turn["route_label"] == "domain-agent"
    assert turn["route_source_path"] == "output_text.outgoing.receiver"


def _downstream_span(
    *,
    trace_id: str,
    span_id: str,
    query_parts: list[str],
    answer: str,
    request_id: str = "router-request-1",
    start_time_ns: int = 10,
) -> dict[str, object]:
    """Нижестоящий агент: вход request от роутера и ответ inform ему же на одном span."""
    incoming = {
        "performative": "request",
        "sender": "d-credit-helper",
        "receiver": "p-credit-helper",
        "conversation_id": "conversation-9",
        "reply_with": request_id,
        "content": {
            "message": [{"type": "text", "value": part} for part in query_parts]
        },
    }
    outgoing = {
        "performative": "inform",
        "sender": "p-credit-helper",
        "receiver": "d-credit-helper",
        "in_reply_to": request_id,
        "content": {"message": [{"type": "text", "value": answer}]},
    }
    return _span(
        trace_id=trace_id,
        span_id=span_id,
        input_payload={"message": incoming},
        output_payload={"outgoing": outgoing},
        start_time_ns=start_time_ns,
    )


def test_downstream_agent_turn_uses_requesting_agent_as_counterpart() -> None:
    row = _downstream_span(
        trace_id="trace-9",
        span_id="boundary",
        query_parts=["issuance", "выдай мне кредит"],
        answer="Кредит можно оформить в СберБанк Онлайн.",
    )

    result = extract_turns(pd.DataFrame([row]), ExtractionConfig(agent_id="CI00000001"))

    assert len(result.turns) == 1
    turn = result.turns.iloc[0]
    assert turn["input_query"] == "выдай мне кредит"
    assert turn["route_label"] == "issuance"
    assert turn["route_source_path"] == "input_text.message.content.message[0].value"
    assert turn["agent_response"] == "Кредит можно оформить в СберБанк Онлайн."
    assert bool(turn["same_trace"]) is True
    assert result.report["counterparts"] == ["d-credit-helper"]


def test_internal_inform_to_non_counterpart_is_not_an_orphan_exit() -> None:
    internal = _span(
        trace_id="trace-internal",
        span_id="internal",
        input_payload={
            "message": _message(
                sender="domain-agent",
                receiver="orchestrator",
                conversation_id="conversation-1",
                in_reply_to="domain-request-1",
                value="downstream result",
            )
        },
        output_payload={
            "outgoing": _outgoing(
                receiver="domain-agent",
                in_reply_to="domain-request-1",
                value="ack to domain agent",
            )
        },
        start_time_ns=15,
    )

    result = extract_turns(
        pd.DataFrame([_fipa_entry(), internal, _fipa_exit()]),
        ExtractionConfig(agent_id="CI00000001"),
    )

    assert len(result.turns) == 1
    assert result.report["exit_without_entry"] == 0


def test_trace_without_boundary_span_is_reported_separately() -> None:
    chain_only = _span(
        trace_id="trace-chain",
        span_id="chain",
        input_payload={"messages": [{"type": "human", "content": "Вопрос"}]},
        output_payload={"messages": [{"type": "ai", "content": "Ответ"}]},
        start_time_ns=1,
        aef_kind="chain",
        span_name="RunnableSequence",
    )

    result = extract_turns(
        pd.DataFrame([_fipa_entry(), _fipa_exit(), chain_only]),
        ExtractionConfig(agent_id="CI00000001"),
    )

    assert result.report["no_boundary_trace_count"] == 1
    assert result.report["unsupported_trace_count"] == 0
    assert "no_boundary_span" in set(result.issues["issue_code"])


def test_downstream_clarifying_request_does_not_make_it_a_counterpart() -> None:
    """Нижестоящий агент шлёт request (уточнение) внутри conversation, открытой человеком;
    даже если начало разговора обрезано окном, он не становится контрагентом."""
    clarify = _span(
        trace_id="trace-clarify",
        span_id="clarify",
        input_payload={
            "message": _message(
                sender="domain-agent",
                receiver="orchestrator",
                conversation_id="conversation-cut",
                reply_with="clarify-1",
                value="Уточните сумму",
            )
        },
        output_payload={
            "outgoing": _outgoing(
                receiver="agent_human",
                reply_with="clarify-fwd-1",
                value="Уточните сумму",
            )
        },
        start_time_ns=5,
    )
    relayed_answer = _span(
        trace_id="trace-clarify-2",
        span_id="clarify-answer",
        input_payload={
            "message": _message(
                sender="agent_human",
                receiver="orchestrator",
                conversation_id="conversation-cut",
                in_reply_to="clarify-fwd-1",
                value="100000",
            )
        },
        output_payload={
            "outgoing": _outgoing(
                receiver="domain-agent", in_reply_to="clarify-1", value="100000"
            )
        },
        start_time_ns=6,
    )
    # та же пара domain-agent → orchestrator как уточнение внутри обычного turn
    inside = dict(clarify, trace_id="trace-inside", span_id="inside", start_time_ns=12)
    inside_payload = json.loads(str(inside["input_text"]))
    inside_payload["message"]["conversation_id"] = "conversation-1"
    inside_payload["message"]["reply_with"] = "clarify-2"
    inside["input_text"] = json.dumps(inside_payload, ensure_ascii=False)

    result = extract_turns(
        pd.DataFrame([_fipa_entry(), inside, _fipa_exit(), clarify, relayed_answer]),
        ExtractionConfig(agent_id="CI00000001"),
    )

    assert result.report["counterparts"] == ["agent_human"]
    assert len(result.turns) == 1
    assert result.turns.iloc[0]["input_query"] == "Где мой ответ?"


def test_agent_own_name_is_never_a_counterpart() -> None:
    row = _downstream_span(
        trace_id="trace-9",
        span_id="boundary",
        query_parts=["issuance", "выдай мне кредит"],
        answer="Ответ",
    )
    self_request = _span(
        trace_id="trace-self",
        span_id="self",
        input_payload={
            "message": _message(
                sender="p-credit-helper",
                receiver="p-credit-helper",
                conversation_id="conversation-self",
                reply_with="self-1",
                value="internal",
            )
        },
        output_payload={},
        start_time_ns=20,
    )

    result = extract_turns(
        pd.DataFrame([row, self_request]), ExtractionConfig(agent_id="CI00000001")
    )

    assert result.report["counterparts"] == ["d-credit-helper"]


def _state_json_span(
    *,
    trace_id: str = "trace-state",
    span_id: str = "state-exit",
    stage: str = "exit",
    query: str = "Подбери вклад",
    answer: str = "Предлагаю вклад «Лучший %».",
    product_agent: str | None = "deposelector",
    start_time_ns: int = 10,
    session_id: str = "session-1",
) -> dict[str, object]:
    """Спан langgraph-агента: диалог лежит в state_json внутри input_text."""
    state = {
        "stage": stage,
        "messages": [
            {"role": "assistant", "content": "Здравствуйте!"},
            {"role": "user", "content": query},
        ],
        "message_to_user": answer,
        "product_agent": product_agent,
    }
    return _span(
        trace_id=trace_id,
        span_id=span_id,
        input_payload=state,
        output_payload={"model": "giga"},
        start_time_ns=start_time_ns,
        session_id=session_id,
        aef_kind="chain",
        span_name="LangGraph",
    )


def test_state_json_exit_span_becomes_turn() -> None:
    """Агент со state_json (CI09840650): вопрос из messages, ответ из message_to_user."""
    result = extract_turns(
        pd.DataFrame([_state_json_span()]),
        ExtractionConfig(agent_id="CI00000001"),
    )

    assert len(result.turns) == 1
    turn = result.turns.iloc[0]
    assert turn["schema_family"] == "state_json"
    assert turn["input_query"] == "Подбери вклад"
    assert turn["agent_response"] == "Предлагаю вклад «Лучший %»."
    assert turn["route_label"] == "deposelector"


def test_state_json_without_product_agent_keeps_route_unknown() -> None:
    result = extract_turns(
        pd.DataFrame([_state_json_span(product_agent=None)]),
        ExtractionConfig(agent_id="CI00000001"),
    )

    assert result.turns.iloc[0]["route_label"] == ""
    assert result.turns.iloc[0]["route_source_path"] == ""


def test_state_json_final_state_from_output_reconciles_incomplete_fipa() -> None:
    root = _state_json_span()
    input_state = json.loads(str(root["input_text"]))
    input_state["stage"] = "scenarist"
    input_state["message_to_user"] = None
    root["input_text"] = json.dumps(input_state, ensure_ascii=False)
    output_state = {
        **input_state,
        "stage": "__end__",
        "message_to_user": "Предлагаю вклад «Лучший %».",
        "product_agent": "deposelector",
    }
    root["output_text"] = json.dumps(output_state, ensure_ascii=False)
    entry = _fipa_entry(trace_id="trace-state", query="Подбери вклад")

    result = extract_turns(
        pd.DataFrame([entry, root]),
        ExtractionConfig(agent_id="CI00000001"),
    )

    assert len(result.turns) == 1
    assert result.turns.iloc[0]["response_source_path"] == (
        "output_text.message_to_user"
    )
    assert result.report["candidate_turn_keys"] == 1
    assert result.report["complete_turns"] == 1
    assert result.report["entry_without_exit"] == 0
    assert "entry_without_exit" not in set(result.issues["issue_code"])


def test_state_json_output_echo_is_not_published_or_reconciled() -> None:
    root = _state_json_span()
    input_state = json.loads(str(root["input_text"]))
    input_state["stage"] = "scenarist"
    input_state["message_to_user"] = None
    root["input_text"] = json.dumps(input_state, ensure_ascii=False)
    root["output_text"] = json.dumps(
        {
            **input_state,
            "stage": "__end__",
            "message_to_user": "Подбери вклад",
            "product_agent": "deposelector",
        },
        ensure_ascii=False,
    )
    entry = _fipa_entry(trace_id="trace-state", query="Подбери вклад")

    result = extract_turns(
        pd.DataFrame([entry, root]),
        ExtractionConfig(agent_id="CI00000001"),
    )

    assert result.turns.empty
    assert result.report["candidate_turn_keys"] == 1
    assert result.report["entry_without_exit"] == 1
    assert "entry_without_exit" in set(result.issues["issue_code"])


def test_state_json_keeps_only_last_exit_per_trace() -> None:
    """Несколько exit-спанов одного trace — публикуется последний (финальное состояние)."""
    spans = pd.DataFrame([
        _state_json_span(span_id="early", answer="Черновик", start_time_ns=10),
        _state_json_span(span_id="late", answer="Финал", start_time_ns=20),
        _state_json_span(span_id="enter", stage="enter", start_time_ns=5),
    ])

    result = extract_turns(spans, ExtractionConfig(agent_id="CI00000001"))

    assert len(result.turns) == 1
    assert result.turns.iloc[0]["agent_response"] == "Финал"


def test_state_json_does_not_shadow_fipa_turns() -> None:
    """state_json дополняет только непокрытые трейсы, не дублируя fipa-turn'ы."""
    spans = pd.DataFrame([
        _fipa_entry(),
        _fipa_exit(),
        _state_json_span(trace_id="trace-entry", span_id="state-dup"),
        _state_json_span(trace_id="trace-solo", span_id="state-solo", session_id="session-2"),
    ])

    result = extract_turns(spans, ExtractionConfig(agent_id="CI00000001"))

    families = sorted(result.turns["schema_family"].tolist())
    assert families == ["fipa_acl", "state_json"]
