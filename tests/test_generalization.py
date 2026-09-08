from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from main import main
from bodies import project_body
from trace_dialogue import ExtractionConfig, TraceExtractionError, extract_turns


def span(
    request, response, *, trace="t", identity="s", parent="", kind="input_request"
):
    return {
        "agent_id": "arbitrary-agent",
        "trace_id": trace,
        "span_id": identity,
        "parent_span_id": parent,
        "session_id": "session",
        "start_time_ns": 100,
        "span_name": "arbitrary-node",
        "aef_kind": kind,
        "input_text": json.dumps(request, ensure_ascii=False),
        "output_text": json.dumps(response, ensure_ascii=False),
    }


def extract(*rows):
    return extract_turns(
        pd.DataFrame(rows), ExtractionConfig(agent_id="arbitrary-agent")
    )


def test_fipa_input_with_synchronous_response_and_custom_message_fields():
    request = {
        "wrapper": {
            "performative": "request",
            "sender": "caller",
            "receiver": "service",
            "conversation_id": "conversation",
            "reply_with": "request",
            "content": {
                "phrases": [
                    {
                        "speaker_type": "CUSTOMER",
                        "text": "Первый вопрос",
                        "time": "2026-09-01",
                    },
                    {"speaker_type": "OPERATOR", "text": "Уточните"},
                    {"speaker_type": "CUSTOMER", "text": "Текущий вопрос"},
                ]
            },
        }
    }
    result = extract(
        span(request, {"response": {"result": "Финальный ответ", "status_code": 200}})
    )
    assert result.turns[["input_query", "agent_response"]].values.tolist() == [
        ["Текущий вопрос", "Финальный ответ"]
    ]
    assert result.report["candidate_turn_keys"] == 1


@pytest.mark.parametrize(
    "answer", ["42", "yes", "https://example.test", "Вопрос", "=1+1"]
)
@pytest.mark.parametrize("kind", ["input_request", "start_agent"])
def test_answer_is_not_filtered_by_its_value(answer, kind):
    result = extract(
        span(
            {"input": "Вопрос"},
            {"answer": answer, "metadata": "Служебное пояснение"},
            kind=kind,
        )
    )
    assert result.turns.agent_response.tolist() == [answer]


@pytest.mark.parametrize("parent_column", [True, False])
def test_start_agent_with_unobserved_parent_is_not_an_external_boundary(parent_column):
    row = span(
        "Внутренний запрос", "Внутренний ответ", parent="missing", kind="start_agent"
    )
    if not parent_column:
        row.pop("parent_span_id")
    result = extract(row)
    assert result.turns.empty
    assert result.report["issue_counts"] == {"start_agent_not_root": 1}


def test_two_boundaries_with_identical_text_are_two_turns():
    first = span("Вопрос", "Ответ", identity="a")
    second = span("Вопрос", "Ответ", identity="b")
    result = extract(first, second, first)
    assert set(result.turns.turn_id) == {"aef:t:a", "aef:t:b"}
    assert result.report["duplicate_span_rows"] == 1


def test_different_boundaries_in_one_trace_are_not_ambiguous():
    result = extract(
        span("Первый", "Ответ 1", identity="a"), span("Второй", "Ответ 2", identity="b")
    )
    assert result.turns.agent_response.tolist() == ["Ответ 1", "Ответ 2"]


def test_fipa_and_plain_boundaries_in_one_trace_are_both_preserved():
    plain = span("Обычный вопрос", "Обычный ответ", identity="plain")
    protocol = span(
        envelope("Вопрос FIPA"),
        envelope(
            "Ответ FIPA", performative="inform", sender="service", receiver="caller"
        ),
        identity="protocol",
    )
    result = extract(plain, protocol)
    assert set(result.turns.agent_response) == {"Обычный ответ", "Ответ FIPA"}
    assert result.report["candidate_turn_keys"] == 2


def test_conflicting_span_copies_are_not_published():
    result = extract(span("Вопрос", "Ответ 1"), span("Вопрос", "Ответ 2"))
    assert result.turns.empty
    assert result.report["conflicting_span_rows"] == 2


def test_parent_ids_are_scoped_by_trace():
    rows = [
        span("Первый", "Ответ 1", trace="t1", identity="same"),
        span(
            "Внутренний",
            "Черновик",
            trace="t1",
            identity="child",
            parent="same",
            kind="start_agent",
        ),
        span("Второй", "Ответ 2", trace="t2", identity="same", parent="remote"),
    ]
    assert extract(*rows).turns.agent_response.tolist() == ["Ответ 1", "Ответ 2"]


def test_ambiguous_object_does_not_choose_a_text_leaf():
    result = extract(span("Вопрос", {"one": "yes", "two": "Другой ответ"}))
    assert result.turns.empty
    assert result.report["ambiguous_rows"] == 1


def test_tool_arguments_do_not_become_answer():
    result = extract(
        span(
            "Вопрос",
            {
                "messages": [
                    {
                        "type": "ai",
                        "content": "",
                        "tool_calls": [
                            {"name": "search", "args": {"question": "Текст аргумента"}}
                        ],
                    }
                ]
            },
        )
    )
    assert result.turns.empty


def test_ids_cannot_be_empty():
    result = extract(span("Вопрос", "Ответ", identity=""))
    assert result.turns.empty
    assert result.report["invalid_span_rows"] == 1


def test_ports_stay_unchanged_and_settings_are_explicit():
    descriptor = json.loads((Path(__file__).parents[1] / "descriptor.json").read_text())
    assert [p["name"] for p in descriptor["ports"]] == [
        "monitoring_traces",
        "monitoring_metric",
        "traces_validation_result",
        "selection",
        "monitoring_umr",
        "processing_report",
        "parquet_test_dataset",
        "umr_artifact",
        "settings",
    ]
    assert list(inspect.signature(main).parameters) == [
        "monitoring_traces",
        "monitoring_metric",
        "traces_validation_result",
        "selection",
        "ignore_traces_checks",
        "min_extraction_coverage",
        "llm_mode",
        "model_id",
        "llm_max_calls",
        "llm_budget_seconds",
    ]
    settings = {
        item["parameter"]: item
        for item in descriptor["ui"]["settings"][0]["components"][0]["config"]["components"]
    }
    signature = inspect.signature(main)
    for name, ui_type in (
        ("llm_mode", "string"), ("model_id", "string"),
        ("llm_max_calls", "integer"), ("llm_budget_seconds", "integer"),
    ):
        assert settings[name]["type"] == ui_type
        assert settings[name]["defaultValue"] == signature.parameters[name].default
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def envelope(
    text, *, performative="request", sender="caller", receiver="service", key="request"
):
    return {
        "performative": performative,
        "sender": sender,
        "receiver": receiver,
        "conversation_id": "conversation",
        "reply_with" if performative == "request" else "in_reply_to": key,
        "content": {"message": [{"type": "text", "value": text}]},
    }


def test_conflicting_parent_cannot_promote_internal_child():
    result = extract(
        span("Вопрос", "Ответ 1"),
        span("Вопрос", "Ответ 2"),
        span(
            "Внутренний запрос",
            "Внутренний ответ",
            identity="child",
            parent="s",
            kind="start_agent",
        ),
    )
    assert result.turns.empty
    assert result.report["blocked_boundary_rows"] == 1
    assert "ancestor_span_conflict" in result.report["issue_counts"]


def test_internal_inform_with_plain_ack_is_not_a_dialogue():
    result = extract(
        span(envelope("Результат инструмента", performative="inform"), "Принято")
    )
    assert result.turns.empty
    assert result.report["unresolved_fipa_rows"] == 1


def test_nested_protocol_conversation_is_not_promoted_to_external():
    root = span(
        envelope("Результат", performative="inform", sender="worker"),
        "Принято",
        identity="root",
    )
    child = span(
        envelope("Внутренний запрос", sender="worker"),
        envelope(
            "Внутренний ответ",
            performative="inform",
            sender="service",
            receiver="worker",
        ),
        identity="child",
        parent="root",
        kind="chain",
    )
    result = extract(root, child)
    assert result.turns.empty
    assert result.report["inner_boundary_rows"] == 1


@pytest.mark.parametrize(
    "container", [lambda x: x, lambda x: [x], lambda x: {"wrapped": [x]}]
)
def test_forwarded_requests_are_never_answers(container):
    result = extract(
        span(
            container(envelope("Вопрос")),
            container(envelope("Внутренний запрос", sender="service", receiver="tool")),
        )
    )
    assert result.turns.empty
    assert result.report["entry_without_exit"] == 1


@pytest.mark.parametrize("parent", ["", "entry"])
def test_multiple_envelopes_cannot_close_another_request(parent):
    entry = span(
        envelope("Вопрос"),
        envelope("Передано", sender="service", receiver="worker"),
        identity="entry",
    )
    answer = span(
        [envelope("Один"), envelope("Другой", key="different")],
        envelope(
            "Непроверенный ответ",
            performative="inform",
            sender="service",
            receiver="caller",
        ),
        identity="exit",
        parent=parent,
    )
    result = extract(entry, answer)
    assert result.turns.empty
    assert result.report["ambiguous_envelope_rows"] == 1


def test_ambiguous_parent_cannot_promote_internal_fipa_child():
    parent = span(
        [envelope("Один"), envelope("Другой", key="different")], None, identity="root"
    )
    child = span(
        envelope("Внутренний запрос", sender="worker"),
        envelope(
            "Внутренний ответ",
            performative="inform",
            sender="service",
            receiver="worker",
        ),
        identity="child",
        parent="root",
        kind="chain",
    )
    result = extract(parent, child)
    assert result.turns.empty
    assert result.report["issue_counts"] == {
        "ancestor_envelopes_ambiguous": 1,
        "envelopes_ambiguous": 1,
    }


@pytest.mark.parametrize("performative", ["agree", "", "request"])
def test_nonterminal_fipa_messages_do_not_complete_turn(performative):
    response = envelope(
        "Промежуточный ответ",
        performative=performative,
        sender="service",
        receiver="caller",
    )
    assert extract(span(envelope("Вопрос"), response)).turns.empty


def test_broken_parent_cycle_is_reported():
    result = extract(
        span("Один", "Ответ", identity="a", parent="b"),
        span("Другой", "Ответ", identity="b", parent="a"),
    )
    assert result.turns.empty
    assert result.report["issue_counts"]["parent_cycle"] == 2


@pytest.mark.parametrize(
    "role_key,role", [("role", "user"), ("type", "human"), ("speaker_type", "CUSTOMER")]
)
def test_message_roles_survive_wrappers_and_metadata(role_key, role):
    query = '"Цитата" и вопрос\n[это текст, не JSON'
    messages = [{role_key: role, "text": query, "phrase_id": "id", "time": "Вчера"}]
    request = {
        "arbitrary": {
            "history": messages,
            "copied_question": query,
            "metadata": {"debug": "debug"},
        }
    }
    result = extract(span(request, {"role": "assistant", "content": "42"}))
    assert result.turns.input_query.tolist() == [query]
    assert result.turns.agent_response.tolist() == ["42"]


def test_latest_tool_call_does_not_reuse_previous_assistant_answer():
    body = {
        "history": [
            {"role": "assistant", "content": "Старый ответ"},
            {"role": "user", "content": "Новый вопрос"},
            {"role": "assistant", "content": "", "tool_calls": [{"args": "Поиск"}]},
        ]
    }
    assert project_body(body, side="response").status == "empty"


def test_transport_reply_takes_precedence_over_internal_state():
    body = {
        "arbitrary_wrapper": {
            "receiver": "caller",
            "content": {
                "message": [
                    {"type": "widget", "value": "Это не текст"},
                    {"type": "text", "value": "Ответ"},
                ]
            },
        },
        "internal_state": {"messages": [{"role": "assistant", "content": "Черновик"}]},
    }
    assert extract(span("Вопрос", body)).turns.agent_response.tolist() == ["Ответ"]


def test_legacy_columns_and_timestamp_are_normalized():
    row = span("Вопрос", "Ответ")
    row["start_time_ns"] = pd.Timestamp("2026-09-01")
    legacy = {
        "trace_id": "traceid",
        "span_id": "spanid",
        "parent_span_id": "parentspanid",
        "start_time_ns": "starttimeunixnano",
        "span_name": "name",
    }
    result = extract_turns(
        pd.DataFrame([{legacy.get(k, k): v for k, v in row.items()}]),
        ExtractionConfig(),
    )
    assert result.turns.entry_time_ns.tolist() == [pd.Timestamp("2026-09-01").value]
    row["traceid"] = "contradiction"
    with pytest.raises(TraceExtractionError, match="traceid.*trace_id"):
        extract(row)


def test_issue_counts_are_not_truncated_with_examples():
    rows = [span("Вопрос", {}, identity=str(i)) for i in range(103)]
    result = extract_turns(pd.DataFrame(rows), ExtractionConfig(max_issue_examples=2))
    assert result.report["issue_counts"] == {"boundary_pair_incomplete": 103}
    assert result.report["issue_rows"] == 103
    assert result.report["issue_examples_truncated"] is True
    assert len(result.issues) == 2


def test_row_order_and_duplicate_index_do_not_change_turns():
    rows = [
        span("Первый", "Ответ 1", identity="a"),
        span("Второй", "Ответ 2", identity="b"),
    ]
    expected = extract(*rows).turns
    actual = extract_turns(
        pd.DataFrame(rows[::-1], index=[0, 0]), ExtractionConfig()
    ).turns
    pd.testing.assert_frame_equal(actual, expected)


def test_json_serialization_does_not_turn_replicas_into_conflicts():
    first = span({"text": "Вопрос", "metadata": {"id": 1}}, {"answer": "Ответ"})
    second = {
        **first,
        "input_text": json.dumps({"metadata": {"id": 1}, "text": "Вопрос"}, indent=2),
    }
    result = extract(first, second)
    assert len(result.turns) == 1
    assert result.report["duplicate_span_rows"] == 1


def test_null_transport_body_is_empty_but_null_message_content_is_text():
    assert extract(span("Вопрос", None)).turns.empty
    assert extract(
        span("Вопрос", {"role": "assistant", "content": "null"})
    ).turns.agent_response.tolist() == ["null"]


@pytest.mark.parametrize(
    "answer", ['"Да", ответил он', "[это текст]", "{не JSON", "null"]
)
@pytest.mark.parametrize("wrapped", [True, False])
def test_json_string_values_are_not_mistaken_for_broken_containers(answer, wrapped):
    response = {"answer": answer} if wrapped else answer
    assert extract(span("Вопрос", response)).turns.agent_response.tolist() == [answer]


def test_plain_fipa_content_preserves_brackets_and_quotes():
    incoming = {**envelope(""), "content": "[вопрос]"}
    outgoing = {
        **envelope("", performative="inform", sender="service", receiver="caller"),
        "content": '"Ответ", без изменений',
    }
    result = extract(span(incoming, outgoing))
    assert result.turns[["input_query", "agent_response"]].values.tolist() == [
        ["[вопрос]", '"Ответ", без изменений']
    ]


def test_message_bodies_and_context_ids_cannot_replace_span_columns():
    bodies = pd.DataFrame(
        [
            {
                "input_text": "Вопрос",
                "output_text": "Ответ",
                "agent_id": "arbitrary-agent",
                "metadata": {"session_id": "context"},
            }
        ]
    )
    with pytest.raises(TraceExtractionError, match="обязательные колонки"):
        extract_turns(bodies, ExtractionConfig())


@pytest.mark.parametrize("answer", ['"Ответ"', "'Ответ'"])
def test_encoded_text_keeps_literal_quotes(answer):
    assert extract(span("Вопрос", answer)).turns.agent_response.tolist() == [answer]


def test_message_content_keeps_literal_json_container():
    answer = '["Ответ"]'
    assert extract(
        span("Вопрос", {"role": "assistant", "content": answer})
    ).turns.agent_response.tolist() == [answer]


def test_double_encoded_container_is_projected():
    response = json.dumps({"role": "assistant", "content": "Ответ"})
    assert extract(span("Вопрос", response)).turns.agent_response.tolist() == ["Ответ"]


def test_double_encoded_fipa_is_detected_without_aef_boundary_kind():
    request = json.dumps(envelope("Вопрос"))
    response = json.dumps(
        envelope("Ответ", performative="inform", sender="service", receiver="caller")
    )
    result = extract(span(request, response, kind="chain"))
    assert result.turns[["input_query", "agent_response"]].values.tolist() == [
        ["Вопрос", "Ответ"]
    ]


def test_request_cannot_be_borrowed_through_conflicting_parent():
    root = span(None, "Ответ", identity="root")
    bridge = span(None, None, identity="bridge", parent="root", kind="chain")
    child = span(
        "Непроверенный запрос",
        "",
        identity="child",
        parent="bridge",
        kind="start_agent",
    )
    result = extract(root, bridge, {**bridge, "output_text": "conflict"}, child)
    assert result.turns.empty
    assert result.report["issue_counts"]["ancestor_span_conflict"] == 1


def test_self_request_with_plain_response_is_not_external():
    result = extract(
        span(
            envelope("Внутренний запрос", sender="service", receiver="service"), "Ответ"
        )
    )
    assert result.turns.empty


def test_equal_timestamps_on_separate_spans_do_not_establish_initiator():
    incoming = span(
        envelope("Запрос участника", sender="peer"),
        envelope("Ответ", performative="inform", sender="service", receiver="peer"),
        identity="in",
    )
    outgoing = span(
        None,
        envelope("Запрос участнику", sender="service", receiver="peer", key="other"),
        identity="out",
    )
    result = extract(incoming, outgoing)
    assert result.turns.empty
    assert result.report["counterparts"] == []


def test_subconversation_clarification_is_not_an_external_turn():
    dispatch = envelope("Передано", sender="service", receiver="worker", key="sub")
    dispatch["conversation_id"] = "subconversation"
    clarification = envelope(
        "Уточнение", sender="worker", receiver="service", key="clarify"
    )
    clarification["conversation_id"] = "subconversation"
    clarification_answer = {
        **clarification,
        "performative": "inform",
        "sender": "service",
        "receiver": "worker",
        "in_reply_to": "clarify",
    }
    returned = {
        **dispatch,
        "performative": "inform",
        "sender": "worker",
        "receiver": "service",
        "in_reply_to": "sub",
    }
    rows = [
        span(envelope("Вопрос"), dispatch, identity="entry"),
        {
            **span(clarification, clarification_answer, identity="internal"),
            "start_time_ns": 110,
        },
        {
            **span(
                returned,
                envelope(
                    "Ответ", performative="inform", sender="service", receiver="caller"
                ),
                identity="exit",
            ),
            "start_time_ns": 120,
        },
    ]
    result = extract(*rows)
    assert result.turns[["input_query", "agent_response"]].values.tolist() == [
        ["Вопрос", "Ответ"]
    ]
    assert result.report["counterparts"] == ["caller"]
    assert result.turns.route_chain_complete.tolist() == [True]


def test_fipa_turn_ids_escape_delimiters():
    rows = []
    for index, (conversation, request) in enumerate([("a|b", "c"), ("a", "b|c")]):
        incoming = {**envelope("Вопрос", key=request), "conversation_id": conversation}
        outgoing = {
            **envelope(
                "Ответ",
                performative="inform",
                sender="service",
                receiver="caller",
                key=request,
            ),
            "conversation_id": conversation,
        }
        rows.append(span(incoming, outgoing, identity=str(index)))
    result = extract(*rows)
    assert len(result.turns) == 2
    assert result.turns.turn_id.is_unique


def test_row_accounting_includes_every_selected_span():
    complete = span("Вопрос", "Ответ")
    rows = [
        complete,
        complete,
        span("Внутренний", "Ответ", identity="chain", kind="chain"),
        span("Вопрос", "Ответ", identity=""),
        span("Конфликт", "Один", identity="bad"),
        span("Конфликт", "Другой", identity="bad"),
    ]
    report = extract(*rows).report
    assert report["selected_agent_rows"] == sum(
        report[field]
        for field in (
            "duplicate_span_rows",
            "conflicting_span_rows",
            "invalid_span_rows",
            "non_candidate_rows",
            "candidate_rows_scanned",
        )
    )
    assert report["candidate_rows_scanned"] == sum(
        report[field]
        for field in (
            "inner_boundary_rows",
            "fipa_rows",
            "aef_candidate_boundaries",
            "ambiguous_envelope_rows",
            "blocked_boundary_rows",
        )
    )
