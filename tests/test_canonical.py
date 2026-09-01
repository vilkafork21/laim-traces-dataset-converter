from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from canonical import MonitoringCanonicalizationError, canonicalize_turns
from trace_dialogue import EXTRACTION_CONTRACT


def _turn(turn_id: str = "conversation|request", session_id: str = "session-1",
          query: str = "Вопрос", answer: str = "Финальный ответ") -> dict[str, Any]:
    return {
        "turn_id": turn_id,
        "session_id": session_id,
        "input_query": query,
        "agent_response": answer,
        "route_label": "domain-agent",
        "schema_family": "fipa_acl",
        "schema_version": "fipa_acl_v1",
        "entry_trace_id": "trace-entry",
        "exit_trace_id": "trace-exit",
        "entry_span_id": "entry",
        "exit_span_id": "exit",
        "entry_time_ns": 10,
        "exit_time_ns": 20,
        "same_trace": False,
        "same_session": True,
        "route_chain_complete": True,
        "turn_latency_ms": 0.00001,
        "query_source_path": "input_text.message.content.message[0].value",
        "response_source_path": "output_text.outgoing.content.message[0].value",
        "route_source_path": "output_text.outgoing.receiver",
    }


def _turns() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "turn_id": "conversation|request",
                "session_id": "session-1",
                "input_query": "Вопрос",
                "agent_response": "Финальный ответ",
                "route_label": "domain-agent",
                "schema_family": "fipa_acl",
                "schema_version": "fipa_acl_v1",
                "entry_trace_id": "trace-entry",
                "exit_trace_id": "trace-exit",
                "entry_span_id": "entry",
                "exit_span_id": "exit",
                "entry_time_ns": 10,
                "exit_time_ns": 20,
                "same_trace": False,
                "same_session": True,
                "route_chain_complete": True,
                "turn_latency_ms": 0.00001,
                "query_source_path": "input_text.message.content.message[0].value",
                "response_source_path": "output_text.outgoing.content.message[0].value",
                "route_source_path": "output_text.outgoing.receiver",
            }
        ]
    )


def _metric(method: str = "mean_criteria") -> dict[str, Any]:
    sources = [
        {
            "source_id": "source_1",
            "column_name": "quality",
            "role": "criterion",
            "normalization": "numeric",
            "polarity": "direct",
        }
    ]
    if method == "accuracy":
        sources = [
            {
                "source_id": "source_1",
                "column_name": "class",
                "role": "prediction",
                "normalization": "label",
                "polarity": "direct",
            },
            {
                "source_id": "source_2",
                "column_name": "GT",
                "role": "target",
                "normalization": "label",
                "polarity": "direct",
            },
        ]
    return {
        "contract_version": "laim-monitoring-metric.v2",
        "status": "computed",
        "assessment_mode": "qa",
        "scoring": {"method": method, "sources": sources},
    }


def _report() -> dict[str, object]:
    return {
        "contract_version": EXTRACTION_CONTRACT,
        "complete_turns": 1,
        "candidate_turn_keys": 1,
    }


def test_output_answer_is_always_final_agent_response() -> None:
    result = canonicalize_turns(
        _turns(),
        monitoring_metric=_metric(),
        extraction_report=_report(),
    )

    row = result.result.iloc[0]
    assert row["output_answer"] == "Финальный ответ"
    assert row["scenario"] == "domain-agent"
    assert result.report["ready_for_scoring"] is True
    assert list(result.result.columns) == [
        "scenario",
        "session_id",
        "query_id",
        "input_query_count",
        "input_query",
        "output_answer",
    ]


def test_query_id_is_scoped_by_session() -> None:
    turns = pd.DataFrame([
        _turn("q1", "s1", "Первый вопрос", "Первый ответ"),
        _turn("q1", "s2", "Второй вопрос", "Второй ответ"),
    ])

    result = canonicalize_turns(
        turns,
        monitoring_metric=_metric(),
        extraction_report={**_report(), "complete_turns": 2, "candidate_turn_keys": 2},
    )

    assert result.result[["session_id", "query_id"]].values.tolist() == [
        ["s1", "q1"],
        ["s2", "q1"],
    ]


def test_query_id_duplicate_inside_session_is_rejected() -> None:
    turns = pd.DataFrame([
        _turn("q1", "s1", "Первый вопрос", "Первый ответ"),
        _turn("q1", "s1", "Второй вопрос", "Второй ответ"),
    ])

    with pytest.raises(
        MonitoringCanonicalizationError,
        match=r"\(session_id, turn_id\) повторяется",
    ):
        canonicalize_turns(
            turns,
            monitoring_metric=_metric(),
            extraction_report={
                **_report(),
                "complete_turns": 2,
                "candidate_turn_keys": 2,
            },
        )
def test_accuracy_exposes_observed_route_but_never_invents_target() -> None:
    result = canonicalize_turns(
        _turns(),
        monitoring_metric=_metric("accuracy"),
        extraction_report=_report(),
    )

    row = result.result.iloc[0]
    assert row["output_answer"] == "Финальный ответ"
    assert row["class"] == "domain-agent"
    assert "route_label" not in result.result
    assert "GT" not in result.result
    assert result.report["ready_for_scoring"] is True
    assert result.report["missing_scoring_sources"] == []
    assert result.report["reference_only_sources"] == ["GT"]


@pytest.mark.parametrize("column_name", ["output_answer", "turn_index"])
def test_accuracy_source_cannot_overwrite_umr_field(column_name: str) -> None:
    metric = _metric("accuracy")
    metric["scoring"]["sources"][0]["column_name"] = column_name

    with pytest.raises(MonitoringCanonicalizationError, match=column_name):
        canonicalize_turns(
            _turns(),
            monitoring_metric=metric,
            extraction_report=_report(),
        )


def test_accuracy_target_cannot_reuse_trace_field() -> None:
    metric = _metric("accuracy")
    metric["scoring"]["sources"][1]["column_name"] = "scenario"

    with pytest.raises(MonitoringCanonicalizationError, match="scenario"):
        canonicalize_turns(
            _turns(),
            monitoring_metric=metric,
            extraction_report=_report(),
        )


def test_flat_umr_matches_test_dataset_format() -> None:
    """qa/turn_with_history: колонки — только из формата тестового датасета."""
    result = canonicalize_turns(
        _turns(),
        monitoring_metric=_metric(),
        extraction_report=_report(),
    )

    assert list(result.result.columns) == [
        "scenario", "session_id", "query_id", "input_query_count",
        "input_query", "output_answer",
    ]
    assert "reference_group_id" not in result.result
    assert "turn_index" not in result.result


def test_dialogue_umr_is_packed_per_session() -> None:
    """dialogue: строка = сессия, реплики упакованы в dialogue-литерал."""
    import ast

    turns = pd.DataFrame([
        _turn("t1", "s1", "q1", "a1"),
        _turn("t2", "s1", "q2", "a2"),
        _turn("t3", "s2", "q3", "a3"),
    ])
    metric = _metric()
    metric["assessment_mode"] = "dialogue"
    report = {**_report(), "complete_turns": 3, "candidate_turn_keys": 3}

    result = canonicalize_turns(turns, monitoring_metric=metric, extraction_report=report)

    frame = result.result
    assert list(frame.columns) == ["scenario", "session_id", "dialogue"]
    assert frame["session_id"].tolist() == ["s1", "s2"]
    assert ast.literal_eval(frame["dialogue"].iloc[0]) == [("t1", "q1", "a1"), ("t2", "q2", "a2")]
    assert result.report["rows"] == 2
    assert result.report["groups"] == 2


def test_dialogue_accuracy_prediction_must_be_constant_per_session() -> None:
    """Меняющийся внутри сессии маршрут не публикуется как dialogue-предсказание."""
    turns = pd.DataFrame([
        _turn("t1", "s1", "q1", "a1"),
        _turn("t2", "s1", "q2", "a2"),
    ])
    turns.loc[1, "route_label"] = "other-agent"
    metric = _metric("accuracy")
    metric["assessment_mode"] = "dialogue"
    report = {**_report(), "complete_turns": 2, "candidate_turn_keys": 2}

    result = canonicalize_turns(turns, monitoring_metric=metric, extraction_report=report)

    assert "class" not in result.result
    assert result.report["missing_scoring_sources"] == ["class"]
    assert result.report["ready_for_scoring"] is False


def test_dialogue_scenario_varying_inside_session_is_dropped() -> None:
    """Меняющийся внутри сессии маршрут — не характеристика диалога."""
    turns = pd.DataFrame([
        _turn("t1", "s1", "q1", "a1"),
        _turn("t2", "s1", "q2", "a2"),
    ])
    turns.loc[1, "route_label"] = "other-agent"
    metric = _metric()
    metric["assessment_mode"] = "dialogue"
    report = {**_report(), "complete_turns": 2, "candidate_turn_keys": 2}

    result = canonicalize_turns(turns, monitoring_metric=metric, extraction_report=report)

    assert "scenario" not in result.result
    assert list(result.result.columns) == ["session_id", "dialogue"]
