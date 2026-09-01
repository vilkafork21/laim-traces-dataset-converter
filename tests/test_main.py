from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from main import main


NODE_ROOT = Path(__file__).resolve().parents[1]


def _metric() -> dict[str, Any]:
    return {
        "contract_version": "laim-monitoring-metric.v2",
        "status": "computed",
        "assessment_mode": "qa",
        "scoring": {
            "method": "accuracy",
            "sources": [
                {
                    "source_id": "prediction",
                    "column_name": "class",
                    "role": "prediction",
                },
                {"source_id": "target", "column_name": "GT", "role": "target"},
            ],
        },
    }


def _trace_quality() -> dict[str, object]:
    criteria = {
        f"K{i}": {
            "result": "пройден",
            "tone": "good",
            "title": f"Критерий K{i}",
        }
        for i in range(1, 9)
    }
    criteria["K7"] = {
        "result": "нарушен · not_ready",
        "tone": "bad",
        "title": "Готовность к мониторингу",
    }
    criteria["K8"] = {
        "result": "критично",
        "tone": "bad",
        "title": "Диагностика структуры телеметрии",
    }
    return {
        "schema": {"критичных нарушено": 0},
        "quality": [
            {
                "null_violations": 0,
                "missing": 10,
                "trash_cells": 0,
                "rule_violations": 2,
                "blocking_rule_violations": 0,
                "advisory_rule_violations": 2,
                "duplicate_keys": 0,
                "trace_breaks": 0,
                "valid": True,
            }
        ],
        "criteria": criteria,
        "readiness": [{"verdict": "not_ready", "reason": "мало трейсов"}],
        "metrics": {"policy": {}, "gates": {}},
    }


def _span(
    trace_id: str,
    span_id: str,
    input_payload: dict[str, object],
    output_payload: dict[str, object],
    timestamp: int,
) -> dict[str, object]:
    return {
        "agent_id": "CI00000001",
        "trace_id": trace_id,
        "span_id": span_id,
        "session_id": "session-1",
        "start_time_ns": timestamp,
        "aef_kind": "input_request",
        "span_name": "POST /invoke",
        "input_text": json.dumps(input_payload, ensure_ascii=False),
        "output_text": json.dumps(output_payload, ensure_ascii=False),
    }


def _fipa_pair(
    answer: str = "Финальный ответ",
    *,
    question: str = "Вопрос",
    request_id: str = "request-1",
    domain_request_id: str = "domain-request-1",
    trace_suffix: str = "",
    start_time: int = 10,
) -> tuple[dict[str, object], dict[str, object]]:
    entry = _span(
        f"trace-entry{trace_suffix}",
        f"entry{trace_suffix}",
        {
            "message": {
                "sender": "agent_human",
                "receiver": "orchestrator",
                "conversation_id": "conversation-1",
                "reply_with": request_id,
                "content": {"message": [{"type": "text", "value": question}]},
            }
        },
        {
            "outgoing": {
                "receiver": "domain-agent",
                "reply_with": domain_request_id,
                "content": {"message": [{"type": "text", "value": "dispatch"}]},
            }
        },
        start_time,
    )
    exit_row = _span(
        f"trace-exit{trace_suffix}",
        f"exit{trace_suffix}",
        {
            "message": {
                "sender": "domain-agent",
                "receiver": "orchestrator",
                "conversation_id": "conversation-1",
                "in_reply_to": domain_request_id,
                "content": {"message": [{"type": "text", "value": "return"}]},
            }
        },
        {
            "outgoing": {
                "receiver": "agent_human",
                "in_reply_to": request_id,
                "content": {"message": [{"type": "text", "value": answer}]},
            }
        },
        start_time + 10,
    )
    return entry, exit_row


def test_node_builds_accuracy_view_without_monitoring_gt() -> None:
    entry, exit_row = _fipa_pair()

    result = main(
        pd.DataFrame([entry, exit_row]),
        _metric(),
        traces_validation_result=_trace_quality(),
    )

    row = result["monitoring_umr"].iloc[0]
    assert row["input_query"] == "Вопрос"
    assert row["output_answer"] == "Финальный ответ"
    assert row["class"] == "domain-agent"
    assert list(result["monitoring_umr"].columns) == [
        "scenario",
        "session_id",
        "query_id",
        "input_query_count",
        "input_query",
        "output_answer",
        "class",
    ]
    assert "GT" not in result["monitoring_umr"]
    assert result["processing_report"]["status"] == "complete"
    assert result["processing_report"]["ready_for_scoring"] is True
    trace_gate = result["processing_report"]["traces_validation"]
    assert trace_gate["scope"] == "K1-K6"
    assert trace_gate["non_gating_criteria"]["K7"]["tone"] == "bad"
    assert trace_gate["non_gating_criteria"]["K8"]["tone"] == "bad"


def test_trace_quality_result_is_required_by_default() -> None:
    with pytest.raises(ValueError, match="traces_validation_result не передан"):
        main(pd.DataFrame(), _metric())


def test_descriptor_requires_trace_validation_result() -> None:
    descriptor = json.loads((NODE_ROOT / "descriptor.json").read_text())
    port = next(
        item
        for item in descriptor["ports"]
        if item["name"] == "traces_validation_result"
    )

    assert port["required"] is True


def test_every_trace_quality_item_must_be_valid() -> None:
    quality = _trace_quality()
    quality["quality"][0]["valid"] = False

    with pytest.raises(ValueError, match="не прошли проверку качества"):
        main(pd.DataFrame(), _metric(), traces_validation_result=quality)


def test_legacy_quality_only_validation_result_is_rejected() -> None:
    with pytest.raises(ValueError, match="полный verdict-контракт"):
        main(
            pd.DataFrame(),
            _metric(),
            traces_validation_result={"quality": [{"valid": True}]},
        )


def test_bad_k1_k6_validation_criterion_is_rejected() -> None:
    quality = _trace_quality()
    quality["criteria"]["K2"]["tone"] = "bad"

    with pytest.raises(ValueError, match="K1–K6"):
        main(pd.DataFrame(), _metric(), traces_validation_result=quality)


def test_all_assessors_metric_is_accepted() -> None:
    metric = _metric()
    metric["scoring"] = {
        "method": "all_assessors",
        "sources": [
            {
                "source_id": f"source_{i}",
                "column_name": f"mark{i}",
                "role": "assessor_vote",
            }
            for i in (1, 2, 3)
        ],
    }
    entry, exit_row = _fipa_pair()

    result = main(
        pd.DataFrame([entry, exit_row]),
        metric,
        traces_validation_result=_trace_quality(),
    )

    assert result["processing_report"]["ready_for_scoring"] is True


def test_control_characters_do_not_break_excel_export() -> None:
    entry, exit_row = _fipa_pair(answer="Строка\x0bс управляющим символом")

    result = main(
        pd.DataFrame([entry, exit_row]),
        _metric(),
        traces_validation_result=_trace_quality(),
    )

    assert (
        result["monitoring_umr"].iloc[0]["output_answer"]
        == "Строка\x0bс управляющим символом"
    )
    exported = pd.read_excel(result["umr_artifact"])
    assert exported.iloc[0]["output_answer"] == "Строка с управляющим символом"


def test_excel_export_keeps_leading_equals_as_text() -> None:
    entry, exit_row = _fipa_pair(answer='=HYPERLINK("https://example.test")')

    result = main(
        pd.DataFrame([entry, exit_row]),
        _metric(),
        traces_validation_result=_trace_quality(),
    )

    workbook = result["umr_artifact"].book
    cell = workbook[result["umr_artifact"].sheet_names[0]]["F2"]
    assert cell.value == '=HYPERLINK("https://example.test")'
    assert cell.data_type == "s"


def test_excel_export_truncates_cell_over_excel_limit() -> None:
    # Один слишком длинный ответ не должен ронять ноду после успешного
    # извлечения (аудит LAIM-0060): dataframe-порт несёт полный текст, в
    # XLSX ячейка обрезается с пометкой, в отчёте — warning.
    entry, exit_row = _fipa_pair(answer="x" * 32_768)

    result = main(
        pd.DataFrame([entry, exit_row]),
        _metric(),
        traces_validation_result=_trace_quality(),
    )

    assert len(result["monitoring_umr"].iloc[0]["output_answer"]) == 32_768
    exported = pd.read_excel(result["umr_artifact"])
    cell = exported.iloc[0]["output_answer"]
    assert len(cell) <= 32_767 and cell.endswith("[обрезано в XLSX: 32768 символов]")
    assert any("XLSX" in warning for warning in result["processing_report"]["warnings"])
    assert result["processing_report"]["serialization"]["excel_truncated_cells"] == [
        {"column": "output_answer", "row": 0, "length": 32_768}
    ]


def test_incomplete_extraction_publishes_complete_turns() -> None:
    entry, exit_row = _fipa_pair()
    malformed = entry.copy()
    malformed["trace_id"] = "trace-malformed"
    malformed["span_id"] = "span-malformed"
    malformed["input_text"] = json.dumps(
        {
            "message": {
                "sender": "agent_human",
                "receiver": "orchestrator",
                "conversation_id": "conversation-malformed",
                "reply_with": "request-malformed",
                "content": {"message": []},
            }
        },
        ensure_ascii=False,
    )

    result = main(
        pd.DataFrame([entry, exit_row, malformed]),
        _metric(),
        traces_validation_result=_trace_quality(),
    )

    assert result["processing_report"]["status"] == "partial"
    assert result["processing_report"]["conservation"]["published_turns"] == 1
    assert result["processing_report"]["conservation"]["unpublished_turns"] == 1


def test_distributive_comes_from_selection() -> None:
    entry, exit_row = _fipa_pair()

    result = main(
        pd.DataFrame([entry, exit_row]),
        _metric(),
        traces_validation_result=_trace_quality(),
        selection={
            "agent_ci": "CI00000001",
            "distributive": "https://nexus.example/repository/CI00000001/"
            "CI00000001-D-01.002.03-distrib.zip?token=secret",
        },
    )

    artifact_path = result["settings"]["fileSystemPath"]
    assert artifact_path.startswith(
        "hdfs://arnsdpsbx/tmp/traces_based_datasets/CI00000001_"
    )
    assert "CI00000001-D-01.002.03-distrib" in artifact_path
    assert "secret" not in artifact_path


def test_solution_version_from_selection_is_published_in_flat_output() -> None:
    entry, exit_row = _fipa_pair()

    result = main(
        pd.DataFrame([entry, exit_row]),
        _metric(),
        traces_validation_result=_trace_quality(),
        selection={"solution_version": "D-01.002.03"},
    )

    frame = result["monitoring_umr"]
    assert frame.columns[0] == "solution_version"
    assert frame["solution_version"].tolist() == ["D-01.002.03"]
    assert pd.read_excel(result["umr_artifact"])["solution_version"].tolist() == [
        "D-01.002.03"
    ]


def test_dialogue_reports_published_turns_separately_from_rows() -> None:
    first_entry, first_exit = _fipa_pair()
    second_entry, second_exit = _fipa_pair(
        answer="Второй ответ",
        question="Второй вопрос",
        request_id="request-2",
        domain_request_id="domain-request-2",
        trace_suffix="-2",
        start_time=30,
    )
    metric = _metric()
    metric["assessment_mode"] = "dialogue"

    result = main(
        pd.DataFrame([first_entry, first_exit, second_entry, second_exit]),
        metric,
        traces_validation_result=_trace_quality(),
        selection={"solution_version": "D-01.002.03"},
    )

    frame = result["monitoring_umr"]
    assert list(frame.columns) == [
        "solution_version", "scenario", "session_id", "dialogue", "class",
    ]
    assert frame["solution_version"].tolist() == ["D-01.002.03"]
    assert ast.literal_eval(frame["dialogue"].iloc[0]) == [
        ("conversation-1|request-1", "Вопрос", "Финальный ответ"),
        ("conversation-1|request-2", "Второй вопрос", "Второй ответ"),
    ]
    conservation = result["processing_report"]["conservation"]
    assert conservation["published_turns"] == 2
    assert conservation["published_rows"] == 1
    assert conservation["unpublished_turns"] == 0


def test_qa_contract_keeps_repeated_session_as_flat_turns() -> None:
    first_entry, first_exit = _fipa_pair()
    second_entry, second_exit = _fipa_pair(
        answer="Второй ответ",
        question="Второй вопрос",
        request_id="request-2",
        domain_request_id="domain-request-2",
        trace_suffix="-2",
        start_time=30,
    )

    result = main(
        pd.DataFrame([first_entry, first_exit, second_entry, second_exit]),
        _metric(),
        traces_validation_result=_trace_quality(),
    )

    frame = result["monitoring_umr"]
    assert len(frame) == 2
    assert "dialogue" not in frame
    assert frame["session_id"].tolist() == ["session-1", "session-1"]
    assert frame["query_id"].tolist() == [
        "conversation-1|request-1", "conversation-1|request-2",
    ]
    assert result["processing_report"]["assessment_mode"] == "qa"
    conservation = result["processing_report"]["conservation"]
    assert conservation["published_turns"] == conservation["published_rows"] == 2
