"""Production-нода: сырые AEF spans в доказательную monitoring UMR."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from hashlib import sha256
from io import BytesIO
import logging
import re
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit

import pandas as pd

from canonical import canonicalize_turns
from table_io import read_table
from trace_dialogue import ExtractionConfig, TraceExtractionError, extract_turns

logger = logging.getLogger(__name__)

# Те же методы и роли, что в MeasurementPlan laim-baskets-adapter.
_METHOD_ROLES: dict[str, dict[str, int | None]] = {
    "identity": {"final_score": 1},
    "accuracy": {"prediction": 1, "target": 1},
    "mean_criteria": {"criterion": None},
    "all_criteria": {"criterion": None},
    "majority": {"assessor_vote": None},
    "all_assessors": {"assessor_vote": None},
}
_MAX_ISSUE_EXAMPLES = 100
_DEFAULT_DISTRIBUTIVE = "monitoring"
_TRACE_QUALITY_FIELDS = (
    "null_violations",
    "missing",
    "trash_cells",
    "rule_violations",
    "blocking_rule_violations",
    "advisory_rule_violations",
    "duplicate_keys",
    "trace_breaks",
)
# Excel не принимает управляющие символы; в DataFrame и parquet текст не меняется.
_EXCEL_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_EXCEL_CELL_LIMIT = 32_767


def _clean_text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _validate_trace_check(value: object, ignore: bool) -> dict[str, Any]:
    if ignore:
        return {
            "source": "laim-ars-env-validation.verdict",
            "scope": "bypassed",
            "status": "bypassed",
            "non_gating_criteria": {},
        }
    if value is None:
        raise ValueError(
            "traces_validation_result не передан; для обхода проверки явно "
            "установите ignore_traces_checks=true"
        )
    if not isinstance(value, dict):
        raise ValueError("traces_validation_result должен быть JSON object")
    required = {"schema", "quality", "criteria", "readiness", "metrics"}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(
            "traces_validation_result должен содержать полный verdict-контракт; "
            f"отсутствуют поля: {missing}"
        )
    schema = value["schema"]
    if not isinstance(schema, dict):
        raise ValueError("traces_validation_result.schema должен быть JSON object")
    critical = schema.get("критичных нарушено")
    if isinstance(critical, bool) or not isinstance(critical, int):
        raise ValueError(
            "traces_validation_result.schema не содержит целое поле "
            "'критичных нарушено'"
        )
    if critical:
        raise ValueError("Трейсы не прошли критическую проверку схемы K1")

    quality = value.get("quality")
    if not isinstance(quality, list) or len(quality) != 1:
        raise ValueError("traces_validation_result.quality должен содержать один итог")
    quality_item = quality[0]
    if not isinstance(quality_item, dict):
        raise ValueError("traces_validation_result.quality[0] должен быть object")
    for field in _TRACE_QUALITY_FIELDS:
        count = quality_item.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                f"traces_validation_result.quality[0].{field} должен быть целым >= 0"
            )
    if quality_item["rule_violations"] != (
        quality_item["blocking_rule_violations"]
        + quality_item["advisory_rule_violations"]
    ):
        raise ValueError(
            "traces_validation_result.quality содержит несогласованные нарушения"
        )
    if quality_item.get("valid") is not True:
        raise ValueError("Трейсы некорректны и не прошли проверку качества")

    criteria = value["criteria"]
    if not isinstance(criteria, dict):
        raise ValueError("traces_validation_result.criteria должен быть JSON object")
    expected_criteria = {f"K{i}" for i in range(1, 9)}
    if not expected_criteria <= set(criteria):
        raise ValueError("traces_validation_result.criteria должен содержать K1–K8")
    for code in sorted(expected_criteria):
        criterion = criteria[code]
        if (
            not isinstance(criterion, dict)
            or criterion.get("tone") not in {"good", "warn", "bad"}
            or not _clean_text(criterion.get("result"))
            or not _clean_text(criterion.get("title"))
        ):
            raise ValueError(f"traces_validation_result.criteria.{code} некорректен")
    failed = [code for code in sorted(expected_criteria)[:6] if criteria[code]["tone"] == "bad"]
    if failed:
        raise ValueError(
            "Трейсы не прошли блокирующий контур K1–K6: " + ", ".join(failed)
        )

    readiness = value["readiness"]
    if (
        not isinstance(readiness, list)
        or not readiness
        or any(
            not isinstance(item, dict)
            or item.get("verdict") not in {"ready", "pilot", "not_ready"}
            for item in readiness
        )
    ):
        raise ValueError("traces_validation_result.readiness имеет неизвестный формат")
    metrics = value["metrics"]
    if (
        not isinstance(metrics, dict)
        or not isinstance(metrics.get("policy"), dict)
        or not isinstance(metrics.get("gates"), dict)
    ):
        raise ValueError("traces_validation_result.metrics не содержит policy и gates")
    warnings = [
        code for code in sorted(expected_criteria)[:6] if criteria[code]["tone"] == "warn"
    ]
    return {
        "source": "laim-ars-env-validation.verdict",
        "scope": "K1-K6",
        "status": "passed_with_warnings" if warnings else "passed",
        "warning_criteria": warnings,
        "non_gating_criteria": {code: criteria[code] for code in ("K7", "K8")},
        "readiness": readiness,
    }


def _validate_metric(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("monitoring_metric должен быть JSON object")
    if value.get("contract_version") != "laim-monitoring-metric.v2":
        raise ValueError("Ожидается laim-monitoring-metric.v2")
    if value.get("status") != "computed":
        raise ValueError("monitoring_metric должен иметь status=computed")
    if value.get("assessment_mode") not in {"qa", "dialogue", "turn_with_history"}:
        raise ValueError(
            "monitoring_metric.assessment_mode должен быть qa, dialogue или turn_with_history"
        )
    scoring = value.get("scoring")
    if not isinstance(scoring, dict):
        raise ValueError("monitoring_metric.scoring должен быть JSON object")
    if not isinstance(scoring.get("sources"), list) or not scoring["sources"]:
        raise ValueError(
            "monitoring_metric.scoring.sources должен быть непустым списком"
        )
    method = _clean_text(scoring.get("method"))
    if method not in _METHOD_ROLES:
        raise ValueError(f"Неизвестный monitoring_metric.scoring.method: {method!r}")
    role_counts: Counter[str] = Counter()
    columns = []
    source_ids = []
    for source in scoring["sources"]:
        if not isinstance(source, dict):
            raise ValueError(
                "Каждый monitoring_metric scoring source должен быть object"
            )
        role = _clean_text(source.get("role"))
        column = _clean_text(source.get("column_name"))
        source_id = _clean_text(source.get("source_id"))
        if not source_id or not role or not column:
            raise ValueError(
                "Каждый scoring source требует source_id, role и column_name"
            )
        role_counts[role] += 1
        columns.append(column)
        source_ids.append(source_id)
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source_id не может повторяться")
    if len(columns) != len(set(columns)):
        raise ValueError("Физическая scoring-колонка не может повторяться")
    expected_roles = _METHOD_ROLES[method]
    if set(role_counts) != set(expected_roles):
        raise ValueError(
            f"Метод {method} требует роли {sorted(expected_roles)}, "
            f"получено {sorted(role_counts)}"
        )
    for role, exact_count in expected_roles.items():
        if exact_count is not None and role_counts[role] != exact_count:
            raise ValueError(f"Метод {method} требует {exact_count} source роли {role}")
    return value


def _selection(selection: object) -> tuple[str, str, str]:
    """(agent_ci, distributive, solution_version) из порта selection."""
    if selection is None:
        return "", _DEFAULT_DISTRIBUTIVE, ""
    if not isinstance(selection, dict):
        raise ValueError("selection должен быть JSON object")
    return (
        _clean_text(selection.get("agent_ci")),
        _clean_text(selection.get("distributive")) or _DEFAULT_DISTRIBUTIVE,
        _clean_text(selection.get("solution_version")),
    )


def _artifact_scope(agent_id: str, distributive: str) -> str:
    parsed = urlsplit(distributive)
    source = parsed.path if parsed.scheme else distributive.split("?", 1)[0]
    source = source.split("#", 1)[0].rstrip("/")
    leaf = source.rsplit("/", 1)[-1]
    if leaf.casefold().endswith(".zip"):
        leaf = leaf[:-4]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", leaf).strip("-._") or "distributive"
    digest = sha256(source.encode("utf-8")).hexdigest()[:12]
    safe_agent = re.sub(r"[^A-Za-z0-9._-]+", "-", agent_id).strip("-._")
    return f"{safe_agent or 'agent'}_{slug[:80]}-{digest}"


# Листы «Формата тестового датасета»: monitoring-корзина публикуется в том же
# виде, что и эталонная корзина baskets-adapter.
FLAT_SHEET = "Вариант для отд. запросов"
DIALOGUE_SHEET = "Вариант для диалога"


def _truncate_cell(value: str) -> str:
    marker = f"…[обрезано в XLSX: {len(value)} символов]"
    return value[: _EXCEL_CELL_LIMIT - len(marker)] + marker


def _excel(
    frame: pd.DataFrame, assessment_mode: str
) -> tuple[pd.ExcelFile, list[dict[str, Any]]]:
    """XLSX-копия UMR и список ячеек, обрезанных до лимита Excel.

    Полный текст живёт в dataframe-порте; XLSX — артефакт для человека, и
    один сверхдлинный ответ не должен ронять ноду после извлечения (LAIM-0060).
    """
    export = frame.map(
        lambda value: (
            _EXCEL_ILLEGAL.sub(" ", value) if isinstance(value, str) else value
        )
    )
    truncated: list[dict[str, Any]] = []
    for column in export.columns:
        too_long = export[column].map(
            lambda value: isinstance(value, str) and len(value) > _EXCEL_CELL_LIMIT
        )
        for row_index in too_long[too_long].index:
            truncated.append(
                {"column": str(column), "row": int(row_index),
                 "length": len(export.loc[row_index, column])}
            )
            export.loc[row_index, column] = _truncate_cell(export.loc[row_index, column])
    sheet = DIALOGUE_SHEET if assessment_mode == "dialogue" else FLAT_SHEET
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        export.to_excel(writer, sheet_name=sheet, index=False)
        for row in writer.sheets[sheet].iter_rows():
            for cell in row:
                if isinstance(cell.value, str) and cell.value.startswith("="):
                    cell.data_type = "s"
    buffer.seek(0)
    return pd.ExcelFile(buffer), truncated


def _issue_summary(issues: pd.DataFrame) -> dict[str, Any]:
    if issues.empty:
        return {"counts": {}, "examples": [], "examples_truncated": False}
    counts = Counter(issues["issue_code"].astype(str))
    return {
        "counts": dict(sorted(counts.items())),
        "examples": issues.head(_MAX_ISSUE_EXAMPLES).to_dict(orient="records"),
        "examples_truncated": len(issues) > _MAX_ISSUE_EXAMPLES,
    }


def main(
    monitoring_traces: pd.DataFrame,
    monitoring_metric: dict,
    traces_validation_result: dict | None = None,
    selection: dict | None = None,
    ignore_traces_checks: bool = False,
) -> dict[str, Any]:
    """Извлечь протокольные turn и вернуть UMR с проверяемым отчётом."""
    if not isinstance(ignore_traces_checks, bool):
        raise ValueError("ignore_traces_checks должен быть bool")
    metric = _validate_metric(monitoring_metric)
    trace_check = _validate_trace_check(traces_validation_result, ignore_traces_checks)
    requested_agent, distributive, solution_version = _selection(selection)

    started = perf_counter()
    stage_started = perf_counter()
    spans = read_table(monitoring_traces, "monitoring_traces")
    read_seconds = perf_counter() - stage_started
    logger.info(
        "LAIM traces dataset converter: rows=%d, agent_id=%s, mode=%s",
        len(spans),
        requested_agent or "auto",
        metric["assessment_mode"],
    )

    stage_started = perf_counter()
    extracted = extract_turns(
        spans,
        ExtractionConfig(
            agent_id=requested_agent, max_issue_examples=_MAX_ISSUE_EXAMPLES
        ),
    )
    extraction_seconds = perf_counter() - stage_started
    if extracted.turns.empty:
        report = extracted.report
        raise TraceExtractionError(
            "Не найдено полных turn по поддерживаемым схемам: "
            f"candidate_turn_keys={report['candidate_turn_keys']}, "
            f"entry_without_exit={report['entry_without_exit']}, "
            f"exit_without_entry={report['exit_without_entry']}, "
            f"unsupported_trace_count={report['unsupported_trace_count']}"
        )
    unresolved_turns = (
        extracted.report["candidate_turn_keys"] - extracted.report["complete_turns"]
    )
    partial_source = bool(
        unresolved_turns
        or extracted.report["unsupported_trace_count"]
        or extracted.report["conflicting_entry_keys"]
        or extracted.report["conflicting_exit_keys"]
    )
    if partial_source:
        logger.warning(
            "Неполное извлечение turn: публикуются только доказанные complete turn "
            "(%d из %d)",
            extracted.report["complete_turns"],
            extracted.report["candidate_turn_keys"],
        )

    stage_started = perf_counter()
    canonical = canonicalize_turns(
        extracted.turns,
        monitoring_metric=metric,
        extraction_report=extracted.report,
    )
    canonical_seconds = perf_counter() - stage_started
    monitoring_umr = canonical.result
    if solution_version:
        monitoring_umr.insert(0, "solution_version", solution_version)

    stage_started = perf_counter()
    parquet_bytes = monitoring_umr.to_parquet(index=False)
    excel_result, excel_truncated = _excel(monitoring_umr, metric["assessment_mode"])
    serialization_seconds = perf_counter() - stage_started
    total_seconds = perf_counter() - started

    agent_id = extracted.report["agent_id"]
    ready = canonical.report["ready_for_scoring"]
    status = "partial" if partial_source else ("complete" if ready else "not_ready")
    warnings = []
    if canonical.report["missing_scoring_sources"]:
        warnings.append(
            "UMR не готова к прямому scoring: отсутствуют источники "
            + ", ".join(canonical.report["missing_scoring_sources"])
        )
    if partial_source:
        warnings.append(
            "Неполное извлечение: опубликованы только доказанные complete turn"
        )
    if excel_truncated:
        warnings.append(
            f"XLSX: {len(excel_truncated)} ячеек длиннее лимита Excel "
            f"{_EXCEL_CELL_LIMIT} обрезаны с пометкой; полный текст — в dataframe-порте"
        )

    settings = {
        "fileSystemPath": (
            "hdfs://arnsdpsbx/tmp/traces_based_datasets/"
            f"{_artifact_scope(agent_id, distributive)}"
        ),
        "fileName": f"monitoring_umr_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.xlsx",
        "rewrite": True,
        "addPostfix": False,
        "structured": False,
    }
    processing_report = {
        "contract_version": "laim-monitoring-trace-converter.v2",
        "status": status,
        "agent_id": agent_id,
        "assessment_mode": metric["assessment_mode"],
        "ready_for_scoring": ready,
        "warnings": warnings,
        "traces_validation": trace_check,
        "semantics": {
            "input_query": "user transport text",
            "output_answer": "final agent response",
            "scenario": "observed routing label, separate from the answer",
            "unknown_schema_policy": "report_and_publish_complete_turns",
        },
        "stage_timings_seconds": {
            "read_input": round(read_seconds, 3),
            "extract_turns": round(extraction_seconds, 3),
            "canonicalization": round(canonical_seconds, 3),
            "serialization": round(serialization_seconds, 3),
            "total": round(total_seconds, 3),
        },
        "conservation": {
            "candidate_turn_keys": extracted.report["candidate_turn_keys"],
            "published_turns": int(extracted.report["complete_turns"]),
            "published_rows": int(len(monitoring_umr)),
            "unpublished_turns": int(unresolved_turns),
            "extraction_coverage": extracted.report["extraction_coverage"],
        },
        "extraction": extracted.report,
        "issues": _issue_summary(extracted.issues),
        "semantic_profile": canonical.filter_report,
        "canonicalization": canonical.report,
        "serialization": {"excel_truncated_cells": excel_truncated},
    }
    logger.info(
        "LAIM traces dataset converter complete: status=%s, rows=%d, seconds=%.3f",
        status,
        len(monitoring_umr),
        total_seconds,
    )
    return {
        "monitoring_umr": monitoring_umr,
        "processing_report": processing_report,
        "parquet_test_dataset": parquet_bytes,
        "umr_artifact": excel_result,
        "settings": settings,
    }
