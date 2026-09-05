"""Проекция доказанных trace-turn в контракт monitoring UMR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from trace_dialogue import EXTRACTION_CONTRACT, TURN_COLUMNS

CANONICALIZATION_CONTRACT = "laim-monitoring-turn-projection.v3"

# Поля UMR по «Формату тестового датасета» УМР (laim-umr.v2): scenario —
# классифицирующий признак запроса, для роутера это наблюдаемая метка маршрута.
_UMR_COLUMNS = {
    "session_id",
    "query_id",
    "input_query",
    "output_answer",
    "scenario",
    "input_query_count",
    "reference_group_id",
    "turn_index",
}


class MonitoringCanonicalizationError(ValueError):
    """Доказанные turn нельзя спроецировать без изменения их смысла."""


@dataclass(frozen=True)
class CanonicalizationResult:
    """Плоская monitoring UMR и доказательства её готовности."""

    result: pd.DataFrame
    report: dict[str, Any]
    filter_report: dict[str, Any]


def _blank_mask(values: pd.Series) -> pd.Series:
    return values.isna() | values.astype(str).str.strip().eq("")


def canonicalize_turns(
    turns: pd.DataFrame,
    *,
    monitoring_metric: dict[str, Any],
    extraction_report: dict[str, Any],
) -> CanonicalizationResult:
    """Спроецировать полные turn без смешения ответа и маршрута.

    monitoring_metric уже проверен нодой по контракту laim-monitoring-metric.v2.
    """
    if not isinstance(turns, pd.DataFrame) or turns.empty:
        raise MonitoringCanonicalizationError("Нет полных turn для monitoring UMR")
    if extraction_report.get("contract_version") != EXTRACTION_CONTRACT:
        raise MonitoringCanonicalizationError(
            "extraction_report имеет неизвестный контракт"
        )

    missing_columns = sorted(set(TURN_COLUMNS) - set(turns.columns))
    if missing_columns:
        raise MonitoringCanonicalizationError(
            f"Извлечённые turn не содержат колонки: {missing_columns}"
        )
    frame = turns.copy().reset_index(drop=True)
    for column in ("turn_id", "session_id", "input_query", "agent_response"):
        if _blank_mask(frame[column]).any():
            raise MonitoringCanonicalizationError(f"{column} содержит пустые значения")
    if frame[["session_id", "turn_id"]].astype(str).duplicated().any():
        raise MonitoringCanonicalizationError("(session_id, turn_id) повторяется")

    answers = frame["agent_response"].astype(str).str.strip()
    questions = frame["input_query"].astype(str).str.strip()
    assessment_mode = monitoring_metric["assessment_mode"]
    sources = monitoring_metric["scoring"]["sources"]
    method = monitoring_metric["scoring"]["method"]

    scenario = frame["route_label"].fillna("").astype(str).str.strip()
    flat = pd.DataFrame(
        {
            "scenario": scenario,
            "session_id": frame["session_id"].astype(str),
            "query_id": frame["turn_id"].astype(str),
            "input_query_count": 1,
            "input_query": questions,
            "output_answer": answers,
        }
    )

    flat["reference_group_id"] = flat["session_id"]
    flat["turn_index"] = flat.groupby("session_id", sort=False).cumcount() + 1
    flat["dataset_role"] = "monitoring"
    flat["definition_id"] = monitoring_metric["definition_id"]
    for column in TURN_COLUMNS:
        if column not in flat and column not in {"agent_response", "turn_id", "route_label"}:
            flat[column] = frame[column]
    flat["evaluation_evidence"] = "{}"
    # Профили пока доказывают только пару запрос/ответ и маршрут.
    # Историю и факты нельзя восстанавливать из произвольных внутренних JSON.
    missing_evidence = monitoring_metric["evaluation"]["required_evidence"]
    flat["evaluation_ready"] = not missing_evidence
    flat["evaluation_reason"] = ("Не поддержаны обязательные свидетельства: " + ", ".join(missing_evidence)) if missing_evidence else ""

    # Для accuracy monitoring поставляет только наблюдаемое prediction. Target
    # остаётся в эталонной корзине, где baskets-adapter вычисляет main_metric.
    prediction_mapping = None
    missing_scoring_sources: list[str] = []
    reference_only_sources: list[str] = []
    prediction_column = None
    if method == "accuracy":
        columns = {source["role"]: source["column_name"].strip() for source in sources}
        prediction_column = columns["prediction"]
        target_column = columns["target"]
        if prediction_column in _UMR_COLUMNS and prediction_column != "scenario":
            raise MonitoringCanonicalizationError(
                f"prediction-колонка {prediction_column!r} конфликтует с полем UMR"
            )
        if target_column in _UMR_COLUMNS:
            raise MonitoringCanonicalizationError(
                f"target-колонка {target_column!r} конфликтует с полем UMR"
            )
        observable = monitoring_metric["evaluation"]["prediction_observable"]
        observed = scenario if observable == "route_label" else answers
        if not _blank_mask(observed).any():
            flat[prediction_column] = observed
            prediction_mapping = {
                "column_name": prediction_column,
                "source": observable,
                "source_paths": sorted(set(frame["route_source_path" if observable == "route_label" else "response_source_path"].astype(str))),
            }
        else:
            missing_scoring_sources.append(prediction_column)
        reference_only_sources.append(target_column)

    candidate_turn_keys = extraction_report.get("candidate_turn_keys")
    complete_turns = extraction_report.get("complete_turns")
    if complete_turns != len(flat):
        raise MonitoringCanonicalizationError(
            f"extraction complete_turns={complete_turns} != UMR rows={len(flat)}"
        )
    if not isinstance(candidate_turn_keys, int) or candidate_turn_keys < complete_turns:
        raise MonitoringCanonicalizationError(
            "candidate_turn_keys противоречит complete_turns"
        )

    result = flat
    ready_for_scoring = not missing_scoring_sources and not missing_evidence
    result["evaluation_ready"] = ready_for_scoring

    repeated_counts = answers.value_counts()
    repeated_threshold = max(len(result) * 0.01, 5)
    repeated_values = set(repeated_counts[repeated_counts >= repeated_threshold].index)
    filter_report = {
        "contract_version": "laim-basket-semantic-profile.v2",
        "rows": int(len(result)),
        "exact_echo_pairs": int(
            questions.str.casefold().eq(answers.str.casefold()).sum()
        ),
        "repeated_answer_rows": int(answers.isin(repeated_values).sum()),
        "repeated_answer_values": int(len(repeated_values)),
        "policy": "observe_without_rewriting",
    }
    report = {
        "contract_version": CANONICALIZATION_CONTRACT,
        "assessment_mode": assessment_mode,
        "rows": int(len(result)),
        "groups": int(result["session_id"].nunique()),
        "candidate_turn_keys": int(candidate_turn_keys),
        "complete_turns": int(complete_turns),
        "source_coverage": complete_turns / candidate_turn_keys
        if candidate_turn_keys
        else 0.0,
        "response_semantics": "final_agent_response",
        "route_semantics": "separate_observed_label",
        "prediction_mapping": prediction_mapping,
        "missing_scoring_sources": missing_scoring_sources,
        "reference_only_sources": reference_only_sources,
        "ready_for_scoring": ready_for_scoring,
        "weight_policy": (
            "one_per_observed_dialogue"
            if assessment_mode == "dialogue"
            else "one_per_observed_turn"
        ),
    }
    return CanonicalizationResult(
        result=result,
        report=report,
        filter_report=filter_report,
    )
