"""Проекция доказанных trace-turn в контракт monitoring UMR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from trace_dialogue import EXTRACTION_CONTRACT, TURN_COLUMNS

CANONICALIZATION_CONTRACT = "laim-monitoring-turn-projection.v2"

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
        if not _blank_mask(scenario).any():
            flat[prediction_column] = scenario
            prediction_mapping = {
                "column_name": prediction_column,
                "source": "route_label",
                "source_paths": sorted(set(frame["route_source_path"].astype(str))),
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

    if assessment_mode == "dialogue":
        result, dropped_prediction = _pack_dialogue(flat, prediction_column)
        if dropped_prediction:
            prediction_mapping = None
            missing_scoring_sources.append(dropped_prediction)
    else:
        result = flat

    ready_for_scoring = not missing_scoring_sources

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


def _pack_dialogue(
    flat: pd.DataFrame, prediction_column: str | None
) -> tuple[pd.DataFrame, str | None]:
    """Строка = сессия по «Варианту для диалога»: реплики (в наблюдённом
    порядке) упакованы в dialogue-литерал троек, как в эталонной корзине.

    Возвращает packed-фрейм и имя prediction-колонки, если её пришлось
    опустить: предсказание, меняющееся внутри сессии, — не dialogue-значение.
    """
    records = []
    dropped_prediction = None
    scenario_constant = True
    for session_id, group in flat.groupby("session_id", sort=False):
        dialogue = [
            (str(row.query_id), str(row.input_query), str(row.output_answer))
            for row in group.itertuples()
        ]
        scenario_constant = scenario_constant and group["scenario"].nunique() == 1
        record = {
            "scenario": group["scenario"].iloc[0],
            "session_id": str(session_id),
            "dialogue": repr(dialogue),
        }
        if prediction_column is not None and prediction_column in flat:
            values = set(group[prediction_column])
            if len(values) == 1:
                record[prediction_column] = group[prediction_column].iloc[0]
            else:
                dropped_prediction = prediction_column
        records.append(record)
    result = pd.DataFrame(records)
    if not scenario_constant:
        result = result.drop(columns=["scenario"])
    if dropped_prediction and dropped_prediction in result:
        result = result.drop(columns=[dropped_prediction])
    return result, dropped_prediction
