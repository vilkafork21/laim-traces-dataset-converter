"""Источник корзины мониторинга: спаны трейсов -> вход конвертера.

Узел получает сырые спаны выгрузки и приводит агентские спаны к виду, который
ждёт конвертер: в input_text лежит вопрос пользователя, в output_text — ответ
агента. Конверт AEF (input_text/output_text) стандартный, но полезная нагрузка
внутри у каждого агента своя, поэтому вопрос и ответ достаются универсально:
диалоги, плоские поля, вложенный JSON, список ответов.

Для каждого trace отдельно выбирается доказуемая пара: готовый agent span,
совпадающая с ним внешняя граница либо явное terminal-поле финального ответа.
Неразрешённые trace изолируются и перечисляются в extraction_report, чтобы одна
нестандартная запись не останавливала обработку остальных.
"""

import glob
import io
import json
import logging
import os
import re
from collections import Counter
from time import perf_counter

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Тип таблицы опознаётся по сигнатуре файла, а не по расширению: части
# выгрузки приходят с именами без расширения. parquet начинается с PAR1,
# xlsx — это zip-контейнер.
_TABLE_READERS = {b"PAR1": pd.read_parquet, b"PK\x03\x04": pd.read_excel}


def _reader_for(head: bytes):
    """Функция чтения по сигнатуре; None — файл таблицей не является."""
    return next((read for signature, read in _TABLE_READERS.items()
                 if head.startswith(signature)), None)


def _signature(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read(8)


def _read_file(path: str, port: str) -> pd.DataFrame:
    read = _reader_for(_signature(path))
    if read is None:
        raise ValueError(f"Порт {port}: файл {path} не parquet и не xlsx")
    return read(path)


def read_table(value, port: str) -> pd.DataFrame:
    """Содержимое входного порта -> DataFrame.

    Форма зависит от того, чем порт наполнил предыдущий узел: готовым
    DataFrame, байтами файла, путём к файлу или каталогом с частями выгрузки.
    Требовать одну форму узел не вправе: связка узлов на холсте меняется, а
    данные за портом те же.
    """
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, (bytes, bytearray)):
        blob = bytes(value)
        read = _reader_for(blob[:8])
        if read is None:
            raise ValueError(f"Порт {port}: байты не parquet и не xlsx")
        return read(io.BytesIO(blob))
    if isinstance(value, str) and os.path.isfile(value):
        return _read_file(value, port)
    if isinstance(value, str) and os.path.isdir(value):
        # В каталоге выгрузки рядом с частями лежат служебные файлы (_SUCCESS,
        # контрольные суммы) — их отсеивает та же сигнатура.
        parts = [path for path in sorted(glob.glob(os.path.join(value, "**", "*"),
                                                   recursive=True))
                 if os.path.isfile(path) and _reader_for(_signature(path))]
        if not parts:
            raise FileNotFoundError(f"Порт {port}: в каталоге {value} нет parquet или xlsx")
        return pd.concat([_read_file(path, port) for path in parts], ignore_index=True)
    raise TypeError(f"Порт {port} отдал {type(value).__name__} — это не таблица")

# Виды спанов, которые в схеме AEF могут нести вопрос и ответ агента.
AGENT_SPAN_KINDS = ("start_agent", "input_request")
# Глубина разворота вложенного JSON при поиске значения.
_MAX_PAYLOAD_DEPTH = 5
_TERMINAL_ANSWER_KEYS = {
    "final_response",
    "message_to_user",
    "assistant_message",
    "generated_answer",
    "agent_answer",
    "agent_response",
    "full_answer",
}

_USER_ROLES = {"user", "human", "client", "customer", "end_user", "enduser",
               "пользователь", "клиент", "юзер", "абонент"}
_QUESTION_KEYS = ["user_query", "user_input", "user_question", "current_phrase",
                  "current_query", "query", "question", "incoming", "start_phrase",
                  "request", "utterance", "main_prompt", "text", "message"]
_ANSWER_KEYS = ["final_response", "message_to_user", "assistant_message",
                "generated_answer", "agent_answer", "agent_response", "answer",
                "response", "result", "reply", "output", "summary", "text",
                "message", "content"]
_DIALOG_KEYS = ["dialog", "dialogue", "messages", "history", "chat",
                "conversation", "turns", "dialogue_phrases"]
_ROLE_KEYS = ["role", "author", "sender", "from", "speaker", "type"]
_CONTENT_KEYS = ["content", "text", "message", "value", "msg", "body", "utterance"]


def _maybe_json(value):
    """Строку, похожую на JSON, разворачивает в объект; остальное отдаёт как есть."""
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in "{[":
            try:
                return json.loads(text)
            except ValueError:
                return value
    return value


def _turn(item):
    """Элемент диалога -> (текст, роль). Поддержаны dict и список [text, role, time]."""
    if isinstance(item, dict):
        role = next((str(item[key]).strip().lower()
                     for key in _ROLE_KEYS if key in item), "")
        text = next((item[key] for key in _CONTENT_KEYS
                     if isinstance(item.get(key), str) and item[key].strip()), "")
        return str(text).strip(), role
    if isinstance(item, (list, tuple)) and item:
        texts = [str(x) for x in item if isinstance(x, str)]
        role = next((t.lower() for t in texts if t.lower() in _USER_ROLES), "")
        return max(texts, key=len, default="").strip(), role
    return "", ""


def _question_from_dialog(payload: dict) -> str:
    """Склейка пользовательских реплик диалога — запасной источник вопроса."""
    for key in _DIALOG_KEYS:
        dialog = payload.get(key)
        if isinstance(dialog, list) and dialog:
            turns = [text for text, role in (_turn(x) for x in dialog)
                     if text and role in _USER_ROLES]
            if turns:
                return " ".join(turns)
    return ""


def _texts_from_list(items) -> str:
    """Тексты элементов списка.

    Строки берутся как есть, словари — по ключам содержимого: в схеме FIPA-ACL
    реплика лежит элементом вида {"type": "text", "value": ...}, и пропуск таких
    элементов оставлял бы вопрос и ответ ненайденными.
    """
    parts = []
    for item in items:
        item = _maybe_json(item)
        if isinstance(item, str) and item.strip():
            parts.append(item.strip())
        elif isinstance(item, dict):
            text = next((item[key].strip() for key in _CONTENT_KEYS
                         if isinstance(item.get(key), str) and item[key].strip()), "")
            if text:
                parts.append(text)
    return " ".join(parts)


def _deep(payload, keys, depth: int = 0) -> str:
    """Первое непустое строковое значение по ключам keys с разворотом вложенности."""
    payload = _maybe_json(payload)
    if depth > _MAX_PAYLOAD_DEPTH or not isinstance(payload, dict):
        return ""
    for key in keys:
        if key not in payload:
            continue
        value = _maybe_json(payload[key])
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            joined = _texts_from_list(value)
            if joined:
                return joined
        if isinstance(value, dict):
            found = _deep(value, keys, depth + 1)
            if found:
                return found
    for value in payload.values():
        nested = _maybe_json(value)
        if isinstance(nested, dict):
            found = _deep(nested, keys, depth + 1)
            if found:
                return found
    return ""


def extract_question(payload) -> str:
    """Вопрос пользователя из полезной нагрузки спана.

    Текущий запрос приоритетнее склейки всего диалога: main_prompt — это
    нормализованный запрос агента, user_question — текущая реплика. Иначе судья
    оценивал бы ответ против вопросов, которых в этом ходе не задавали.
    """
    payload = _maybe_json(payload)
    payload = payload if isinstance(payload, dict) else {}
    return (_deep(payload, ["main_prompt", "user_question"])
            or _question_from_dialog(payload)
            or _deep(payload, _QUESTION_KEYS))


def extract_answer(payload) -> str:
    """Ответ агента из полезной нагрузки спана."""
    payload = _maybe_json(payload)
    payload = payload if isinstance(payload, dict) else {}
    return _deep(payload, _ANSWER_KEYS)


def _pair_from_row(row: pd.Series) -> tuple[str, str]:
    """Вопрос и ответ строки без зависимости от индекса DataFrame."""
    incoming, outgoing = row.get("input_text", ""), row.get("output_text", "")
    question = extract_question(incoming) or extract_question(outgoing)
    answer = extract_answer(outgoing) or extract_answer(incoming)
    return question, answer


def _semantic_text(value: str) -> str:
    """Нормализованное представление только для сравнения, не для публикации."""
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def _distinct_pair(question: str, answer: str) -> bool:
    technical_answers = {
        "success", "ok", "done", "completed", "true", "false", "accepted",
    }
    return bool(
        question
        and answer
        and _semantic_text(question) != _semantic_text(answer)
        and _semantic_text(answer) not in technical_answers
    )


def _external_boundary(row: pd.Series) -> bool:
    """Явная входная граница агента, а не произвольный корневой span."""
    name = str(row.get("span_name", "")).strip().casefold()
    kind = str(row.get("aef_kind", "")).strip().casefold()
    return kind in {"input_request", "other"} and bool(
        re.match(r"^(get|post|put|patch|delete)\s+\S+", name)
        or re.search(r"(^|[./])invoke$", name)
    )


def _identifier(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _row_order(row: pd.Series) -> tuple:
    start = row.get("start_time_ns")
    try:
        start = int(start)
    except (TypeError, ValueError):
        start = 0
    return start, _identifier(row.get("span_id"))


def _primary_candidates(group: pd.DataFrame) -> list[tuple[int, pd.Series, str, str]]:
    candidates = []
    for index, row in group[group["aef_kind"].isin(AGENT_SPAN_KINDS)].iterrows():
        question, answer = _pair_from_row(row)
        if question:
            candidates.append((index, row, question, answer))
    priority = {"start_agent": 0, "input_request": 1}
    return sorted(
        candidates,
        key=lambda item: (
            priority.get(str(item[1].get("aef_kind")), 2),
            _row_order(item[1]),
        ),
    )


def _rows_by_span_id(group: pd.DataFrame) -> dict[str, tuple[int, pd.Series]]:
    result = {}
    for index, row in group.iterrows():
        span_id = _identifier(row.get("span_id"))
        if span_id and span_id not in result:
            result[span_id] = (index, row)
    return result


def _matching_ancestor(
    primary: tuple[int, pd.Series, str, str],
    rows_by_span_id: dict[str, tuple[int, pd.Series]],
) -> tuple[int, pd.Series, str, str] | None:
    """Ближайшая внешняя граница с тем же вопросом и отличающимся ответом."""
    _, row, question, _ = primary
    parent_id = _identifier(row.get("parent_span_id"))
    visited = set()
    while parent_id and parent_id not in visited and parent_id in rows_by_span_id:
        visited.add(parent_id)
        index, parent = rows_by_span_id[parent_id]
        parent_question, parent_answer = _pair_from_row(parent)
        if (
            _external_boundary(parent)
            and _distinct_pair(parent_question, parent_answer)
            and _semantic_text(parent_question) == _semantic_text(question)
        ):
            return index, parent, question, parent_answer
        parent_id = _identifier(parent.get("parent_span_id"))
    return None


def _terminal_values(value, path: str = "$") -> list[tuple[str, str]]:
    """Строки только из полей, явно предназначенных для финального ответа."""
    value = _maybe_json(value)
    if isinstance(value, dict):
        found = []
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in _TERMINAL_ANSWER_KEYS:
                text = _terminal_field_text(child)
                if text:
                    found.append((text, child_path))
            if isinstance(_maybe_json(child), dict):
                found.extend(_terminal_values(child, child_path))
        return found
    return []


def _terminal_field_text(value) -> str:
    value = _maybe_json(value)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for item in value:
            item = _maybe_json(item)
            if isinstance(item, str) and item.strip():
                return item.strip()
            if isinstance(item, dict):
                text = _deep(item, _CONTENT_KEYS)
                if text:
                    return text
    if isinstance(value, dict):
        return _deep(value, _CONTENT_KEYS)
    return ""


def _terminal_answer(
    group: pd.DataFrame,
    question: str,
) -> tuple[str, str] | None:
    """Единственный ответ из семантически сильного terminal-поля trace."""
    variants: dict[str, dict] = {}
    for _, row in group.iterrows():
        for answer, path in _terminal_values(row.get("output_text", "")):
            key = _semantic_text(answer)
            if not key or key == _semantic_text(question):
                continue
            item = variants.setdefault(key, {
                "answer": answer,
                "sources": set(),
            })
            item["sources"].add(f"{row.get('span_name', '')}:{path}")
    if len(variants) != 1:
        return None
    only = next(iter(variants.values()))
    return only["answer"], ", ".join(sorted(only["sources"])[:3])


def _unique_resolution(
    candidates: list[tuple[int, pd.Series, str, str]],
) -> tuple[int, pd.Series, str, str] | None:
    variants = {
        (_semantic_text(question), _semantic_text(answer))
        for _, _, question, answer in candidates
    }
    if len(variants) != 1:
        return None
    return sorted(candidates, key=lambda item: _row_order(item[1]))[0]


def _resolve_group(group: pd.DataFrame) -> dict | None:
    """Одна доказуемая пара для trace; вопрос и ответ могут быть из разных spans."""
    primary = _primary_candidates(group)
    by_span_id = _rows_by_span_id(group)

    direct = [candidate for candidate in primary if _distinct_pair(*candidate[2:])]
    selected_direct = _unique_resolution(direct) if direct else None
    if selected_direct is not None:
        index, row, question, answer = selected_direct
        return {
            "target_index": index,
            "question": question,
            "answer": answer,
            "method": "direct_agent_span",
            "source_span": str(row.get("span_name", "")),
        }
    if direct:
        return None

    if primary:
        boundaries = []
        for candidate in primary:
            boundary = _matching_ancestor(candidate, by_span_id)
            if boundary is not None:
                _, source, question, answer = boundary
                boundaries.append((candidate[0], source, question, answer))
        boundary = _unique_resolution(boundaries) if boundaries else None
        if boundary is not None:
            target_index, source, question, answer = boundary
            return {
                "target_index": target_index,
                "question": question,
                "answer": answer,
                "method": "parent_boundary",
                "source_span": str(source.get("span_name", "")),
            }

        questions = {_semantic_text(candidate[2]) for candidate in primary}
        question = primary[0][2]
        terminal = _terminal_answer(group, question) if len(questions) == 1 else None
        if terminal is not None:
            answer, source_span = terminal
            return {
                "target_index": primary[0][0],
                "question": question,
                "answer": answer,
                "method": "semantic_terminal",
                "source_span": source_span,
            }

    external = []
    for index, row in group.iterrows():
        if not _external_boundary(row):
            continue
        question, answer = _pair_from_row(row)
        if _distinct_pair(question, answer):
            external.append((index, row, question, answer))
    boundary = _unique_resolution(external) if external else None
    if not primary and boundary is not None:
        index, row, question, answer = boundary
        return {
            "target_index": index,
            "question": question,
            "answer": answer,
            "method": "external_boundary",
            "source_span": str(row.get("span_name", "")),
        }
    return None


def _eligible_group(group: pd.DataFrame) -> bool:
    return bool(
        group["aef_kind"].isin(AGENT_SPAN_KINDS).any()
        or any(_external_boundary(row) for _, row in group.iterrows())
    )


def normalize_agent_spans(spans: pd.DataFrame) -> pd.DataFrame:
    """Приводит агентские спаны к виду input_text=вопрос, output_text=ответ.

    Вопрос и финальный ответ выбираются отдельно внутри каждого trace. Это
    сохраняет обычный start_agent/input_request, но позволяет восстановить
    финальный внешний ответ, когда start_agent содержит эхо входа. Неразрешённая
    группа не останавливает весь пакет: её carrier понижается до other и
    учитывается в extraction_report.
    """
    spans = spans.copy().reset_index(drop=True)
    spans["aef_kind"] = spans["aef_kind"].astype(str)
    spans["_laim_extraction_method"] = ""
    spans["_laim_answer_source_span"] = ""
    original_agent = spans["aef_kind"].isin(AGENT_SPAN_KINDS)

    group_columns = ["trace_id"]
    if "session_id" in spans.columns:
        group_columns.insert(0, "session_id")
    methods = Counter()
    resolutions = []
    published_indexes = []
    unresolved_trace_ids = []
    grouped = spans.groupby(group_columns, dropna=False, sort=False)
    total_groups = grouped.ngroups
    started = perf_counter()
    logger.info("Восстановление Q/A: найдено %d trace-групп", total_groups)
    for position, (_, group) in enumerate(grouped, start=1):
        if not _eligible_group(group):
            methods["non_eligible"] += 1
        else:
            resolution = _resolve_group(group)
            if resolution is None:
                methods["unresolved"] += 1
                spans.loc[group.index, "_laim_extraction_method"] = "unresolved"
                if len(unresolved_trace_ids) < 20:
                    unresolved_trace_ids.append(_identifier(group.iloc[0].get("trace_id")))
            else:
                published_indexes.extend(group.index.tolist())
                resolutions.append(resolution)
                methods[resolution["method"]] += 1
        if position % 500 == 0 or position == total_groups:
            logger.info(
                "Восстановление Q/A: %d/%d (%.1f%%); восстановлено=%d, "
                "не разрешено=%d; %.1f с",
                position,
                total_groups,
                position * 100 / total_groups,
                len(resolutions),
                methods["unresolved"],
                perf_counter() - started,
            )

    spans.loc[original_agent, "aef_kind"] = "other"
    for resolution in resolutions:
        index = resolution["target_index"]
        spans.at[index, "input_text"] = json.dumps(
            {"text": resolution["question"]}, ensure_ascii=False
        )
        spans.at[index, "output_text"] = json.dumps(
            {"answer": resolution["answer"]}, ensure_ascii=False
        )
        spans.at[index, "aef_kind"] = "start_agent"
        spans.at[index, "_laim_extraction_method"] = resolution["method"]
        spans.at[index, "_laim_answer_source_span"] = resolution["source_span"]

    selected_groups = len(resolutions)
    eligible_groups = selected_groups + methods["unresolved"]
    report = {
        "input_trace_groups": sum(methods.values()),
        "overall_selected_rate": (
            selected_groups / sum(methods.values()) if sum(methods.values()) else 0.0
        ),
        "eligible_trace_groups": eligible_groups,
        "selected_groups": selected_groups,
        "coverage": selected_groups / eligible_groups if eligible_groups else 0.0,
        "methods": dict(sorted(methods.items())),
        "unresolved_trace_ids": unresolved_trace_ids,
        "unresolved_ids_truncated": methods["unresolved"] > len(unresolved_trace_ids),
    }
    spans = spans.loc[published_indexes].copy().reset_index(drop=True)
    spans["_laim_order_session"] = (
        spans["session_id"].map(_identifier)
        if "session_id" in spans else ""
    )
    order_start = pd.to_numeric(
        spans["start_time_ns"]
        if "start_time_ns" in spans
        else pd.Series(index=spans.index, dtype=float),
        errors="coerce",
    )
    has_chronological_order = bool(len(spans)) and order_start.notna().all()
    spans["_laim_order_start"] = order_start.fillna(0)
    spans["_laim_order_trace"] = spans["trace_id"].map(_identifier)
    spans["_laim_order_span"] = spans["span_id"].map(_identifier)
    spans = spans.sort_values(
        ["_laim_order_session", "_laim_order_start", "_laim_order_trace", "_laim_order_span"],
        kind="stable",
    ).drop(columns=[
        "_laim_order_session", "_laim_order_start", "_laim_order_trace", "_laim_order_span",
    ]).reset_index(drop=True)
    report["order_policy"] = (
        "session_id,start_time_ns,trace_id,span_id"
        if has_chronological_order
        else "session_id,trace_id,span_id (timestamp unavailable)"
    )
    spans.attrs["laim_extraction_report"] = report
    return spans


def resolve_spans(spans, agent_id: str = "") -> dict:
    """Спаны выгрузки -> parquet агентских пар для конвертера трейсов.

    Parameters
    ----------
    spans : pd.DataFrame
        Сырые спаны окна выборки.
    agent_id_in : str
        Идентификатор агента с порта (приходит из настроек прогона).
    agent_id : str
        Идентификатор агента настройкой узла; используется, если порт не подключён.

    Returns
    -------
    dict
        Нормализованные spans в ``parquet`` и обезличенная диагностика
        извлечения в ``extraction_report``.
    """
    spans = read_table(spans, "spans")
    target = str(agent_id).strip()
    logger.info(f"Спанов на входе: {len(spans):,}")
    if spans.empty:
        raise ValueError(
            "Выгрузка пуста. Частая причина — окно прогона вне периода жизни "
            "версии дистрибутива: вне его витрина не отдаёт ни строки.")

    if "agent_id" not in spans.columns:
        raise ValueError("raw spans не содержат agent_id: identity не доказана")
    observed_agents = sorted({
        value for value in spans["agent_id"].astype(str).str.strip()
        if value and value.lower() != "nan"
    })
    if not target:
        if len(observed_agents) != 1:
            raise ValueError(
                "Без порта agent_id выгрузка должна содержать ровно одного агента; "
                f"найдено: {observed_agents}"
            )
        target = observed_agents[0]

    if target and "agent_id" in spans.columns:
        selected = spans["agent_id"].astype(str).str.strip().str.upper() == target.upper()
        spans = spans[selected]
        logger.info(f"Спанов агента {target}: {len(spans):,}")
        if spans.empty:
            raise ValueError(f"В выгрузке нет спанов агента {target}")

    parquet = normalize_agent_spans(spans)
    extraction_report = parquet.attrs["laim_extraction_report"]
    extraction_report["contract_version"] = "laim-trace-extraction.v1"
    extraction_report["agent_id"] = target.upper()
    logger.info(f"Извлечение пар: {extraction_report}")
    parquet = parquet.reset_index(drop=True)
    rows = int((parquet["aef_kind"] == "start_agent").sum())
    logger.info(f"Строк корзины: {rows:,}, трейсов: {parquet['trace_id'].nunique():,}")
    return {"parquet": parquet, "extraction_report": extraction_report}
