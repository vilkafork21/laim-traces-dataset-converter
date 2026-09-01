"""
Модуль для преобразования трейсов в тестовые датасеты.
"""
import json
import pandas as pd
from typing import Any


class DatasetCreator:
    """
    Класс для генерации тестовых датасетов из трейсов AEF SDK.

    Поддерживает два формата:
    - create_qa_dataset(): один вопрос-ответ на строку
    - create_dialog_dataset(): диалог с историей сообщений

    Форматы соответствуют требованиям из docs/Формат тестового датасета.xlsx

    Входной формат трейсов: JSON-список спанов с полями:
    - session_id (строка) - идентификатор сессии
    - trace_id (строка) - идентификатор трейса (используется как query_id)
    - span_id (строка) - идентификатор спана
    - parent_span_id (строка) - родительский спан
    - agent_id (строка) - идентификатор агента
    - aef_kind (строка) - тип спана (input_request, start_agent, llm, tool, retriever, guard, other)
    - span_name (строка) - название спана
    - input_text (строка) - входной текст
    - output_text (строка) - выходной текст

    Формат конфигурации датасета (config):
    - main_fields: list[dict] — основные поля датасета:
        - field_name (строка): название колонки
        - data_type (строка): тип данных (string, integer, float)
        - description (строка): описание колонки
      Обязательные поля: input_query, output_answer, session_id, query_id
    - end2end_metrics: list[dict] — метрики end-to-end (после output_answer):
        - field_name (строка): название колонки метрики
        - data_type (строка): тип данных (string, integer, float)
        - description (строка): описание колонки
    - modules: list[dict] — дополнительные модульные спаны с вложенными полями и метриками:
        - module_name (строка): название модуля
        - module_fields (list[dict]): поля спана для данного модуля:
            - field_name (строка): название колонки (совпадает с span_name из трейсов)
            - data_type (строка): тип данных
            - description (строка): описание колонки
        - module_metrics (list[dict]): метрики для данного модуля:
            - field_name (строка): название колонки метрики
            - data_type (строка): тип данных
            - description (строка): описание колонки
    """

    # Обязательные поля тестового датасета (всегда присутствуют)
    BASE_FIELDS = ["session_id", "query_id", "input_query", "output_answer"]

    def __init__(
        self,
        traces: list[dict],
        config: dict,
    ):
        """
        Инициализация создателя датасета.

        :param traces: Список спанов (каждый спан — словарь)
        :param config: Словарь конфигурации колонок датасета
        """
        self.traces: list[dict] = traces
        self.config: dict = config

        # Извлекаем main_fields и modules
        self.main_fields: list[dict] = self.config.get("main_fields", [])
        self.end2end_metrics: list[dict] = self.config.get("end2end_metrics", [])
        self.modules: list[dict] = self.config.get("modules", [])

        # Валидируем
        self._validate_config()

        # Список имён модулей для группировки по span_name в трейсах
        self.module_names: list[str] = [m['module_name'] for m in self.modules]

    def _validate_config(self) -> None:
        """
        Валидирует конфигурацию:
        - main_fields должны содержать все поля из BASE_FIELDS
        - field_name не может быть пустым
        - data_type должно быть string, integer или float
        - description не может быть пустым
        - end2end_metrics: каждый элемент — словарь с field_name, data_type, description
        - modules: каждый module должен иметь непустой module_name
        - внутри module_fields и module_metrics: field_name, data_type, description
        """
        if not self.main_fields:
            raise ValueError("main_fields в конфигурации не должен быть пустым")

        config_field_names: set[str] = set()
        for item in self.main_fields:
            field_name = item.get("field_name")
            data_type = item.get("data_type")
            description = item.get("description")

            if not field_name:
                raise ValueError("Каждый элемент main_fields должен иметь field_name")

            if not description:
                raise ValueError(f"Поле '{field_name}' должно иметь description")

            if data_type not in ["string", "integer", "float"]:
                raise ValueError(
                    f"Поле '{field_name}' имеет недопустимый data_type: {data_type}. "
                    "Допустимые значения: string, integer, float"
                )

            config_field_names.add(field_name)

        missing_required = set(self.BASE_FIELDS) - config_field_names
        if missing_required:
            raise ValueError(f"В main_fields отсутствуют обязательные поля: {missing_required}")

        # Валидация end2end_metrics
        for em in self.end2end_metrics:
            em_name = em.get("field_name", "")
            if not em_name:
                raise ValueError("Каждый элемент end2end_metrics должен иметь field_name")

            em_dtype = em.get("data_type")
            if em_dtype not in ["string", "integer", "float"]:
                raise ValueError(
                    f"Метрика '{em_name}' имеет недопустимый data_type: {em_dtype}. "
                    "Допустимые значения: string, integer, float"
                )

            em_desc = em.get("description", "")
            if not em_desc:
                raise ValueError(f"Метрика '{em_name}' должна иметь description")

        # Валидация modules
        for mod in self.modules:
            mod_name = mod.get("module_name", "")
            if not mod_name:
                raise ValueError("Каждый элемент modules должен иметь непустой module_name")

            # Проверяем, что спан с таким module_name есть в трейсах
            self._validate_span_exists(mod_name)

            # Проверяем module_fields
            for mf in mod.get("module_fields", []):
                fname = mf.get("field_name", "")
                if not fname:
                    raise ValueError(f"В модуле '{mod_name}' module_fields: field_name не может быть пустым")

            # Проверяем module_metrics
            for mm in mod.get("module_metrics", []):
                mname = mm.get("field_name", "")
                if not mname:
                    raise ValueError(f"В модуле '{mod_name}' module_metrics: field_name не может быть пустым")

    def _validate_span_exists(self, span_name: str) -> None:
        """
        Проверяет, что спан с таким span_name существует в трейсах.

        :param span_name: Название спана для проверки
        :raises ValueError: если спан не найден
        """
        found = any(span.get("span_name") == span_name for span in self.traces)
        if not found:
            raise ValueError(f"Спан с span_name '{span_name}' не найден в трейсах")
        
    @staticmethod
    def _is_agent_span(span: dict) -> bool:
        aef_kind = span.get("aef_kind", "")
        return aef_kind in ["input_request", "start_agent"]

    @staticmethod
    def _parse_input_message(span: dict) -> str:
        input_text = span.get("input_text", "")
        aef_kind = span.get("aef_kind", "")

        if not isinstance(input_text, str) or not input_text:
            return ""

        try:
            data = json.loads(input_text)
        except json.JSONDecodeError:
            return input_text

        if aef_kind == "llm":
            if isinstance(data, list) and data and isinstance(data[0], dict):
                return str(data[0].get("content", str(data[0])))
        elif aef_kind == "start_agent":
            if isinstance(data, dict):
                for key in ["goal", "task", "message", "query", "text", "user_question"]:
                    if key in data:
                        return str(data[key])
                return str(data)
        elif aef_kind == "input_request":
            if isinstance(data, dict):
                for key in ["message", "query", "text", "input"]:
                    if key in data:
                        return str(data[key])
                return str(data)
        elif aef_kind in ["tool", "retriever"]:
            if isinstance(data, dict):
                if "query" in data:
                    return str(data["query"])
                if "text" in data:
                    return str(data["text"])
                if "location" in data:
                    return str(data["location"])
                return str(data)

        if isinstance(data, dict):
            for key in ["text", "query", "input"]:
                if key in data:
                    return str(data[key])
            return str(data)

        return str(data)

    @staticmethod
    def _parse_output_message(span: dict) -> str:
        output_text = span.get("output_text", "")

        if not isinstance(output_text, str) or not output_text:
            return ""

        try:
            data = json.loads(output_text)
        except json.JSONDecodeError:
            return output_text

        aef_kind = span.get("aef_kind", "")

        if aef_kind == "start_agent":
            if isinstance(data, dict):
                for key in ["summary", "answer", "final_response", "error"]:
                    if key in data:
                        value = data[key]
                        if isinstance(value, list):
                            return "\n".join(str(v) for v in value)
                        return str(value)
                return str(data)
        elif aef_kind == "llm":
            if isinstance(data, str):
                return data
            return str(data)
        elif aef_kind == "input_request":
            if isinstance(data, dict):
                if "status" in data:
                    return str(data["status"])
                return str(data)
        elif aef_kind in ["tool", "retriever"]:
            if isinstance(data, dict):
                if "status" in data:
                    return str(data["status"])
                return str(data)
        elif aef_kind == "guard":
            if isinstance(data, dict) and "allowed" in data:
                return str(data["allowed"])
            return str(data)

        return str(data)

    def _group_traces_by_session_id(self) -> dict[str, dict]:
        """
        Группирует трейсы по session_id и собирает агентские и модульные данные.

        :return: dict[str, dict] где:
            - внешний ключ: session_id (строка)
            - внутренний ключ: query_id (строка, trace_id спана)
            - значение: dict с ключами "agent_request" и "module_data"
                - "agent_request": dict с "input_query" и "output_answer"
                - "module_data": dict {module_name: {"input_query": ..., "output_answer": ...}}
        """
        grouped: dict[str, dict] = {}

        for span in self.traces:
            session_id = span.get("session_id", "")
            if not session_id:
                continue

            if session_id not in grouped:
                grouped[session_id] = {}

            trace_id = span.get("trace_id", "")

            if trace_id not in grouped[session_id]:
                grouped[session_id][trace_id] = {
                    "agent_request": None,
                    "module_data": {},
                }

            # Агентские спаны: start_agent приоритетнее input_request,
            # уже выбранный start_agent повторно не перезаписывается
            if self._is_agent_span(span):
                current = grouped[session_id][trace_id]["agent_request"]
                incoming_kind = span.get("aef_kind", "")
                if current is None or (
                    incoming_kind == "start_agent"
                    and current.get("source_kind") != "start_agent"
                ):
                    grouped[session_id][trace_id]["agent_request"] = {
                        "input_query": self._parse_input_message(span),
                        "output_answer": self._parse_output_message(span),
                        "source_kind": incoming_kind,
                    }

            # Модульные спаны — ищем по module_name (совпадает с span_name из трейсов)
            span_name = span.get("span_name", "")
            if span_name in self.module_names:
                for mod in self.modules:
                    if mod.get("module_name") == span_name:
                        grouped[session_id][trace_id]["module_data"][span_name] = {
                            "input_query": self._parse_input_message(span),
                            "output_answer": self._parse_output_message(span),
                        }
                        break

        return grouped

    def create_qa_dataset(self) -> pd.DataFrame:
        """
        Создаёт QA датасет (один вопрос-ответ на строку).

        Порядок колонок (определяется main_fields + end2end_metrics + modules из конфига):
        1. session_id
        2. query_id
        3. input_query
        4. output_answer
        5. [end2end_metrics колонки из self.end2end_metrics]
        6. [для каждого модуля: колонки module_fields + module_metrics]

        :return: DataFrame с QA датасетом
        """
        grouped = self._group_traces_by_session_id()

        records = []

        for session_id, data in grouped.items():
            for query_id, item in data.items():
                agent_request = item["agent_request"]
                module_data = item["module_data"]

                if not agent_request or not str(agent_request.get("input_query", "")).strip():
                    continue

                record: dict[str, Any] = {}
                for bf in self.BASE_FIELDS:
                    if bf == "session_id":
                        record[bf] = session_id
                    elif bf == "query_id":
                        record[bf] = query_id
                    elif bf == "input_query":
                        record[bf] = agent_request["input_query"]
                    elif bf == "output_answer":
                        record[bf] = agent_request["output_answer"]

                # --- End2end метрики (после output_answer) ---
                for em in self.end2end_metrics:
                    em_name = em.get("field_name", "")
                    if em_name:
                        record[em_name] = ""

                # --- Дополнительные модули (из self.modules) ---
                for mod in self.modules:
                    mod_name = mod.get("module_name", "")
                    if not mod_name:
                        continue

                    # Если спана для этого модуля нет в трейсах — пропускаем
                    if mod_name not in module_data:
                        continue

                    # Добавляем колонки из module_fields (явно заданные в конфиге)
                    mod_span_data = module_data[mod_name]
                    for mf in mod.get("module_fields", []):
                        fname = mf.get("field_name", "")
                        if not fname:
                            continue
                        # Определяем: если field_name оканчивается на _input_query — берём input_query, иначе output_answer
                        if fname.endswith("_input_query"):
                            record[fname] = mod_span_data.get("input_query", "")
                        elif fname.endswith("_output_answer"):
                            record[fname] = mod_span_data.get("output_answer", "")
                        else:
                            # Если не удалось определить — ставим пустую строку
                            record[fname] = ""

                    # Добавляем колонки метрик из module_metrics
                    for mm in mod.get("module_metrics", []):
                        mname = mm.get("field_name", "")
                        if mname:
                            record[mname] = ""

                records.append(record)

        if not records:
            return pd.DataFrame(
                columns=self.BASE_FIELDS + [m.get("field_name", "") for m in self.end2end_metrics if m]
            )

        return pd.DataFrame(records)

    def create_dialog_dataset(self) -> pd.DataFrame:
        """
        Создаёт датасет для диалогов (с историей сообщений).

        Колонки:
        - session_id: uuid/int - идентификатор сессии
        - dialogue: list[(query_id, input_query, output_answer)] - полный текст диалога
        - [end2end_metrics колонки из self.end2end_metrics]

        :return: DataFrame с датасетом диалогов
        """
        grouped = self._group_traces_by_session_id()

        records = []

        for session_id, data in grouped.items():

            dialogue = [
                (query_id, item["agent_request"]["input_query"], item["agent_request"]["output_answer"])
                for query_id, item in data.items()
                if item["agent_request"]
                and str(item["agent_request"].get("input_query", "")).strip()
            ]

            record: dict[str, Any] = {
                "session_id": session_id,
                "dialogue": dialogue,
            }

            # End2end метрики
            for em in self.end2end_metrics:
                em_name = em.get("field_name", "")
                if em_name:
                    record[em_name] = ""

            records.append(record)

        if not records:
            return pd.DataFrame(columns=["session_id", "dialogue"] + [m.get("field_name", "") for m in self.end2end_metrics if m])

        return pd.DataFrame(records)
    
