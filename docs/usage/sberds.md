# Запуск конвертера трейсов в SberDS

## Этап 0. Создание проекта

Проект создается на основе готового шаблона, который находится в пространстве **autovalidation**:

1. В SberDS перейти на вкладку **"Templates"**.
2. Найти шаблон с названием **`traces-converter`**.
3. Нажать **"Create project"**.
4. Указать название и описание проекта.
5. После создания в проекте будут доступны все необходимые ноды (Jupyter notebook, конвертер трейсов, сохранение датасета) с предустановленными связями между портами.

## Общая схема пайплайна

```
[Jupyter notebook] --(agent_id, distributive, is_dialog, config)--> [Конвертер трейсов] --(result: ExcelFile, settings)--> [Сохранение датасета]
         |
         |--- (sql_dict: SQL + LIMIT) --> [Data Source] --(parquet: pd.DataFrame) --> [Конвертер трейсов]
```

Конвертер трейсов принимает на вход:
- **parquet** (pd.DataFrame) — DataFrame со спанами трейсинга, полученный из трейсов AEF SDK
- **config** (dict) — конфигурация колонок датасета, описывающая структуру выходных полей
- **agent_id** (str) — идентификатор агента в формате КЭ
- **distributive** (str) — хэш версии решения
- **is_dialog** (bool) — бинарный флаг: True = диалоговый датасет, False = QA датасет
- **traces_validation_result** (dict) — словарь с результатами проверок качества трейсов (ключи `schema`, `quality`)

Выход конвертера:
- **result** (`pd.ExcelFile`) — датасет в формате `.xlsx`
- **parquet_test_dataset** (`pd.DataFrame`) — тот же датасет в dataframe-порту для downstream-потребителей
- **settings** (dict) — настройки для сохранения файла

## Этап 1. Настройка Jupyter notebook (первая нода)

Jupyter-нода подготавливает **параметры пайплайна**: идентификаторы (`AGENT_ID`, `DISTRIBUTIVE`), формат (`IS_DIALOG`), конфигурацию колонок (`CONFIG`) и SQL-запрос с `LIMIT` для выборки спанов из таблицы `t_team_cdiotraces.aef_trace_expanded`. Все параметры передаются в downstream-узлы через `write_to_port`.

### Ячейка 1. Идентификаторы и формат

```python
AGENT_ID = "CI09583909"
DISTRIBUTIVE = "hash128736123"
IS_DIALOG: bool = False
```

- **AGENT_ID** — КЭ (конфигурационный элемент) агента. Используется в SQL-фильтре `WHERE agent_id = '{AGENT_ID}'` и в пути сохранения датасета.
- **DISTRIBUTIVE** — хэш дистрибутива/версии решения. Используется в пути сохранения датасета.
- **IS_DIALOG** — бинарный флаг:
  - `False` — на выходе одна строка = один вопрос-ответ (QA датасет);
  - `True` — на выходе одна строка = одна сессия с историей сообщений (диалоговый датасет).

### Ячейка 2. Параметры выборки спанов

```python
N_MIN = 500
SPANS_PER_TRACE_MEAN = 5
DIALOG_COEFF = 4 if IS_DIALOG else 1
MAXIMIZING_COEFF = 10
```

- **N_MIN** — минимальное количество **трейсов**, которое должно быть отобрано. Это же значение дублируется в параметр `min_eff_traces` ноды `laim-ars-env-validation`.
- **SPANS_PER_TRACE_MEAN** — среднее количество спанов на трейс для агента.
- **DIALOG_COEFF** — диалоговый коэффициент: из скольких пар "запрос-ответ" в среднем состоит диалог (только если `IS_DIALOG = True`).
- **MAXIMIZING_COEFF** — коэффициент максимизации количества спанов; страховка на случай малого числа эффективных (прошедших отбраковку) трейсов.

### Ячейка 3. Расчёт LIMIT

```python
LIMIT = N_MIN * SPANS_PER_TRACE_MEAN * DIALOG_COEFF * MAXIMIZING_COEFF
print(f"Максимальное количество спанов в выдаче равно {LIMIT}")
```

При значениях по умолчанию для QA-кейса: `LIMIT = 500 * 5 * 1 * 10 = 25000`.

### Ячейка 4. Конфигурация колонок датасета (CONFIG)

```python
CONFIG = {
  "main_fields": [
    {"field_name": "session_id", "data_type": "string", "description": "Идентификатор сессии для группировки диалогов"},
    {"field_name": "query_id", "data_type": "string", "description": "Идентификатор запроса (trace_id из спана)"},
    {"field_name": "input_query", "data_type": "string", "description": "Текст входного запроса (из агентского спана)"},
    {"field_name": "output_answer", "data_type": "string", "description": "Текст выходного ответа (из агентского спана)"}
  ],
  "end2end_metrics": [
    {"field_name": "quality_metric", "data_type": "string", "description": "Метрика качества end-to-end"}
  ],
  "modules": [
    {
      "module_name": "content_filter",
      "module_fields": [
        {"field_name": "content_filter_input_query", "data_type": "string", "description": "Текст входного запроса для модуля content_filter"},
        {"field_name": "content_filter_output_answer", "data_type": "string", "description": "Текст выходного ответа для модуля content_filter"}
      ],
      "module_metrics": [
        {"field_name": "content_filter_recall_metric", "data_type": "float", "description": "Метрика recall для модуля content_filter"}
      ]
    }
  ]
}
```

Обязательные поля `main_fields`: `session_id`, `query_id`, `input_query`, `output_answer`. Если в `modules` указан модуль, `module_name` которого отсутствует в спанах, конвертер выбрасывает ошибку валидации.

### Ячейка 5. Формирование SQL-запроса

```python
sql = f"SELECT * FROM t_team_cdiotraces.aef_trace_expanded WHERE agent_id = '{AGENT_ID}' AND input_text != '' AND output_text != '' LIMIT {LIMIT}"
SQL_DICT = {"sql": sql}
```

SQL-фильтр исключает спаны с пустыми `input_text` / `output_text` (такие спаны неинформативны для QA и диалоговой корзины).

### Ячейка 6. Передача параметров в downstream-узлы

```python
write_to_port("sql_dict", SQL_DICT)
write_to_port("agent_id", AGENT_ID)
write_to_port("distributive", DISTRIBUTIVE)
write_to_port("is_dialog", IS_DIALOG)
write_to_port("config", CONFIG)
```

Здесь:
- `sql_dict` идёт в Data Source, которая возвращает `parquet` для конвертера.
- `agent_id`, `distributive`, `is_dialog`, `config` идут напрямую в конвертер трейсов.

### Действия после настройки

1. Запустить все ячейки в ноутбуке.
2. Нажать **"Save"** в интерфейсе Jupyter-ноды.
3. Нажать **"Execute"**, чтобы SberDS зафиксировал значения выходных портов.

## Этап 2. Запуск пайплайна

После выполнения Jupyter notebook нажать **"Execute all nodes"**. SberDS последовательно запустит все оставшиеся ноды:

1. **Data Source** — получает `sql_dict` из Jupyter-ноды, выполняет SQL-запрос к `t_team_cdiotraces.aef_trace_expanded` и возвращает результат в виде pd.DataFrame на вход конвертера.
2. **Конвертер трейсов** — принимает `parquet` (pd.DataFrame), `config` (dict), `agent_id`, `distributive`, `is_dialog`, `traces_validation_result`. Проверяет `traces_validation_result["quality"][0]["valid"]` — если `False`, пайплайн падает с `ValueError`. При успешной проверке строит датасет через `DatasetCreator` и упаковывает его в `pd.ExcelFile`.
3. **Сохранение датасета** — получает `pd.ExcelFile` из выходного порта `result` и сохраняет его как `.xlsx` в HDFS по пути из `settings`. Порт `parquet_test_dataset` (тот же датасет в формате `.parquet`) может быть подключен к дополнительным downstream-узлам, которым нужен parquet-формат.

## Этап 3. Путь сохранения

Датасет сохраняется по пути:

```
hdfs://arnsdpsbx/tmp/traces_based_datasets/{agent_id}_{distributive}/
```

Имя файла:

```
candidate_dataset_{dd_mm_YYYY_HH_MM_SS}.xlsx
```

Параметры settings:
- `fileSystemPath`: `hdfs://arnsdpsbx/tmp/traces_based_datasets/{agent_id}_{distributive}`
- `fileName`: `candidate_dataset_{dd_mm_YYYY_HH_MM_SS}.xlsx`
- `rewrite`: `true` — перезаписывает существующий файл
- `addPostfix`: `false` — без постфикса (если `true`, то добавит лишние дату и время сохранения файла)
- `structured`: `false`

Внутри `.xlsx` находится один лист с именем:
- `qa_dataset` — если `IS_DIALOG = False`
- `dialog_dataset` — если `IS_DIALOG = True`

### Альтернативный parquet-порт

Помимо `result` (`.xlsx`) конвертер возвращает **`parquet_test_dataset`** как
`pd.DataFrame`. Тип значения совпадает с опубликованным типом порта
`dataframe`; содержимое Excel и dataframe идентично по колонкам и значениям.

## Этап 4. Проверка результата

1. Открыть папку по указанному пути.
2. Убедиться, что файл `candidate_dataset_{dd_mm_YYYY_HH_MM_SS}.xlsx` существует и не пуст.
3. Скачать его себе на устройство из hdfs;
4. Считать через `pd.read_excel()` в SberDS Lab или Datalab AI:

```python
import pandas as pd

xlsx_path = "<путь_до_текущей_рабочей_директории>/candidate_dataset_{dd_mm_YYYY_HH_MM_SS}.xlsx"

# При необходимости можно явно указать лист:
dataset = pd.read_excel(xlsx_path, sheet_name="qa_dataset")  # или "dialog_dataset"

print(dataset.shape)
print(dataset.columns)
```

5. Проверить, что колонки и данные соответствуют заданной конфигурации.


## Этап 5. Контроль количества датасетов-кандидатов

В директории `hdfs://arnsdpsbx/tmp/traces_based_datasets/{agent_id}_{distributive}/` необходимо следить за количеством файлов датасетов-кандидатов. **Базовое ограничение — не более 25 кандидатов на одного агента.**

При превышении лимита рекомендуется удалить устаревшие датасеты вручную.

## Типичные ошибки

| Ошибка | Причина | Решение |
|--------|---------|---------|
| `KeyError: ...` | Параметр не передан в kwargs конвертера | Проверить, что все обязательные параметры заданы (parquet, config, agent_id, distributive, is_dialog, traces_validation_result). Обычно причина — не выполнены `write_to_port(...)` в Jupyter-ноде или не выполнена сама Jupyter-нода |
| `ValueError: Трейсы некорректны и не прошли проверку качества` | `traces_validation_result["quality"][0]["valid"]` == `False` | Проверить результат ноды валидации трейсов и устранить найденные проблемы (пустые `input_text`/`output_text`, неизвестные модули и т.п.) |
| `Неверная схема входных трейсов` | В TDC подключён производный датасет или legacy-порт вместо raw spans | Подключить mart с точными колонками `trace_id`, `span_id`, `span_name`, `aef_kind`, `input_text`, `output_text` |
| `TDC не сформировал ни одной единицы оценки` | Нет полных корневых `input_request` / `start_agent`; в ошибке приведены безопасные счётчики | Проверить `aef_kind_counts`, полноту трейсов и upstream SQL; не подменять корень внутренними `chain`/`llm`/`output_request` |
| `ValueError: Спан с span_name '...' не найден в трейсах` | Модуль из `modules` отсутствует в спанах | Убрать модуль из `CONFIG` или проверить трейсы |
