# laim-traces-dataset-converter

Нода мониторингового контура LAIM. Принимает **сырые AEF-спаны** агента за
мониторинговый период, **контракт метрики** и **вердикт проверки трейсов** и
отдаёт в контур **monitoring-корзину в формате тестового датасета** — тот же
лист, что эталонная корзина `laim-baskets-adapter`, — плюс отчёт о том, какие
turn доказаны, а какие нет.

## Зачем нода нужна

Трейсы разных агентов физически не похожи: роутер общается FIPA-конвертами
(`conversation_id`, `reply_with`, `in_reply_to`), и один его turn может
пересекать несколько trace; non-FIPA агент отдаёт пару запрос/ответ на
HTTP-границе; langgraph-агент хранит весь диалог в `state_json`. Нода снимает
эту разницу: единица извлечения — **пользовательский turn, доказанный
протоколом**, а не trace и не «первый вход / последний выход».

Ключевые решения: **детерминизм без LLM и сети** — каждая строка подтверждена
correlation key FIPA либо явной внешней границей AEF, произвольные JSON-ключи
рекурсивно не ищутся, неизвестная схема не угадывается (fail-closed), а
попадает в отчёт как `unsupported_trace_schema`; **ответ и маршрут — разные
сигналы** — финальный ответ идёт в `output_answer`, наблюдаемая метка
маршрутизации — в `scenario`, и метка никогда не подменяет ответ;
**деградация вместо падения** — неполное окно выгрузки не уничтожает
корректные turn: доказанные публикуются со статусом `partial`, а падение —
только когда полных turn нет совсем или нарушен контракт входа.

## Место в контуре

```text
monitoring_traces (parquet витрины) ─┐
traces_validation_result ────────────┤
selection ───────────────────────────┤
laim-kriteria-selector.validated_monitoring_metric ─┘
        │
        ▼
 laim-traces-dataset-converter ──► monitoring_umr        ─► laim-asessor-agent,
        │                                                   laim-local-drift-test,
        │                                                   laim-oos-oot-test
        ├──────────────────────► processing_report     (в port_wiring не подключён)
        ├──────────────────────► parquet_test_dataset  (в port_wiring не подключён)
        ├──────────────────────► umr_artifact          (в port_wiring не подключён)
        └──────────────────────► settings              (в port_wiring не подключён;
                                                        для платформенного file-writer)
```

## Порты и настройки

### Входы

| Порт | Обязателен | Что приходит |
|---|---|---|
| `monitoring_traces` | да | Таблица спанов: `DataFrame`, байты, путь к parquet/xlsx или каталог с частями (тип по сигнатуре файла `PAR1`/`PK`, не по расширению) |
| `monitoring_metric` | да | `laim-monitoring-metric.v2` от `laim-kriteria-selector`; `status` — `computed` или `not_computable`. Для `computed` требуются `assessment_mode` из `qa \| dialogue \| turn_with_history`, `scoring.method` из `identity \| accuracy \| mean_criteria \| all_criteria \| majority \| all_assessors`, у каждого источника `source_id`, `role`, `column_name` и полный набор ролей метода |
| `traces_validation_result` | да (без `ignore_traces_checks`) | Для `computed`: полный verdict `laim-ars-env-validation` — `schema`, `quality`, `criteria` (K1–K8 с `tone`, `result`, `title`), `readiness`, `metrics`. При `not_computable` не читается |
| `selection` | нет | `agent_ci` (фильтр по `agent_id`; без него выгрузка должна содержать ровно одного агента), `distributive` (по умолчанию `monitoring`), `solution_version` |

Обязательные колонки спанов: `trace_id`, `span_id`, `aef_kind`, `span_name`,
`input_text`, `output_text`, `agent_id`, `session_id`, `start_time_ns`.
Стратегия `aef_parent_boundary_v1` дополнительно читает `parent_span_id` и
`end_time_ns`, если они есть.

### Выходы

| Порт | Тип | Что отдаёт |
|---|---|---|
| `monitoring_umr` | dataframe | Корзина в формате тестового датасета (`laim-umr.v2`): flat для `qa`/`turn_with_history`, packed dialogue для `dialogue` |
| `processing_report` | default | Отчёт `laim-monitoring-trace-converter.v2` (см. «Наблюдаемость») |
| `parquet_test_dataset` | default | Байты parquet с тем же фреймом, что `monitoring_umr` |
| `umr_artifact` | default | `pd.ExcelFile` с одним листом «Вариант для отд. запросов» или «Вариант для диалога» |
| `settings` | default | Параметры file-writer: `fileSystemPath = hdfs://arnsdpsbx/tmp/traces_based_datasets/<agent_ci>_<slug дистрибутива>-<sha256[:12]>`, `fileName = monitoring_umr_<дд_мм_ГГГГ_ЧЧ_ММ_СС>.xlsx`, `rewrite = true`, `addPostfix = false`, `structured = false` |

### Настройки ноды

| Настройка | По умолчанию | Зачем |
|---|---|---|
| `ignore_traces_checks` | `false` | Пропустить гейт `traces_validation_result` (в отчёте `traces_validation.status = bypassed`). Только для локального исследования; в контуре не включать |
| `min_extraction_coverage` | `0.9` | Покрытие извлечения turn ниже порога переводит `data_readiness.state` в `limited` (`low_extraction_coverage`) |

## Как проходит прогон

```text
1. Контракт КМ   not_computable → пустые UMR/parquet/XLSX, status=not_ready, без чтения трейсов
2. Контракты     computed + traces_validation_result + selection → ValueError при структурном нарушении; красный DQ (K1 критичные, quality.valid=false, K1–K6 bad) → status=not_ready, reason_code=dq_failed, трейсы не читаются
3. Чтение        read_table: DataFrame / байты / файл / каталог → спаны
4. Извлечение    фильтр по agent_id → кандидаты-границы → пять стратегий по порядку
5. Проекция      turn → строки UMR (flat или packed dialogue), prediction для accuracy
6. Сериализация  parquet + XLSX (управляющие символы, ведущий «=», лимит ячейки)
7. Отчёт         статус complete / partial / not_ready, conservation, issues, data_readiness (6.3.2)
```

**3. Извлечение.** Кандидатами считаются спаны с `aef_kind` из
`input_request | start_agent`, с `span_name` вида `POST /…` или `…/invoke`,
либо с FIPA-маркером (`conversation_id`/`reply_with`/`in_reply_to`) в
`input_text` или `output_text`. Стратегии применяются к trace, ещё не покрытым
предыдущими:

1. `fipa_acl_v1` — конверт в `input_text.message`/`input_text.incoming` (или
   сам `input_text`) и в `output_text.outgoing` (или сам `output_text`):
   словарь с `content`, `sender`/`receiver` и одним из
   `conversation_id`/`reply_with`/`in_reply_to`/`message_id`. Контрагент —
   отправитель, чей `request` открывает conversation раньше, чем агент шлёт
   ему свой, в строгом большинстве своих conversation; имя самого агента
   контрагентом быть не может. Вход — `request` контрагента с ключом
   `(conversation_id, reply_with)`, выход — ответ контрагенту (перформатив не
   `request`, есть `in_reply_to`) с ключом `(conversation_id, in_reply_to)`;
   join может пересекать trace. Текст — все текстовые части
   `content.message[*].value` через пустую строку; вход вида `[label, текст]`
   отдаёт метку в `scenario`, иначе маршрут — текстовая часть `outgoing`, не
   равная эху вопроса, либо `outgoing.receiver`, если он не контрагент. Равные
   реплики одного ключа дедуплицируются, конфликтующие не публикуются.
2. `aef_boundary_v1` / `aef_start_agent_v1` — non-FIPA спан на внешней границе
   (`input_request` или HTTP `span_name`); `start_agent` — только если в trace
   нет внешней границы. Запрос: строка `input_text` либо ключи
   `message | query | question | user_query | input_query | text`, последнее
   `human`/`user` в `messages`, затем `body`/`request`. Ответ: строка
   `output_text` либо ключи `answer | agent_response | output_answer |
   final_response | message_to_user`, последнее `assistant`/`ai`/`agent` в
   `messages`, затем `body` и `response.body`. Подтверждения `ok | success |
   done | accepted | processing | true | false` ответом не считаются; для
   `start_agent` запасные ключи `goal`/`task`/`text` и `summary`/`error`.
3. `aef_parent_boundary_v1` — `start_agent`, чей `parent_span_id` — внешняя
   граница с FIPA-конвертами; запрос совпадает с текстом границы, `session_id`
   общий; в trace допускается ровно один вариант пары.
4. `aef_semantic_terminal_v1` — все `start_agent` trace несут один запрос, а
   ровно одно значение ответа найдено в верхнеуровневых ключах
   `final_response | message_to_user | assistant_message | generated_answer |
   agent_answer | agent_response | full_answer` спанов той же сессии; префикс
   запроса, равный уже известной метке маршрута, уходит в `scenario`.
5. `state_json_v1` — состояние со `stage` `exit`/`__end__` в `output_text`
   (затем `input_text`): ответ `message_to_user`, вопрос — последняя
   `user`/`human`-реплика `messages`, маршрут `product_agent`; берётся
   последний такой спан trace.

Стратегии 3–5 закрывают незамкнутые FIPA-ключи и `boundary_pair_incomplete`
тех же trace с тем же текстом, чтобы один turn не считался дважды.

### Пример лога прогона (агент CI09997554, 5 028 спанов, 104 trace)

```text
INFO main: LAIM traces dataset converter: rows=5028, agent_id=CI09997554, mode=dialogue
INFO trace_dialogue: Dialogue extraction complete: turns=104, candidates=104, FIPA=97, AEF=0
INFO main: LAIM traces dataset converter complete: status=complete, rows=94, seconds=0.387
```

Отчёт того же прогона: `strategies = {fipa_acl_v1: 97,
aef_parent_boundary_v1: 5, aef_semantic_terminal_v1: 2}`,
`extraction_coverage = 1.0`, 104 turn упакованы в 94 строки-сессии,
`counterparts = [d-credit-helper]`, `traces_validation.status =
passed_with_warnings` (K1, K2 — `warn`; K7, K8 — `bad`, не блокируют).
Неполное извлечение (из `tests/test_main.py`) выглядит так:

```text
INFO trace_dialogue: Dialogue extraction complete: turns=1, candidates=2, FIPA=1, AEF=0
WARNING main: Неполное извлечение turn: публикуются только доказанные complete turn (1 из 2)
INFO main: LAIM traces dataset converter complete: status=partial, rows=1, seconds=0.125
```

## Форматы выхода и контракты

`monitoring_umr`, `parquet_test_dataset` и `umr_artifact` несут один и тот же
фрейм; вариант выбирает `monitoring_metric.assessment_mode`.

- **Flat** (`qa`, `turn_with_history`; лист «Вариант для отд. запросов»):
  строка = turn. Колонки по порядку: `scenario`, `session_id`, `query_id`,
  `input_query_count` (всегда 1), `input_query`, `output_answer`; для метода
  `accuracy` — prediction-колонка с именем из `scoring.sources`, заполненная
  из `scenario`, если метка есть во всех строках. Порядок строк —
  `session_id`, время входа, `query_id`.
- **Packed dialogue** (`dialogue`; лист «Вариант для диалога»): строка =
  `session_id`. Колонки: `scenario`, `session_id`, `dialogue` — Python-литерал
  списка троек `(query_id, input_query, output_answer)` в наблюдённом порядке;
  prediction-колонка публикуется только при постоянном значении внутри
  сессии; `scenario` опускается, если меняется внутри хотя бы одной сессии
  (в прогоне CI09997554 опубликованы только `session_id` и `dialogue`);
  `input_query_count` не публикуется.
- `selection.solution_version`, если задан, — первая колонка обоих вариантов.
- При `monitoring_metric.status = not_computable` все три порта содержат пустой
  фрейм соответствующего варианта, а `processing_report` — `status = not_ready`,
  `ready_for_scoring = false` и исходные `reason_code`/`reason`. Трейсы и их
  verdict не читаются. Если `assessment_mode` в отказе отсутствует, используется
  пустой flat-вариант `qa`.

`query_id` стабилен: FIPA — `conversation_id|reply_with`; AEF —
`aef:<trace_id>:<span_id>`, `parent:…`, `terminal:…`, `state:…`. Target (`GT`)
из трейсов не синтезируется и остаётся в эталонной корзине; имена
prediction/target-колонок не могут совпадать с полями UMR (`session_id`,
`query_id`, `input_query`, `output_answer`, `scenario`, `input_query_count`,
`reference_group_id`, `turn_index`), prediction — только со `scenario`.
Диагностика (trace/span ID, source paths, латентность) в UMR не попадает.

## Падение против деградации

Нода завершает вызов исключением (сообщение называет поле и значение):

| Причина | Исключение |
|---|---|
| `monitoring_metric` не `laim-monitoring-metric.v2`, `status` не `computed`/`not_computable`; для `computed` — неизвестный метод, повтор `source_id`/`column_name`, неполные роли источников | `ValueError` |
| `traces_validation_result` отсутствует без `ignore_traces_checks`, неполный verdict, нецелое `schema["критичных нарушено"]`, несогласованные `rule_violations`, некорректные `criteria`/`readiness`/`metrics` | `ValueError` |
| `min_extraction_coverage` не число от 0 до 1 | `ValueError` |
| `selection` не объект; `ignore_traces_checks` не bool | `ValueError` |
| `monitoring_traces` не таблица; файл не parquet и не xlsx; в каталоге нет частей | `TypeError`, `ValueError`, `FileNotFoundError` |
| Пустая выгрузка, нет обязательных колонок, нет спанов `agent_ci`, несколько агентов без `agent_ci` | `TraceExtractionError` |
| Turn без `session_id`/текста после извлечения, повтор `(session_id, query_id)`, prediction/target-колонка конфликтует с полем UMR, `complete_turns` не равен числу строк | `MonitoringCanonicalizationError` |

Всё остальное — деградация с записью в `processing_report`:

| Событие | Реакция |
|---|---|
| `monitoring_metric.status = not_computable` | Трейсы не читаются; публикуются пустые валидные UMR/parquet/XLSX, `status = not_ready`, `ready_for_scoring = false`, исходные `reason_code`/`reason` |
| Красный DQ: `schema["критичных нарушено"] > 0`, `quality[0].valid != true`, любой из K1–K6 с `tone = bad` | Трейсы не читаются; пустые UMR/parquet/XLSX, `status = not_ready`, `reason_code = dq_failed`, `traces_validation.status = failed` с `failed_criteria`, `data_readiness.state = failed` |
| Ни одного полного turn по поддерживаемым схемам | пустые UMR/parquet/XLSX, `status = not_ready`, `reason_code = no_turns_extracted`, счётчики извлечения в `extraction`, `data_readiness.state = insufficient` |
| Незамкнутые FIPA-ключи, конфликтующие реплики, `unsupported_trace_schema` | `status = partial`, WARNING, `conservation.unpublished_turns`, коды в `issues` |
| Для `accuracy` метка маршрута есть не во всех turn (или меняется внутри сессии в `dialogue`) | prediction-колонка не публикуется, `ready_for_scoring = false`, `status = not_ready`, warning `UMR не готова к прямому scoring`; при одновременном `partial` статус — `partial` |
| K1–K6 с `tone = warn` | `traces_validation.status = passed_with_warnings`, `warning_criteria` |
| K7 (`readiness`) и K8 (структура телеметрии) с `tone = bad` | сохраняются в `traces_validation.non_gating_criteria`, гейт не срабатывает |
| Ячейка длиннее 32 767 символов | в XLSX обрезана с пометкой `…[обрезано в XLSX: N символов]`, полный текст в dataframe/parquet, `{column, row, length}` в `serialization.excel_truncated_cells`, warning |
| Управляющие символы, значение с ведущим `=` | в XLSX заменены пробелом / записаны как текст, не формула |
| `session_id` на одной стороне FIPA-пары; trace без граничного спана (`chain`/`llm`) | issue `session_id_partial` / `no_boundary_span` с `severity = warning`, статус не меняется |
| `failure` без текста | считается в `failures_without_text`, не публикуется |

## Внешние сервисы

Не применимо: нода не обращается к LLM, эмбеддингам и сети. Путь HDFS в порте
`settings` только формируется (query-параметры и токены URL дистрибутива в
него не попадают); запись выполняет платформенный file-writer.

## Наблюдаемость

В лог платформы уходят три строки (вход, итог извлечения, итог прогона) и
WARNING на каждую деградацию. Полная картина — в `processing_report`:

```json
{
  "contract_version": "laim-monitoring-trace-converter.v2",
  "status": "complete | partial | not_ready",
  "agent_id": "CI09997554", "assessment_mode": "dialogue",
  "ready_for_scoring": true, "warnings": [],
  "traces_validation": {"scope": "K1-K6", "status": "passed_with_warnings", "warning_criteria": ["K1", "K2"],
                        "non_gating_criteria": {"K7": "...", "K8": "..."}, "readiness": ["..."]},
  "semantics": {"input_query": "user transport text", "...": "..."},
  "stage_timings_seconds": {"read_input": 0.0, "extract_turns": 0.264, "canonicalization": 0.017,
                            "serialization": 0.106, "total": 0.387},
  "conservation": {"candidate_turn_keys": 104, "published_turns": 104, "published_rows": 94,
                   "unpublished_turns": 0, "extraction_coverage": 1.0},
  "extraction": {"contract_version": "laim-trace-turn-extraction.v2", "input_rows": 5028,
                 "input_trace_count": 104, "counterparts": ["d-credit-helper"],
                 "strategies": {"fipa_acl_v1": 97, "aef_parent_boundary_v1": 5, "aef_semantic_terminal_v1": 2},
                 "entry_without_exit": 0, "exit_without_entry": 0, "unsupported_trace_count": 0, "...": "..."},
  "issues": {"counts": {}, "examples": [], "examples_truncated": false},
  "semantic_profile": {"exact_echo_pairs": 0, "repeated_answer_rows": 0, "...": "..."},
  "canonicalization": {"contract_version": "laim-monitoring-turn-projection.v2",
                       "prediction_mapping": null, "missing_scoring_sources": [], "...": "..."},
  "serialization": {"excel_truncated_cells": []}
}
```

Триаж сотни прогонов — по этим JSON без чтения логов: `status`,
`data_readiness` — готовность данных периода по карточке 6.3.2: `state`
(`sufficient` / `limited` / `insufficient` / `failed`), `reason_code`, `reason`,
`limits` (`partial_extraction`, `low_extraction_coverage`,
`not_ready_for_scoring`, `dq_warnings`, `bypassed_dq`), `unit`
(`turn` / `session`), `published_units`, `extraction_coverage`, `dq_status`,
`min_extraction_coverage`. Читает агрегатор как базовый тест 6.3.2.
`conservation.extraction_coverage`, `issues.counts`, `extraction.strategies`,
`extraction.counterparts`, `canonicalization.missing_scoring_sources`,
`serialization.excel_truncated_cells`. Примеры в `issues.examples` (не больше
100) содержат только ID и код — без текстов и payload.

## Карта кода

```text
descriptor.json      порты, единственная настройка, sourceFiles, py312-simple
main.py              контракт ноды: гейты входов, оркестрация, XLSX, отчёт, settings
table_io.py          чтение dataframe-порта: DataFrame / байты / parquet / xlsx / каталог
trace_dialogue.py    ядро извлечения: пять versioned-стратегий, issues, extraction report
canonical.py         проекция turn в UMR (flat / packed dialogue), готовность к scoring
tests/               test_main (контракт ноды), test_trace_dialogue (стратегии), test_canonical (UMR)
docs/                целевая архитектура, требования УМР к трейсингу, спека формата
```

## Что делать, если

- **`TraceExtractionError: Не найдено полных turn`** — в сообщении есть
  `candidate_turn_keys`, `entry_without_exit`, `exit_without_entry`,
  `unsupported_trace_count`. Нули везде — в выгрузке нет граничных спанов
  (только `chain`/`llm`): проверить SQL витрины и `aef_kind`. Большие
  `entry_without_exit`/`exit_without_entry` — окно режет conversation.
- **`status = partial`** — смотреть `issues.counts`: `entry_without_exit` —
  обрезанное окно; `conflicting_*_replicas` — дубли спанов с разным текстом;
  `unsupported_trace_schema` — нужна новая versioned-стратегия с fixtures.
- **`status = not_ready`** — метод `accuracy`, а метки маршрута нет
  (`missing_scoring_sources`): агент не сообщает класс в трейсах; корзина
  опубликована, но автоассесору нечего сравнивать с `GT`. Если в отчёте есть
  `reason_code`, это исходный `not_computable` контракта метрики и трейсы не
  читались.
- **`ValueError` на `traces_validation_result`** — прочитать вердикт
  `laim-ars-env-validation`: гейт срабатывает на K1–K6, а не на K7/K8.
- **`Без agent_id выгрузка должна содержать ровно одного агента`** — в витрине
  несколько `agent_id`; передать `selection.agent_ci`.

## Деплой

База — `py312-simple`; синтаксис и stdlib новее Python 3.12 не используются.
`descriptor.json` перечисляет в `script.runConfiguration.sourceFiles` четыре
файла — `main.py`, `table_io.py`, `trace_dialogue.py`, `canonical.py`; точка
входа — функция `main` в `main.py`. Теста соответствия `sourceFiles` диску в
ноде нет — список проверяется при сборке вручную. Зависимости
`requirements.txt`: `pandas`, `pyarrow`, `openpyxl`. Нода самодостаточна:
импортов извне каталога нет.

Проверки перед сборкой (CI `.github/workflows/ci.yml`, Python 3.12,
`ruff==0.15.5`): `ruff check .` и `python -m pytest -q` (60 passed).
Production ZIP собирается из головы ветки `dev` при зелёном CI: `descriptor.json`,
`requirements.txt`, `README.md` и четыре файла из `sourceFiles`; `tests/`,
`docs/` и кеши в сборку не попадают. Готовая версия переносится в снимок
## Глоссарий

- **Turn** — пара «запрос пользователя → финальный ответ агента», доказанная
  протоколом; единица извлечения. Один turn может занимать несколько trace,
  один trace — содержать несколько агентов.
- **FIPA ACL** — протокол конвертов с `conversation_id`, `reply_with`,
  `in_reply_to`; ключ turn — `(conversation_id, reply_with/in_reply_to)`.
- **Контрагент** — отправитель, чьи turn оцениваются (человек или вышестоящий
  роутер); выводится из данных, не из имён endpoint.
- **Внешняя граница (boundary)** — спан `input_request` или HTTP-вызов, на
  котором виден запрос пользователя и ответ ему.
- **Метка маршрута (`scenario`)** — наблюдаемое решение роутера/класс запроса;
  для метода `accuracy` служит prediction.
- **Conservation** — сверка: сколько ключей turn найдено, сколько
  опубликовано, сколько строк получилось.
