# LAIM traces dataset converter

Production-нода преобразует сырые AEF spans в monitoring UMR. Единицей
извлечения является пользовательский turn, а не trace. Нода не использует LLM,
сеть и рекурсивный поиск по произвольным JSON-ключам.

## Поддерживаемые схемы

### FIPA ACL v1

Вход и финальный выход соединяются между trace по протокольному ключу:

```text
(conversation_id, incoming.reply_with)
                 =
(conversation_id, final outgoing.in_reply_to)
```

- контрагент агента (тот, чьи turn оцениваются) определяется по данным, без
  имён endpoint: это отправители, чьи `request` в большинстве случаев открывают
  `conversation_id`. Для роутера CI09997438 это `agent_human`/`elchhumanagent`,
  для нижестоящего CI09997554 — роутер `d-credit-helper`; нижестоящий агент,
  присылающий уточняющий `request` внутри уже открытого разговора, контрагентом
  не становится. Список и число открытых conversation — в
  `processing_report.extraction.counterparts`;
- вход turn — `incoming` с перформативом `request` от контрагента; выход —
  `outgoing` с `in_reply_to` и перформативом, отличным от `request`, адресованный
  контрагенту (на том же span или в другом trace);
- `input_query` — текстовые части `input_text.message.content.message[*].value`;
  если вход имеет вид `[label, текст]` (так роутер передаёт вопрос нижестоящему
  агенту), метка отделяется в `route_label`, а вопрос — остальной текст;
- `agent_response` — все текстовые части финального
  `output_text.outgoing.content.message[*].value`; виджеты и другие нетекстовые
  части пропускаются, текстовые соединяются через пустую строку; `failure` без
  текста считается отдельно (`failures_without_text`);
- `route_label` никогда не заменяет `agent_response`: если
  `output_text.outgoing.content.message` роутера имеет вид `[label, echo вопроса]`,
  меткой является текстовая часть, не равная вопросу (для CI09997438 это
  `liabilities`, `available_credits`, …); иначе — `output_text.outgoing.receiver`,
  если он не контрагент. Источник фиксируется в `route_source_path`;
- одинаковые реплики correlation key дедуплицируются, конфликтующие значения не
  публикуются и получают отдельный issue code.

### AEF boundary v1

Для non-FIPA агента публикуется только явная пара на внешнем
`input_request`/HTTP boundary: запрос из request body и финальный ответ из
response body или ключа `answer`. `start_agent` используется как аналог только
когда в trace нет внешней boundary. Для известных legacy-схем допускаются два
ограниченных варианта: непосредственный parent-boundary с тем же запросом и
`session_id`, либо ровно один явный terminal-ключ ответа при единственном
варианте запроса в той же `(trace_id, session_id)`. Произвольный текст из
внутренних `chain`, LLM и tool spans ответом пользователя не становится.

Новая trace-схема требует отдельной versioned extraction strategy и тестовых
fixtures. Unknown schema обрабатывается fail-closed.

## Выходы и семантика

`monitoring_umr` и `parquet_test_dataset` содержат только поля UMR по
«Формату тестового датасета» (laim-umr.v2). Для `qa` и
`turn_with_history` это плоские строки:

- `session_id` — сессия пользователя;
- `query_id` — стабильный turn ID, для FIPA это
  `conversation_id|user_request_id`;
- `input_query` — транспортный текст пользователя без FIPA-envelope;
- `output_answer` — финальный ответ пользователю;
- `scenario` — наблюдаемая метка маршрутизации/класса (пусто, если агент её
  не сообщает);
- `input_query_count` = 1, `reference_group_id` и `turn_index` — по
  `assessment_mode`;
- для accuracy — объявленная в `monitoring_metric` prediction-колонка,
  заполненная из `scenario`.

Диагностика (trace/span ID, source paths, cross-trace, латентность, схема)
в UMR не публикуется: она агрегируется в `processing_report`, а примеры
проблем — в `processing_report.issues` без текстов и payload.

Для `assessment_mode=dialogue` все три выхода — DataFrame, parquet и Excel —
упаковываются одинаково: одна строка на `session_id`, колонка `dialogue`
содержит упорядоченные тройки `(query_id, input_query, output_answer)`.
`input_query_count` в диалоговом варианте не публикуется. Опциональный
`selection.solution_version` становится первым полем обоих вариантов.

Для accuracy наблюдаемый `route_label` может заполнить объявленную prediction
колонку. Target (`GT`) из трейсов не синтезируется: он остаётся в эталонной
корзине, где `laim-baskets-adapter` материализует `main_metric`. Если prediction
извлечено, monitoring UMR готова для автоассесора без `GT`; финальный ответ
сохраняется отдельно от route-label.

## Диагностика

`processing_report` содержит:

- число protocol turn keys и полных опубликованных turn;
- контрагентов и число открытых ими conversation;
- `entry_without_exit` и `exit_without_entry`;
- trace без граничного span (`no_boundary_span`) отдельно от trace с
  неподдерживаемой схемой (`unsupported_trace_schema`);
- эквивалентные и конфликтующие дубликаты;
- cross-trace и полные downstream correlation chains;
- unsupported trace schemas и неполные AEF boundaries;
- агрегированные issue codes и ограниченные примеры без текстов/payload;
- отдельный результат canonicalization и отсутствующие scoring sources.

При неполном покрытии публикуются только доказанные complete turn, а
`processing_report.status` становится `partial` с conservation и issue codes.
Пустая выгрузка без полного turn по-прежнему завершает вызов ошибкой.

Порт `traces_validation_result` обязателен и принимает полный verdict
`laim-ars-env-validation`: критическая схема, итог quality и критерии K1–K6
являются gate; readiness K7 и структурная диагностика K8 сохраняются в отчёте,
но не подменяют проверку качества строк. Обход допускается только явной
настройкой `ignore_traces_checks=true` для локального исследования. Это
единственная настройка ноды: контур
автоматический, `agent_ci` и `distributive` приходят портом `selection`,
лимит примеров в отчёте (100) и список методов КМ зафиксированы в коде и
совпадают с MeasurementPlan `laim-baskets-adapter` (`identity`, `accuracy`,
`mean_criteria`, `all_criteria`, `majority`, `all_assessors`).

Excel-выход очищается от управляющих символов и не исполняет строки, начинающиеся
с `=`, как формулы. Ячейка длиннее лимита Excel 32 767 символов обрезается в XLSX с пометкой «обрезано в XLSX», полный текст остаётся в dataframe-порте; список таких ячеек — в `processing_report.serialization.excel_truncated_cells`, плюс warning

## Runtime

Deploy-состав задаётся `descriptor.json` и включает только:

- `main.py` — контракт ноды и orchestration;
- `table_io.py` — транспорт DataFrame/parquet/xlsx;
- `trace_dialogue.py` — versioned extraction core;
- `canonical.py` — UMR projection и readiness gate.

Production ZIP содержит только `descriptor.json`, `requirements.txt`, README и
четыре файла из `sourceFiles`; тестовые данные, legacy-код и кеши в сборку не
попадают.

## Проверки

```bash
python -m pytest -q monitoring/laim-traces-dataset-converter/tests
ruff check monitoring/laim-traces-dataset-converter/main.py \
  monitoring/laim-traces-dataset-converter/table_io.py \
  monitoring/laim-traces-dataset-converter/trace_dialogue.py \
  monitoring/laim-traces-dataset-converter/canonical.py \
  monitoring/laim-traces-dataset-converter/tests
```

Acceptance baseline CI09997438 для локального parquet 6–11 августа:

- 2 151 полных FIPA-turn;
- 782 `entry_without_exit`, 700 `exit_without_entry`;
- 2 144 полных downstream chains;
- 6 cross-trace turn;
- 0 пустых запросов/ответов и 0 ответов вида `route + query`;
- `route_label` — 12 классов (`liabilities` 722, `available_credits` 422,
  `decline` 308, …), 11 из них есть в GT эталонной корзины; 45 ответов
  содержат более одной текстовой части.
