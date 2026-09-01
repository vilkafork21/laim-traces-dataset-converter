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

- `input_query` берётся только из текстовых частей
  `input_text.message.content.message[*].value` или эквивалентного плоского
  FIPA-envelope;
- `agent_response` — все текстовые части финального
  `output_text.outgoing.content.message[*].value`, адресованного user endpoint;
  виджеты и другие нетекстовые части пропускаются, текстовые соединяются через
  пустую строку;
- `route_label` берётся отдельно из входного события и никогда не заменяет
  `agent_response`: если `output_text.outgoing.content.message` роутера имеет вид
  `[label, echo вопроса]`, меткой является текстовая часть, не равная вопросу
  (для CI09997438 это `liabilities`, `available_credits`, …); иначе —
  `output_text.outgoing.receiver`, если он не user endpoint. Источник фиксируется
  в `route_source_path`;
- одинаковые реплики correlation key дедуплицируются, конфликтующие значения не
  публикуются и получают отдельный issue code.

### AEF boundary v1

Для non-FIPA агента публикуется только явная пара на внешнем
`input_request`/HTTP boundary: запрос из request body и финальный ответ из
response body или ключа `answer`. `start_agent` используется как аналог только
когда в trace нет внешней boundary. Внутренние `chain`, LLM и tool spans не
могут стать ответом пользователя.

Новая trace-схема требует отдельной versioned extraction strategy и тестовых
fixtures. Unknown schema обрабатывается fail-closed.

## Выходы и семантика

`monitoring_umr` и `parquet_test_dataset` содержат плоские строки:

- `query_id` — стабильный turn ID, для FIPA это
  `conversation_id|user_request_id`;
- `input_query` — транспортный текст пользователя без FIPA-envelope;
- `output_answer` и `agent_response` — финальный ответ пользователю;
- `route_label` — отдельное решение маршрутизации;
- trace/span IDs и source paths — provenance без сырых payload и метаданных
  пользователя;
- `reference_group_id` и `turn_index` определяются `assessment_mode`.

Для `assessment_mode=dialogue` Excel-выход упаковывается по требованию УМР: одна
строка на `session_id`, колонка `dialogue` содержит упорядоченные тройки
`(query_id, input_query, output_answer)`. DataFrame и parquet остаются плоскими —
это канонический внутренний транспорт мониторинга.

Для accuracy наблюдаемый `route_label` может заполнить объявленную prediction
колонку. Target (`GT`) из трейсов не синтезируется: он остаётся в эталонной
корзине, где `laim-baskets-adapter` материализует `main_metric`. Если prediction
извлечено, monitoring UMR готова для автоассесора без `GT`; финальный ответ
сохраняется отдельно от route-label.

## Диагностика

`processing_report` содержит:

- число protocol turn keys и полных опубликованных turn;
- `entry_without_exit` и `exit_without_entry`;
- эквивалентные и конфликтующие дубликаты;
- cross-trace и полные downstream correlation chains;
- unsupported trace schemas и неполные AEF boundaries;
- агрегированные issue codes и ограниченные примеры без текстов/payload;
- отдельный результат canonicalization и отсутствующие scoring sources.

Неполное покрытие имеет статус `partial`; семантически корректная выборка при
этом сохраняется. Ноль полных turn завершает вызов диагностической ошибкой.

Проверка `traces_validation_result` включена по умолчанию. Отсутствующий или
невалидный результат останавливает ноду до чтения данных. Обход допускается
только явной настройкой `ignore_traces_checks=true`, например для локального
исследовательского запуска. Это единственная настройка ноды: контур
автоматический, `agent_ci` и `distributive` приходят портом `selection`,
лимит примеров в отчёте (100) и список методов КМ зафиксированы в коде и
совпадают с MeasurementPlan `laim-baskets-adapter` (`identity`, `accuracy`,
`mean_criteria`, `all_criteria`, `majority`, `all_assessors`).

Excel-выход очищается от управляющих символов (Excel их не принимает);
`monitoring_umr` и parquet содержат текст без изменений.

## Runtime

Deploy-состав задаётся `descriptor.json` и включает только:

- `main.py` — контракт ноды и orchestration;
- `table_io.py` — транспорт DataFrame/parquet/xlsx;
- `trace_dialogue.py` — versioned extraction core;
- `canonical.py` — UMR projection и readiness gate.

Старые `trace_resolver.py`, `converter.py`, `src/` и `data_utils/` сохранены в
снимке только как lineage/reference и не входят в deploy `sourceFiles`.

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
python -m mypy --config-file \
  monitoring/laim-traces-dataset-converter/mypy.ini \
  monitoring/laim-traces-dataset-converter/main.py \
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
