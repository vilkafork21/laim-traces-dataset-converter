# Production readiness

Дата проверки: 2026-08-19.

Этот документ фиксирует историческую сборку r5. Текущий локальный модуль содержит
более строгий validation/extraction gate и не должен называться новым release,
пока не собран и не проверен отдельный deploy-артефакт.

## Закрытые риски r5 (2026-08-19, EDA CI09997554 + регрессия CI09997438)

- контрагент определяется по данным (инициаторы conversation), regex имён
  endpoint удалён: CI09997554 (`p-credit-helper`, вход только от роутера
  `d-credit-helper`) перестаёт падать «Не найдено полных turn»; на CI09997438
  результат идентичен r4 — 2 151 turn, те же тексты, counterparts
  `agent_human` 2 616 / `elchhumanagent` 33, уточняющие request
  `p-credit-helper` (94, из них 25 на обрезанном начале окна) контрагентом не
  признаются;
- вход вида `[label, вопрос]` разделяется: метка → `scenario`, вопрос →
  `input_query` (по EDA это форма dispatch роутера к нижестоящему агенту);
- `failure` без текста (15 на CI09997438) учитывается отдельно
  (`failures_without_text`), а не как malformed;
- trace без граничного span (`no_boundary_span`, у CI09997554 это 39,7% trace —
  только `chain`/`output_request`/`llm`) отделены от `unsupported_trace_schema`.

## Закрытые риски r4 (2026-08-19, прогон на полном parquet CI09997438)

- `monitoring_umr` содержит только поля UMR «Формата тестового датасета»:
  `session_id, query_id, input_query, output_answer, scenario, input_query_count,
  reference_group_id, turn_index` (+ prediction-колонка для accuracy);
  `agent_response`-дубль, trace/span ID, source paths, флаги и латентность из UMR
  убраны — агрегаты остались в `processing_report`;
- согласованность с ассесором проверена через его копию `laim_monitoring/core.py`:
  `normalize_tdc_monitoring` + `unitize` принимают плоскую UMR (2 151 unit в qa,
  1 772 session-unit в dialogue) и упакованный dialogue-Excel; label-space
  `class` в monitoring (14 меток) и в эталоне adapter-а (14 меток) пересекается
  по 13 (`refinance` есть только в трафике, `common_credits` — только в корзине).

- `route_label` брался из `outgoing.receiver` и был константой `p-credit-helper`
  на всех 2 151 turn; теперь для роутера вида `[label, echo вопроса]` берётся
  текстовая метка (`liabilities` 722, `available_credits` 422, `decline` 308, …;
  11 из 12 классов есть в GT эталонной корзины), источник — `route_source_path`;
- `agent_response` собирается из всех текстовых частей `content.message`
  (виджеты пропускаются): в выборке 45 ответов содержали вторую текстовую часть,
  которая раньше терялась;
- `all_assessors` (метод MeasurementPlan baskets-adapter) принимается;
- управляющие символы в тексте больше не роняют Excel-экспорт
  (`IllegalCharacterError`); DataFrame/parquet не меняются;
- в `descriptor.json` осталась единственная настройка `ignore_traces_checks`;
  `min_meaningful_len`, `max_issue_examples`, `distributive` удалены из
  контракта ноды (`distributive` — из порта `selection`);
- строки с пустым `agent_id` не приписываются найденному агенту в авторежиме;
- `extract_turns` разбит на `_collect_fipa_events` / `_join_fipa_turns` /
  `_fipa_turn` / `_collect_aef_turns`; удалены неиспользуемый `parent_span_id`,
  недостижимая ветка `turn_latency_ms`, дубль списка колонок и повторная
  валидация метрики в `canonical.py`.

## Закрытые риски повторного code review

- комментарии и docstrings runtime-кода переведены на русский язык;
- `traces_validation_result` обязателен по умолчанию, обход только явный;
- повреждённый JSON не публикуется как обычный пользовательский текст;
- FIPA-envelope требует endpoint, correlation ID и content;
- turn с конфликтующим `session_id` не публикуется;
- prediction/target не могут перезаписать поля UMR;
- target accuracy остаётся в эталонной корзине и не синтезируется из трейсов;
- query-параметры URL дистрибутива не попадают в HDFS-путь;
- настройки целочисленных лимитов валидируются строго.

## Проверки

- unit/negative tests: 31 passed;
- `ruff check`, `ruff format --check`: passed;
- `mypy` (mypy.ini, без pandas-stubs): passed, 8 source/test files;
- `descriptor.json`: valid JSON; `zip -T`: passed;
- изолированный импорт и smoke-run из production ZIP: passed;
- CI09997438 regression (70 946 span, 3 776 trace): 2 151 полных FIPA-turn,
  782/700 незамкнутых входов/выходов, 6 cross-trace, `class` заполнен
  наблюдаемой меткой класса в 2 151 из 2 151 строк, `GT` отсутствует;
- смоделированные `monitoring_metric` adapter-а для CI09997438 (accuracy),
  CI09997554 (majority/all_assessors, dialogue), CI09877398
  (identity/all_criteria), CI0984670 (mean_criteria) проходят валидацию ноды;
  `status=not_computable` от adapter-а останавливает ноду — это осознанный
  fail-closed: без baseline KM дальше по контуру не считается.

Известные границы: агенты без `input_request`/`start_agent`/FIPA-envelope
(по `span_distribution.csv` ~10 пар agent×distributive, например LangGraph-only
`chain`) завершаются `TraceExtractionError: unsupported` — им нужна отдельная
versioned strategy. Контрагент выводится из данных: отправитель должен открывать conversation
своим `request` в большинстве своих request; агент, у которого в окне выгрузки
нет ни одного полного request→reply с инициатором, получает 0 turn.

## Release artifact

- файл: `dist/laim-traces-dataset-converter-production-20260819-r5.zip`;
- SHA-256: `4ea0ba17878b2af0d0b5aa0623e673920a40740ae53df8c1ae09202b5cb7348b`;
- состав: семь файлов, только descriptor, requirements, README и runtime;
- проверено локально на Python 3.14.3 (mypy python_version=3.12, синтаксис
  stdlib новее 3.12 не используется).

Согласованный ассесор: `dist/laim-asessor-agent-production-20260819-r2.zip`,
SHA-256 `a0b5534ee069050595f1640613db268d9ec2c1212a0e6d7c0d82a35846d4c0eb`.

## Связь с эталонной корзиной

Для accuracy-метрики CI09997438 converter публикует наблюдаемое prediction из
`route_label`, а `GT` остаётся только в reference UMR baskets-adapter. Ассесор
калибруется по материализованному `main_metric` эталона и не требует `GT` на
monitoring-входе.
