# Целевая архитектура извлечения диалогов

## Решение

Единицей данных является turn, доказанный транспортным протоколом. Trace —
только контейнер spans и provenance: один turn может пересекать trace, а один
trace может содержать несколько внутренних агентов.

Публичный API ядра:

```python
extract_turns(
    spans: pandas.DataFrame,
    config: ExtractionConfig,
) -> ExtractionResult
```

`ExtractionResult` содержит:

1. `turns` — только полные пары с раздельными `input_query`,
   `agent_response`, `route_label` и source provenance;
2. `issues` — неполные, конфликтующие и неподдерживаемые случаи без raw payload;
3. `report` — знаменатели, coverage и versioned extraction strategy.

Проекция в тестовую корзину отделена от извлечения:

```text
raw spans
   │
   ▼
versioned extraction strategies
   │  turns + issues + evidence
   ▼
monitoring UMR projection
   │  metric-aware readiness gate
   ├── flat DataFrame / parquet
   └── QA или packed-dialogue Excel
```

Это разделение запрещает менять смысл наблюдённого ответа ради формата метрики.

## Инварианты

- Никакого generic deep-search по `text`, `answer`, `message`, `content`.
- Ответ маршрутизатора и финальный ответ пользователю — разные сигналы.
- Join FIPA выполняется по correlation ID и может пересекать trace.
- Противоречивые реплики не разрешаются выбором первой/последней строки.
- Unknown schema не публикуется до появления versioned mapping и fixtures.
- Target метрики не выводится из prediction или текста ответа.
- В UMR не попадают raw envelopes, cookies, headers и user metadata.
- Знаменатель качества — protocol turn keys/boundaries, не число trace groups.

## Границы готовности

Extraction readiness означает, что каждая опубликованная строка имеет
доказанный запрос и финальный ответ. Assessment readiness дополнительно требует
наблюдаемые monitoring-источники `monitoring_metric`. Для accuracy FIPA route
даёт prediction, а target остаётся в эталонной корзине: baskets-adapter заранее
материализует `main_metric`, по которому калибруется автоассесор.

Частичное окно не уничтожает корректные turn: gaps остаются в отчёте и меняют
статус на `partial`. Политика минимального coverage должна задаваться владельцем
мониторинга отдельно, потому что она зависит от окна витрины и SLA агента.

## Расширение на новых агентов

Для новой семьи трейсов добавляется стратегия с:

1. именем и версией схемы;
2. точными request/response paths;
3. стабильным turn key;
4. правилами duplicate/conflict;
5. положительными, incomplete и conflict fixtures;
6. acceptance-счётчиками на реальном sample.

Наличие похожих текстовых ключей не считается поддержкой схемы.

## Переиспользование

`trace_dialogue.py` — целевое ядро для converter и pre-processing Andes. По
правилу самодостаточности нод runtime-import между каталогами не допускается:
при интеграции Andes один и тот же versioned source artifact включается в
deploy-состав каждой ноды, а parity-тест прогоняет одинаковые fixtures через
обе копии. После такой интеграции `laim-anomaly-dialogue` больше не должен
восстанавливать пары постфактум.

## Этапы внедрения

1. Converter: FIPA/AEF core, UMR projection, diagnostics — реализовано.
2. DataLab: прогон expanded mart по агентам разных schema families и фиксация
   acceptance baselines.
3. Andes: заменить first-input/last-output на тот же extraction contract до
   построения features и scoring.
4. Basket references: использовать target/GT только для материализации
   эталонного `main_metric` и калибровки автоассесора.
5. Удалить post-converter и эвристические fallback после parity-проверки.
