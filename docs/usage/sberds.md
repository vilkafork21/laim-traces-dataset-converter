# Подключение в Sber DS

Нода преобразует спаны AEF в monitoring UMR внутри мониторингового workflow.
Режим UMR задаёт метрика, подготовленная `laim-baskets-adapter`.

## Установка

Импортируйте пакет ноды с `descriptor.json`, `requirements.txt` и исходниками
из `script.runConfiguration.sourceFiles`. Используется образ `py312-simple`,
точка входа — `main.main`.

## Соединения

| Источник | Вход конвертера |
|---|---|
| Таблица спанов AEF из источника данных | `monitoring_traces` |
| `monitoring_metric` из `laim-baskets-adapter` | `monitoring_metric` |
| Результат проверки схемы и качества трейсов | `traces_validation_result` |
| Параметры выбранного агента и версии | `selection` |

`selection` может содержать `agent_ci`, `distributive`, `solution_version`.
Без `agent_ci` таблица должна содержать одного агента. Версию и временное окно
отбирают в источнике данных: `distributive` используется в пути артефакта,
а `solution_version` добавляется в UMR.

Передавайте полные спаны выбранного окна, включая строки с пустыми
`input_text` или `output_text`: запрос и ответ могут находиться в разных спанах
и trace. Сохраните `parent_span_id`, чтобы нода могла отличить внешнюю границу
от вложенного вызова.

## Настройки и результат

По умолчанию `ignore_traces_checks=False`: обязателен DQ-вердикт.
`min_extraction_coverage=0.9` задаёт минимальное покрытие для готовности данных.
Отключение DQ-гейта отражается в отчёте как `bypassed_dq`.

`monitoring_umr` передаётся следующей ноде оценки вместе с `processing_report`.
Перед scoring проверяйте `ready_for_scoring` и `data_readiness`: наличие строк
само по себе не означает готовность набора.

Для сохранения подключите `umr_artifact` и `settings` к платформенному
file-writer. `parquet_test_dataset` содержит байты parquet с теми же данными.
Конвертер сам не записывает файлы во внешнее хранилище.

Описание колонок, правил извлечения и причин неполноты — в [README](../../README.md).
