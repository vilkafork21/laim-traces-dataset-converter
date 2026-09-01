## 1. Краткие выводы и замечания

В ходе анализа кода агента `gigachat-agent-insourcehub-c` были проверены следующие требования УМР:

| Требование УМР | Статус | Комментарий |
|----------------|--------|-------------|
| Подключение SDK AEF Tracing | Частично | SDK подключен, но через кастомный `AEFTracingClient` |
| Начало работы с контекстного менеджера | НЕТ | Отсутствуют `aef_input_request`, `aef_agent_start` |
| Запись результата в `aef.response.body` | НЕТ | Отсутствует |
| Классы сообщений (HumanMessage, AIMessage, etc.) | ДА | Используются корректно |
| `session_id` для группировки | Частично | Устанавливается `session_id_cvar.set(session_id)` но **без использования `aef_input_request`** |
| Трейсинг модулей GenAI-решения | Частично | Спаны собираются только через `AEFHandler` в LangGraph |
| Заполнение атрибутов спанов | НЕТ | Ручные спаны отсутствуют, то есть, не заполняются атрибуты `aef.input`, `aef.output` |

---

## 2. Детальный анализ реализации трейсинга

### 2.1 Инициализация трейсинга

**Файл:** `src/insource_agent/core/aef_tracing/client.py`

```python
class AEFTracingClient:
    def __init__(...):
        self._provider = OnlyLangraphAEFTracerProvider()  # ← Кастомный провайдер
        self._kafka_sender = AEFKafkaSender(...)
        proto_exporter = AEFProtobufSenderExporter(senders=[self._kafka_sender])
        self._provider.add_span_processor(proto_processor)
        self._aef_handler = AEFHandler(self._provider.get_tracer(__name__))
    
    def get_aef_handler(self) -> AEFHandler:
        return self._aef_handler
```

**Проблема:** Вместо стандартного `AEFTracerProvider` используется кастомный `OnlyLangraphAEFTracerProvider`. Это ограничивает функциональность, плюс ко всему нарушена логика инициализации handler'а в `src/insource_agent/core/aef_tracing/client.py`, поскольку мы всегда будем получать handler с `self._provider = OnlyLangraphAEFTracerProvider()`.

---

### 2.2 Использование трейсинга в API

**Файл:** `src/insource_agent/api/v1/router.py`

```python
@router.post("/invoke", ...)
async def gigachat_invoke(
    request: schemas.GigachatInvokeRequest,
    headers: dict = Depends(common_headers),
    langfuse: LangfuseClient = Depends(APP_CTX.get_langfuse),
    aef_tracing: AEFTracingClient = Depends(APP_CTX.get_aef_tracing)
):
    user_id, session_id = headers.get(H_USER_ID), headers.get(H_SESSION_ID)
    
    if aef_tracing and not langfuse:
        aef_handler = aef_tracing.get_aef_handler()
        config = {
            "configurable": {"thread_id": session_id, "checkpoint_ns": ""},
            "callbacks": [aef_handler],
        }
        message = HumanMessage(request.question)
        buckets.add_bucket(user_id)
        session_id_cvar.set(session_id)  # Установка session_id без aef_input_request
        
        response = workflow.invoke({...}, config, debug=debug)
```

**Проблемы:**
1. **Отсутствует `aef_input_request`** — входящий запрос не оборачивается в спан
2. **Отсутствует `aef_agent_start`** — запуск агента не обернут
3. **`session_id_cvar.set(session_id)`** вызывается вне контекста `aef_input_request`, что бессмысленно согласно требованиям УМР

---

### 2.3 Сбор спанов LangChain/LangGraph

Трейсинг LangGraph работает только через `AEFHandler`:

```python
config = {
    "callbacks": [aef_handler],  # Автоматический сбор chain/llm спанов
}
response = workflow.invoke({...}, config, debug=debug)
```

**Суть проблемы:** Спаны собираются **только** для узлов LangGraph (`chain`, `llm`). Все остальные ноды и операции **не трейсятся**:
- `censor_node` (guardrail логика)
- `interpreter` (интерпретация запроса)
- `controller` (управление потоком)
- `interview_node` (интервью пользователя)
- `statistics_node` (статистика)
- `label_node` (метки)

---

### 2.4 Отсутствие ручных спанов

В коде полностью отсутствует использование контекстных менеджеров:
- `aef_guardrail_run` — для проверки контента (censor_node)
- `aef_retriever_run` — если есть RAG
- `aef_tool_run` — для вызовов инструментов (в ToolNodeWithRateLimiter)
- `aef_agent_start` — для запуска агента
- `aef_input_request` — для входящего запроса

---

## 3. Список несоответствий требованиям УМР

1. Начало работы решения с контекстного менеджера `aef_input_request` — **НЕ ВЫПОЛНЕНО**
   - Входящий запрос `/invoke` не обернут в `aef_input_request`
   - `aef.response.body` не заполняется

2. Подключен трейсинг для модулей GenAI-решения — **ВЫПОЛНЕНО НЕ ПОЛНОСТЬЮ**
   - Спаны собираются только для LangGraph нод (спаны типа chain, llm)
   - Все специфичные узлы агента (censor, interpreter, interview, statistics, label) **не трейсятся**

3. Заполнены все атрибуты в сохраняемых спанах — **НЕ ВЫПОЛНЕНО**
   - Ручные спаны отсутствуют
   - Нет `aef.input`, `aef.output`, `aef.metadata` для бизнес-логики

4. Подключение SDK AEF Tracing — **ЧАСТИЧНО**
   - Используется кастомный `OnlyLangraphAEFTracerProvider` вместо стандартного `AEFTracerProvider`
   - Отсутствует экспорт в консоль/JSON (только Kafka)

5. `session_id` для группировки запросов — **ЧАСТИЧНО**
   - `session_id_cvar.set(session_id)` вызывается, но **без контекста** `aef_input_request`

---

## 4. Рекомендации по исправлениям

### 4.1 Базовая структура эндпоинтов (router.py)

```python
@router.post("/invoke", ...)
async def gigachat_invoke(...):
    user_id, session_id = headers.get(H_USER_ID), headers.get(H_SESSION_ID)
    
    # 1. Спан входящего запроса (ОБЯЗАТЕЛЬНО)
    with aef_input_request(
        span_name="input_request",
        headers=headers,
        body={"question": request.question, "user": request.user},
        path="/invoke",
        method="POST"
    ) as input_request_controller:
        
        # 2. Установка session_id ВНУТРИ aef_input_request
        session_id_cvar.set(session_id)
        
        # 3. Спан запуска агента (ОБЯЗАТЕЛЬНО)
        with aef_agent_start(
            input={"question": request.question, "user": request.user}
        ) as agent_controller:
            
            # 4. Вызов workflow (спаны chain/llm соберутся через AEFHandler)
            response = workflow.invoke({...}, config, debug=debug)
            
            # 5. Добавление результата в agent_controller
            answer = response["messages"][-1].content
            agent_controller.add_output_result(output={"answer": answer, "labels": response.get("labels", [])})
        
        # 6. Запись финального ответа в aef.response.body
        response_body = {"answer": answer, "labels": response.get("labels", [])}
        input_request_controller.add_response(
            headers={"Content-Type": "application/json"},
            body=response_body,
            http_code=200
        )
```

### 4.2 Обернуть специфичные тулы и ноды спанами

```python
# В censor_node.py (guardrail)
with aef_guardrail_run(input={"text": message.content}, span_name="content_filter") as guard_span:
    response = self.agent.send_human_message(message.content)
    content = response.as_json["answer"]
    guard_span.add_output_result(output={"allowed": content == "0"})
    if content != "0":
        return {**state, "censored": True, "messages": [AIMessage(content)], ...}
    return {**state, "censored": False}

# В tool_node.py (tool вызовы)
with aef_tool_run(input={"tool_calls": tool_calls}, span_name="tool_executor") as tool_span:
    result = tool_node.invoke(state)
    tool_span.add_tool_result(output=result)
```

### 4.3 Исправить AEFTracingClient (provider.py)

Заменить кастомный `OnlyLangraphAEFTracerProvider` на стандартный:

```python
from aef_tracing import AEFTracerProvider

class AEFTracingClient:
    def __init__(...):
        # self._provider = OnlyLangraphAEFTracerProvider()
        self._provider = AEFTracerProvider()
        # ...
```

---

## 5. Выводы

Интеграция агента InSourceHub с SDK AEF Tracing **не соответствует** требованиям УМР к трейсингу.

**Основные проблемы:**
- Отсутствуют обязательные контекстные менеджеры `aef_input_request` и `aef_agent_start`
- Трейсинг ограничен только LangGraph нодами, остальные важные ноды не трейсятся
- `session_id` устанавливается некорректно (без контекста `aef_input_request`)
- Используется кастомный провайдер вместо стандартного (+ в `router.py` есть условие `if aef_tracing and not langfuse:` и `elif langfuse`, но мы в любом случае получим `OnlyLangraphAEFTracerProvider()`)

**Рекомендация:** Переработка трейсинга согласно требованиям УМР
