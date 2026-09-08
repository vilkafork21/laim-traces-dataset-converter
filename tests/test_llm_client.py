from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json

import httpx
import pytest

import llm_client
from llm_client import LLMError, LLMSettings, choose_fields

ALLOWED = {"query": {"q0": ""}, "response": {"a0": ".answer", "a1": ".note"}}


def completion(content='{"query":"q0","response":"a0"}', **changes):
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": content},
                **changes,
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


def call(handler, payload="{}"):
    async def run():
        async with httpx.AsyncClient(
            base_url=llm_client.DEFAULT_URL,
            transport=httpx.MockTransport(handler),
        ) as client:
            return await choose_fields(
                client, LLMSettings("shadow"), "Правила", payload, ALLOWED
            )

    return asyncio.run(run())


def test_exact_gateway_path_closed_selection_and_token_usage():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=completion())

    choice, usage = call(handler)
    assert choice == {"query": "q0", "response": "a0"}
    assert usage["total_tokens"] == 8
    assert len(requests) == 1
    assert str(requests[0].url) == llm_client.DEFAULT_URL
    body = json.loads(requests[0].content)
    assert body["max_tokens"] == 16384
    assert body["stream"] is False
    assert "tools" not in body


@pytest.mark.parametrize(
    "content,code",
    [
        ('{"query":"q0","response":"new text"}', "invalid_selection"),
        ('{"query":"q0","response":null}', "invalid_selection"),
        ('{"query":[],"response":"a0"}', "invalid_selection"),
        ('{"query":"q0","response":"a0","code":"print(1)"}', "invalid_selection"),
        ('{"query":"q0","response":"a0","response":"a1"}', "invalid_json"),
        ('{"query":"q0"', "invalid_json"),
        ("[]", "invalid_json"),
        ("<think>Незавершённое рассуждение", "incomplete_reasoning"),
        ("```json\n{}", "invalid_json"),
        (None, "invalid_response"),
    ],
)
def test_rejects_untrusted_or_incomplete_selections(content, code):
    with pytest.raises(LLMError) as error:
        call(lambda _: httpx.Response(200, json=completion(content)))
    assert error.value.code == code


@pytest.mark.parametrize(
    "prefix,suffix", [("", ""), ("<think>Анализ</think>\n```json\n", "\n```")]
)
def test_abstention_and_balanced_reasoning(prefix, suffix):
    selection, _ = call(
        lambda _: httpx.Response(
            200,
            json=completion(
                prefix + '{"query":null,"response":null}' + suffix,
            ),
        )
    )
    assert selection is None


@pytest.mark.parametrize("status", [301, 400, 401, 429, 500, 503])
def test_http_errors_are_not_retried_or_exposed(status):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(status, text="Секретное тело ответа")

    with pytest.raises(LLMError) as error:
        call(handler)
    assert error.value.code == f"http_{status}"
    assert "Секретное" not in str(error.value)
    assert len(seen) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"finish_reason": "length"},
        {"message": {"content": "{}", "tool_calls": [{"id": "call"}]}},
        {"message": {"content": "{}", "refusal": "Отказ"}},
    ],
)
def test_incomplete_and_tool_responses_are_rejected(changes):
    with pytest.raises(LLMError):
        call(lambda _: httpx.Response(200, json=completion(**changes)))


def test_input_and_response_size_limits_prevent_unbounded_payloads():
    with pytest.raises(LLMError, match="input_limit"):
        call(lambda _: pytest.fail("Запрос не должен отправляться"), "я" * 100_000)
    with pytest.raises(LLMError, match="response_limit"):
        call(lambda _: httpx.Response(200, content=b"x" * 131_073))


def test_missing_usage_is_unknown():
    response = completion()
    del response["usage"]
    assert call(lambda _: httpx.Response(200, json=response))[1] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", "apply"),
        ("mode", []),
        ("max_calls", True),
        ("max_calls", 0),
        ("max_calls", 7),
        ("budget_seconds", 181),
        ("budget_seconds", 1.5),
        ("model", None),
        ("model", "\ud800"),
    ],
)
def test_settings_reject_invalid_values(field, value):
    with pytest.raises(ValueError):
        LLMSettings(**{field: value})


def test_model_precedence(monkeypatch):
    monkeypatch.setenv("LAIM_LLM_MODEL", "contour-model")
    assert LLMSettings(model=" explicit ").model == "explicit"
    assert LLMSettings().model == "contour-model"
    monkeypatch.delenv("LAIM_LLM_MODEL")
    assert LLMSettings().model == "glm-5.2"


def test_gateway_credentials_stay_out_of_errors(monkeypatch):
    monkeypatch.setenv("LAIM_LLM_URL", "https://gateway.example/completions")
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)

    async def run():
        async with llm_client.gateway_client():
            pytest.fail("Без ключа другой шлюз не используется")

    with pytest.raises(LLMError, match="gateway_credentials"):
        asyncio.run(run())


@pytest.mark.parametrize(
    "url",
    [
        "http://example.test/\x01",
        "http://example.test/\ud800",
        "http://[invalid",
        "https://user:secret@example.test/path",
    ],
)
def test_invalid_gateway_url_is_a_domain_error(monkeypatch, url):
    monkeypatch.setattr(
        llm_client.os, "environ", {"LAIM_LLM_URL": url, "AI_GATEWAY_API_KEY": "secret"}
    )

    async def run():
        async with llm_client.gateway_client():
            pytest.fail("Некорректный адрес не принимается")

    with pytest.raises(LLMError, match="gateway_configuration"):
        asyncio.run(run())


def test_truncated_response_still_reports_token_usage():
    with pytest.raises(LLMError) as error:
        call(lambda _: httpx.Response(200, json=completion(finish_reason="length")))
    assert error.value.usage == {
        "prompt_tokens": 5,
        "completion_tokens": 3,
        "total_tokens": 8,
    }


@asynccontextmanager
async def mock_gateway(handler):
    async with httpx.AsyncClient(
        base_url=llm_client.DEFAULT_URL,
        transport=httpx.MockTransport(handler),
    ) as client:
        yield client
