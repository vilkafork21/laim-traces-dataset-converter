"""Ограниченные запросы к LLM-шлюзу контура; ответ содержит только ID кандидатов."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import os
from typing import AsyncIterator
from urllib.parse import urlsplit

import httpx

DEFAULT_URL = "http://sds-ai-gateway:8097/api/v1/chat/completions"
_MAX_INPUT_BYTES = 98_304
_MAX_RESPONSE_BYTES = 131_072


class LLMError(ValueError):
    """Безопасный код отказа без содержимого запроса или ответа шлюза."""

    def __init__(self, code: str) -> None:
        self.code = code
        self.usage: dict | None = None
        super().__init__(f"LLM-анализ не выполнен: {code}")


@dataclass(frozen=True)
class LLMSettings:
    mode: str = "off"
    model: str = ""
    max_calls: int = 6
    budget_seconds: int = 180

    def __post_init__(self) -> None:
        if self.mode not in ("off", "shadow"):
            raise ValueError(f"llm_mode должен быть off или shadow: {self.mode!r}")
        for name, value, maximum in (
            ("llm_max_calls", self.max_calls, 6),
            ("llm_budget_seconds", self.budget_seconds, 180),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(
                    f"{name} должен быть целым от 1 до {maximum}: {value!r}"
                )
        if not isinstance(self.model, str) or len(self.model) > 256:
            raise ValueError("model_id должен быть строкой длиной до 256 символов")
        model = (
            self.model.strip()
            or os.environ.get("LAIM_LLM_MODEL", "").strip()
            or "glm-5.2"
        )
        if len(model) > 256 or any(
            ord(char) < 32 or 0xD800 <= ord(char) <= 0xDFFF for char in model
        ):
            raise ValueError("model_id / LAIM_LLM_MODEL содержит недопустимое значение")
        object.__setattr__(self, "model", model)


@asynccontextmanager
async def gateway_client() -> AsyncIterator[httpx.AsyncClient]:
    url = os.environ.get("LAIM_LLM_URL", DEFAULT_URL).strip()
    try:
        parts = urlsplit(url)
        valid = (
            parts.scheme in {"http", "https"}
            and parts.hostname
            and not (parts.username or parts.password or parts.query or parts.fragment)
        )
        parts.port
        httpx.URL(url)
    except (ValueError, httpx.InvalidURL):
        valid = False
    if not valid:
        raise LLMError("gateway_configuration")
    key = os.environ.get("AI_GATEWAY_API_KEY", "").strip()
    if not key and url.rstrip("/") == DEFAULT_URL:
        key = "123"
    if not key or not key.isascii() or any(ord(char) < 32 for char in key):
        raise LLMError("gateway_credentials")
    async with httpx.AsyncClient(
        base_url=url,
        headers={"Authorization": f"Bearer {key}"},
        timeout=httpx.Timeout(60, connect=10),
        follow_redirects=False,
        trust_env=False,
    ) as client:
        yield client


def _object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise LLMError("invalid_json")
        result[key] = value
    return result


def _decode(text: str) -> dict:
    try:
        result = json.loads(text, object_pairs_hook=_object)
    except (ValueError, RecursionError):
        raise LLMError("invalid_json") from None
    if not isinstance(result, dict):
        raise LLMError("invalid_json")
    return result


def _selection(content: object, allowed: dict[str, dict[str, str]]) -> dict | None:
    if not isinstance(content, str):
        raise LLMError("invalid_response")
    content = content.strip()
    if content.startswith("<think>"):
        _, closing, content = content.partition("</think>")
        if not closing:
            raise LLMError("incomplete_reasoning")
        content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines[0] not in {"```json", "```"} or lines[-1] != "```":
            raise LLMError("invalid_json")
        content = "\n".join(lines[1:-1])
    selection = _decode(content)
    if set(selection) != {"query", "response"}:
        raise LLMError("invalid_selection")
    if all(value is None for value in selection.values()):
        return None
    if any(
        not isinstance(value, str) or value not in allowed[side]
        for side, value in selection.items()
    ):
        raise LLMError("invalid_selection")
    return selection


async def choose_fields(
    client: httpx.AsyncClient,
    settings: LLMSettings,
    prompt: str,
    payload: str,
    allowed: dict[str, dict[str, str]],
) -> tuple[dict | None, dict | None]:
    if len((prompt + payload).encode()) > _MAX_INPUT_BYTES:
        raise LLMError("input_limit")
    request = {
        "model": settings.model,
        "temperature": 0,
        "top_p": 0.05,
        "max_tokens": 16_384,
        "stream": False,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": payload},
        ],
    }
    try:
        async with asyncio.timeout(60):
            async with client.stream(
                "POST", str(client.base_url).rstrip("/"), json=request
            ) as response:
                if response.status_code != 200:
                    raise LLMError(f"http_{response.status_code}")
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(body) + len(chunk) > _MAX_RESPONSE_BYTES:
                        raise LLMError("response_limit")
                    body.extend(chunk)
    except TimeoutError:
        raise LLMError("request_timeout") from None
    except httpx.RequestError:
        raise LLMError("transport_error") from None
    try:
        envelope = _decode(body.decode("utf-8"))
    except UnicodeDecodeError:
        raise LLMError("invalid_json") from None
    usage = envelope.get("usage")
    token_usage = (
        {
            key: usage[key]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if type(usage.get(key)) is int and usage[key] >= 0
        }
        if isinstance(usage, dict)
        else {}
    )
    try:
        choices = envelope.get("choices")
        if (
            not isinstance(choices, list)
            or len(choices) != 1
            or not isinstance(choices[0], dict)
        ):
            raise LLMError("invalid_response")
        choice = choices[0]
        message = choice.get("message")
        if choice.get("finish_reason") != "stop" or not isinstance(message, dict):
            raise LLMError("incomplete_response")
        if (
            message.get("tool_calls")
            or message.get("function_call")
            or message.get("refusal")
        ):
            raise LLMError("unsupported_response")
        selection = _selection(message.get("content"), allowed)
    except LLMError as exc:
        exc.usage = token_usage or None
        raise
    return selection, token_usage or None
