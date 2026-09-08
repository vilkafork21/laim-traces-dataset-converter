"""Проекция тела спана по структуре сообщения, без правил для конкретных агентов."""

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass, replace

PART_SEPARATOR = "\n\n"
USER_ROLES = {"user", "human", "customer", "client"}
AGENT_ROLES = {"assistant", "ai", "agent", "bot"}
OTHER_ROLES = {"system", "tool", "function", "operator", "robot"}
# Эти поля принадлежат протоколу, а не содержимому реплики.
_METADATA = {
    "metadata",
    "meta_extra",
    "response_metadata",
    "additional_kwargs",
    "usage",
    "usage_metadata",
    "headers",
    "status",
    "status_code",
    "id",
    "name",
    "time",
    "timestamp",
    "created_at",
    "model",
    "finish_reason",
    "tool_calls",
    "function_call",
    "tool_call_id",
    "invalid_tool_calls",
    "conversation_id",
    "session_id",
    "trace_id",
}
_ROLE_FIELDS = {"role", "type", "speaker", "speaker_type"}
MALFORMED = object()


@dataclass(frozen=True)
class Leaf:
    path: str
    text: str


@dataclass(frozen=True)
class Projection:
    status: str
    text: str = ""
    path: str = ""
    candidates: tuple[Leaf, ...] = ()
    source: str = ""


def parse_body(raw: object) -> object:
    """JSON, двойной JSON и legacy Python repr; повреждённая структура не становится текстом."""
    current = raw
    decoded_text: str | None = None
    for layer in range(4):
        if current is None or (isinstance(current, float) and math.isnan(current)):
            return decoded_text
        if not isinstance(current, str):
            return (
                current
                if isinstance(current, (dict, list)) or decoded_text is None
                else decoded_text
            )
        value = current.strip()
        if not value or (value == "null" and layer == 0):
            return None
        if value[0] not in "[{\"'":
            return decoded_text if decoded_text is not None else current
        try:
            decoded = json.loads(value)
        except (ValueError, RecursionError):
            try:
                decoded = ast.literal_eval(value)
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                return decoded_text if layer else MALFORMED
        if decoded == current:
            return current
        if layer == 0 and isinstance(decoded, str):
            decoded_text = decoded
        current = decoded
    return MALFORMED


def looks_like_envelope(value: object) -> bool:
    return (
        isinstance(value, dict)
        and "content" in value
        and bool(
            {"sender", "receiver"} & value.keys()
            and {"conversation_id", "reply_with", "in_reply_to", "message_id"}
            & value.keys()
        )
    )


def envelopes(value: object, path: str = "", depth: int = 0) -> list[tuple[str, dict]]:
    """Конверт в теле или обёртке; содержимое конверта повторно не сканируется."""
    value = parse_body(value)
    if looks_like_envelope(value):
        return [(path, value)]
    if depth >= 32 or _role(value):
        return []
    if isinstance(value, list):
        return [
            found
            for index, item in enumerate(value)
            for found in envelopes(item, f"{path}[{index}]", depth + 1)
        ]
    if not isinstance(value, dict):
        return []
    return [
        found
        for key, item in value.items()
        if key not in _METADATA
        for found in envelopes(item, f"{path}.{key}", depth + 1)
    ]


def message_parts(envelope: dict) -> list[tuple[int, str]]:
    content = parse_body(envelope.get("content"))
    messages = parse_body(content.get("message")) if isinstance(content, dict) else None
    if not isinstance(messages, list):
        return []
    parts = []
    for index, item in enumerate(messages):
        if isinstance(item, dict) and item.get("type") == "text":
            value = item.get("value")
            if isinstance(value, str) and value.strip():
                parts.append((index, value.strip()))
    return parts


def _choose(projections: list[Projection]) -> Projection:
    for source in ("envelope", "message"):
        structured = [item for item in projections if item.source == source]
        if structured:
            projections = structured
            break
    if any(item.status == "malformed" for item in projections):
        return Projection("malformed", source=projections[0].source)
    leaves = tuple(
        leaf
        for item in projections
        for leaf in (
            (Leaf(item.path, item.text),) if item.status == "text" else item.candidates
        )
    )
    # Равные значения в разных полях не доказывают, какое поле является ответом.
    if len(leaves) > 1 or any(item.status == "ambiguous" for item in projections):
        return Projection("ambiguous", candidates=leaves, source=projections[0].source)
    source = projections[0].source if projections else ""
    return (
        Projection("text", leaves[0].text, leaves[0].path, source=source)
        if leaves
        else Projection("empty", source=source)
    )


def _role(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    roles = {str(item).casefold() for key, item in value.items() if key in _ROLE_FIELDS}
    known = roles & (USER_ROLES | AGENT_ROLES | OTHER_ROLES)
    if len(known) != 1:
        return ""
    role = known.pop()
    return (
        "request"
        if role in USER_ROLES
        else "response"
        if role in AGENT_ROLES
        else "other"
    )


def _content(value: object, path: str, side: str, depth: int) -> Projection:
    """Текстовые части чат-сообщения; изображения и вызовы инструментов не являются текстом."""
    if isinstance(value, str):
        return (
            Projection("text", value.strip(), path)
            if value.strip()
            else Projection("empty")
        )
    if not isinstance(value, list):
        return _project(value, path, side, depth + 1)
    parts = []
    for index, part in enumerate(value):
        part_path = f"{path}[{index}]"
        if isinstance(part, str):
            parts.append(_content(part, part_path, side, depth + 1))
        elif isinstance(part, dict) and part.get("type") in {
            "text",
            "input_text",
            "output_text",
        }:
            key = "text" if "text" in part else "value"
            parts.append(_content(part.get(key), f"{part_path}.{key}", side, depth + 1))
    if any(part.status not in {"text", "empty"} for part in parts):
        return _choose(parts)
    text = PART_SEPARATOR.join(part.text for part in parts if part.status == "text")
    return Projection("text", text, f"{path}[*]") if text else Projection("empty")


def _project(value: object, path: str, side: str, depth: int) -> Projection:
    if depth > 32:
        return Projection("malformed")
    decoded = parse_body(value)
    # Строка внутри валидного JSON может содержать кавычки, скобки или слово null.
    # Повторно раскрываются только закодированные контейнеры.
    if not (depth and isinstance(value, str)) or isinstance(decoded, (dict, list)):
        value = decoded
    if value is MALFORMED:
        return Projection("malformed")
    if isinstance(value, str):
        return (
            Projection("text", value.strip(), path)
            if value.strip()
            else Projection("empty")
        )
    if (
        isinstance(value, dict)
        and "content" in value
        and {"sender", "receiver"} & value.keys()
    ):
        content = value.get("content")
        decoded_content = parse_body(content)
        if isinstance(decoded_content, (dict, list)):
            content = decoded_content
        if isinstance(content, dict) and isinstance(content.get("message"), list):
            parts = message_parts(value)
            text = PART_SEPARATOR.join(text for _, text in parts)
            return (
                Projection(
                    "text", text, f"{path}.content.message[*].value", source="envelope"
                )
                if text
                else Projection("empty", source="envelope")
            )
        return replace(
            _project(content, f"{path}.content", side, depth + 1), source="envelope"
        )
    role = _role(value)
    if role:
        if role != side:
            return Projection("empty", source="message")
        for key in ("content", "text", "value"):
            if key in value:
                return replace(
                    _content(value[key], f"{path}.{key}", side, depth), source="message"
                )
        return replace(
            _choose(
                [
                    _project(item, f"{path}.{key}", side, depth + 1)
                    for key, item in value.items()
                    if key not in _METADATA | _ROLE_FIELDS
                ]
            ),
            source="message",
        )
    if isinstance(value, list):
        roles = [_role(item) for item in value]
        if any(roles):
            if not all(roles):
                return Projection("ambiguous")
            # Пустое последнее assistant-сообщение с tool_calls не заменяется предыдущим ответом.
            for index in range(len(value) - 1, -1, -1):
                if roles[index] == side:
                    return _project(value[index], f"{path}[{index}]", side, depth + 1)
                if side == "response" and roles[index] == "request":
                    break
            return Projection("empty", source="message")
        return _choose(
            [
                _project(item, f"{path}[{index}]", side, depth + 1)
                for index, item in enumerate(value)
            ]
        )
    if isinstance(value, dict):
        return _choose(
            [
                _project(item, f"{path}.{key}", side, depth + 1)
                for key, item in value.items()
                if key not in _METADATA
            ]
        )
    return Projection("empty")


def project_body(value: object, *, side: str) -> Projection:
    """Реплика, пустое тело, неоднозначность или повреждённая структура с путями кандидатов."""
    if side not in {"request", "response"}:
        raise ValueError(f"side должен быть request или response: {side!r}")
    return _project(value, "", side, 0)
