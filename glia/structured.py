"""Structured outputs: get a typed object back, not a string to parse.

The trick is provider-agnostic and glass-box: define one tool whose schema *is*
the shape you want, force the model to call it, and read the validated
arguments. No special output mode, no provider lock-in — it works on any
:class:`~glia.llm.LLM`, including the offline :class:`EchoLLM`.

``schema`` can be a JSON-schema ``dict`` (you get a ``dict`` back), a
``dataclass`` type, or a Pydantic model (you get an instance back). Pydantic is
imported only if you use it.
"""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Callable
from typing import Any

from .errors import StructuredOutputError
from .llm import LLM, LLMRequest, ToolSchema
from .tools import _type_to_schema
from .types import user


async def generate_structured(
    llm: LLM,
    prompt: str,
    schema: Any,
    *,
    system: str | None = None,
    tool_name: str = "respond",
    description: str = "Return the final answer in this exact structure.",
    max_tokens: int = 1024,
) -> Any:
    """Ask ``llm`` for output matching ``schema`` and return it coerced."""
    json_schema, coerce = _resolve(schema)
    tool = ToolSchema(name=tool_name, description=description, parameters=json_schema)
    request = LLMRequest(
        messages=[user(prompt)],
        system=system,
        tools=[tool],
        tool_choice=tool_name,
        max_tokens=max_tokens,
    )
    response = await llm.generate(request)
    calls = response.message.tool_uses()
    if not calls:
        raise StructuredOutputError(
            f"model returned no structured output (stop_reason={response.stop_reason!r})"
        )
    try:
        return coerce(calls[0].input)
    except Exception as exc:  # noqa: BLE001
        raise StructuredOutputError(f"could not coerce output into schema: {exc}") from exc


# -- schema resolution ---------------------------------------------------------

Coerce = Callable[[dict[str, Any]], Any]


def _resolve(schema: Any) -> tuple[dict[str, Any], Coerce]:
    if isinstance(schema, dict):
        return _ensure_object(schema), lambda data: data

    if dataclasses.is_dataclass(schema) and isinstance(schema, type):
        return _dataclass_schema(schema), lambda data: schema(**data)

    # Pydantic v2 model.
    if hasattr(schema, "model_json_schema") and hasattr(schema, "model_validate"):
        json_schema = _ensure_object(dict(schema.model_json_schema()))
        return json_schema, schema.model_validate

    raise StructuredOutputError(
        "schema must be a JSON-schema dict, a dataclass type, or a Pydantic model"
    )


def _ensure_object(schema: dict[str, Any]) -> dict[str, Any]:
    schema = dict(schema)
    schema.setdefault("type", "object")
    schema.setdefault("additionalProperties", False)
    return schema


def _dataclass_schema(cls: type) -> dict[str, Any]:
    hints = typing.get_type_hints(cls, include_extras=True)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for f in dataclasses.fields(cls):
        properties[f.name] = _type_to_schema(hints.get(f.name, str))
        has_default = f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING  # type: ignore[misc]
        if not has_default:
            required.append(f.name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
