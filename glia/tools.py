"""Tools are plain Python functions. The ``@tool`` decorator reads their type
hints and docstring to build a JSON schema — no schema DSL, no base class.

    @tool
    async def get_weather(city: str, unit: str = "celsius") -> str:
        '''Look up the current weather for a city.'''
        ...

The function stays callable exactly as written; the decorator attaches a
:class:`Tool` describing it. Both sync and async functions work. Argument
schemas are derived from stdlib ``typing`` — so there is no Pydantic dependency
in the core, though you can pass any JSON schema by hand if you outgrow the
inference.
"""

from __future__ import annotations

import asyncio
import inspect
import types as _types
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_args, get_origin

from .errors import ToolError
from .llm import ToolSchema
from .types import ToolResult

_JSON_SCALARS: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


@dataclass
class Tool:
    """A callable plus the schema the model uses to call it.

    Attributes are all plain data — you can read ``tool.parameters`` to see
    exactly what the model is shown, and ``tool.func`` is your original function.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    is_async: bool

    def schema(self) -> ToolSchema:
        """The wire-level view handed to a provider."""
        return ToolSchema(name=self.name, description=self.description, parameters=self.parameters)

    async def call(self, arguments: dict[str, Any]) -> Any:
        """Invoke the underlying function, awaiting it if it is async."""
        if self.is_async:
            return await self.func(**arguments)
        # Run sync tools off the event loop so a slow tool can't block the agent.
        return await asyncio.to_thread(self.func, **arguments)


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Turn a function into a :class:`Tool`. Usable bare (``@tool``) or with
    arguments (``@tool(name="search")``)."""

    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or fn.__name__
        tool_desc = description or inspect.getdoc(fn) or ""
        parameters = _schema_from_signature(fn)
        fn.tool = Tool(  # type: ignore[attr-defined]
            name=tool_name,
            description=tool_desc.strip(),
            parameters=parameters,
            func=fn,
            is_async=asyncio.iscoroutinefunction(fn),
        )
        return fn

    if func is not None:
        return wrap(func)
    return wrap


def as_tool(obj: Any) -> Tool:
    """Coerce a decorated function or a :class:`Tool` into a :class:`Tool`."""
    if isinstance(obj, Tool):
        return obj
    attached = getattr(obj, "tool", None)
    if isinstance(attached, Tool):
        return attached
    raise ToolError(f"{obj!r} is not a tool — decorate it with @tool")


def _schema_from_signature(fn: Callable[..., Any]) -> dict[str, Any]:
    """Build a JSON-schema ``object`` from a function's typed parameters.

    Supports scalars, ``list``/``dict``, ``Optional[...]``, ``Literal[...]``
    (rendered as an enum), and ``Annotated[T, "description"]``. Anything else
    falls back to an unconstrained value so the tool still works.
    """
    sig = inspect.signature(fn)
    hints = typing.get_type_hints(fn, include_extras=True)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if pname == "self" or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = hints.get(pname, str)
        properties[pname] = _type_to_schema(annotation)
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _type_to_schema(annotation: Any) -> dict[str, Any]:
    origin = get_origin(annotation)

    # Annotated[T, "description", ...] -> schema of T with a description.
    if origin is typing.Annotated:
        base, *extras = get_args(annotation)
        schema = _type_to_schema(base)
        for extra in extras:
            if isinstance(extra, str):
                schema = {**schema, "description": extra}
                break
        return schema

    if annotation in _JSON_SCALARS:
        return {"type": _JSON_SCALARS[annotation]}

    if origin is typing.Literal:
        options = list(get_args(annotation))
        scalar = type(options[0]) if options else str
        return {"type": _JSON_SCALARS.get(scalar, "string"), "enum": options}

    # Optional[T] / Union[T, None] / PEP 604 `T | None` -> schema of T (JSON
    # schema treats it as simply-nullable; keeping it flat is friendlier to
    # strict-mode providers). Both spellings must be handled: typing.Union and
    # the newer types.UnionType have *different* origins.
    if origin is typing.Union or origin is _types.UnionType:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return _type_to_schema(non_none[0])
        return {}  # genuine union — leave unconstrained

    if origin in (list, list):
        args = get_args(annotation)
        item = _type_to_schema(args[0]) if args else {}
        return {"type": "array", "items": item}

    if origin in (dict, dict):
        return {"type": "object"}

    # Unknown / bare type: don't over-constrain.
    return {}


class ToolRegistry:
    """A name → :class:`Tool` map that runs tools and always returns a
    :class:`ToolResult` — never raises out of a tool call.

    A tool raising an exception becomes a ``ToolResult(is_error=True)`` so the
    model can see the failure and recover, exactly as a production agent should.
    """

    def __init__(self, tools: list[Any] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, obj: Any) -> None:
        t = as_tool(obj)
        if t.name in self._tools:
            raise ToolError(f"duplicate tool name: {t.name!r}")
        self._tools[t.name] = t

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[ToolSchema]:
        return [t.schema() for t in self._tools.values()]

    async def invoke(self, tool_use_id: str, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Run a tool by name and wrap the outcome in a :class:`ToolResult`."""
        t = self._tools.get(name)
        if t is None:
            return ToolResult(
                tool_use_id=tool_use_id,
                content=f"No such tool {name!r}. Available: {', '.join(self.names()) or 'none'}.",
                is_error=True,
            )
        try:
            result = await t.call(arguments)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the model
            return ToolResult(tool_use_id=tool_use_id, content=f"{type(exc).__name__}: {exc}", is_error=True)
        return ToolResult(tool_use_id=tool_use_id, content=_stringify(result), is_error=False)


def _stringify(value: Any) -> str:
    """Render a tool's return value as the string the model will read.

    Strings pass through, ``None`` becomes empty, containers are JSON-encoded
    (so the model sees structured data), and anything else falls back to
    ``str()`` — no surprising quotes around a plain object.
    """
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        import json

        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            return str(value)
    return str(value)
