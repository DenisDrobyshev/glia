from __future__ import annotations

from typing import Annotated, Literal

from conftest import run

from glia import ToolRegistry, tool
from glia.tools import as_tool


@tool
async def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool(name="lookup", description="Look up a city.")
def city_lookup(
    city: Annotated[str, "City name"],
    unit: Literal["c", "f"] = "c",
    note: str | None = None,  # PEP 604 union must flatten to "string" too
) -> str:
    return f"{city}/{unit}/{note}"


def test_schema_from_type_hints():
    t = as_tool(add)
    assert t.name == "add"
    assert t.description == "Add two integers."
    assert t.parameters["properties"]["a"] == {"type": "integer"}
    assert t.parameters["required"] == ["a", "b"]
    assert t.parameters["additionalProperties"] is False


def test_annotated_literal_and_optional():
    t = as_tool(city_lookup)
    props = t.parameters["properties"]
    assert props["city"] == {"type": "string", "description": "City name"}
    assert props["unit"] == {"type": "string", "enum": ["c", "f"]}
    assert props["note"] == {"type": "string"}  # Optional[str] flattens to str
    # Only the parameter without a default is required.
    assert t.parameters["required"] == ["city"]


def test_registry_runs_sync_and_async():
    reg = ToolRegistry([add, city_lookup])
    assert set(reg.names()) == {"add", "lookup"}

    ok = run(reg.invoke("id1", "add", {"a": 2, "b": 3}))
    assert ok.content == "5"
    assert ok.is_error is False

    sync = run(reg.invoke("id2", "lookup", {"city": "Paris"}))
    assert sync.content == "Paris/c/None"


def test_registry_missing_tool_is_error_not_exception():
    reg = ToolRegistry([add])
    result = run(reg.invoke("id", "nope", {}))
    assert result.is_error is True
    assert "No such tool" in result.content


def test_tool_exception_becomes_error_result():
    @tool
    def boom() -> str:
        raise ValueError("kaboom")

    reg = ToolRegistry([boom])
    result = run(reg.invoke("id", "boom", {}))
    assert result.is_error is True
    assert "kaboom" in result.content


def test_duplicate_tool_name_rejected():
    import pytest

    from glia.errors import ToolError

    with pytest.raises(ToolError):
        ToolRegistry([add, add])
