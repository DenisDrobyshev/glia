from __future__ import annotations

from dataclasses import dataclass

import pytest
from conftest import run

from glia import generate_structured
from glia.errors import StructuredOutputError
from glia.providers import EchoLLM, call
from glia.structured import _dataclass_schema, _ensure_object, _resolve


@dataclass
class D:
    a: int
    b: str = "x"


def test_ensure_object_fills_defaults():
    out = _ensure_object({"properties": {"x": {"type": "string"}}})
    assert out["type"] == "object"
    assert out["additionalProperties"] is False


def test_dict_schema_passthrough_coerce():
    schema, coerce = _resolve({"type": "object", "properties": {"x": {"type": "string"}}})
    assert coerce({"x": "v"}) == {"x": "v"}


def test_dataclass_schema_required_excludes_defaults():
    schema = _dataclass_schema(D)
    assert schema["required"] == ["a"]  # b has a default
    assert schema["properties"]["a"] == {"type": "integer"}


def test_dataclass_coerce():
    _, coerce = _resolve(D)
    assert coerce({"a": 1, "b": "y"}) == D(1, "y")


def test_generate_structured_forces_and_returns_dataclass():
    llm = EchoLLM([call("respond", {"a": 7, "b": "z"})])
    person = run(generate_structured(llm, "?", D))
    assert person == D(7, "z")
    assert llm.calls[0].tool_choice == "respond"


def test_unsupported_schema_type_raises():
    with pytest.raises(StructuredOutputError):
        _resolve(int)


def test_no_tool_call_raises():
    llm = EchoLLM(["just prose, no structured call"])
    with pytest.raises(StructuredOutputError):
        run(generate_structured(llm, "?", {"type": "object", "properties": {}}))


def test_bad_arguments_raise_structured_error():
    llm = EchoLLM([call("respond", {"unexpected": 1})])
    with pytest.raises(StructuredOutputError):
        run(generate_structured(llm, "?", D))


def test_pydantic_path_if_available():
    pydantic = pytest.importorskip("pydantic")

    class M(pydantic.BaseModel):
        name: str
        n: int

    _, coerce = _resolve(M)
    obj = coerce({"name": "a", "n": 3})
    assert obj.name == "a" and obj.n == 3
