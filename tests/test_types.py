from __future__ import annotations

import pytest

from glia.types import (
    Message,
    Text,
    Thinking,
    ToolResult,
    ToolUse,
    Usage,
    _block_from_dict,
    _block_to_dict,
    assistant,
    user,
)


def test_message_accessors():
    msg = Message(
        "assistant",
        [Thinking("hmm"), Text("Hello "), ToolUse("t1", "f", {"x": 1}), Text("world")],
    )
    assert msg.text() == "Hello world"
    assert msg.thinking() == "hmm"
    assert [t.name for t in msg.tool_uses()] == ["f"]


@pytest.mark.parametrize(
    "block",
    [
        Text("hi"),
        Thinking("because"),
        ToolUse("t1", "add", {"a": 1, "b": 2}),
        ToolResult("t1", "3", is_error=True),
    ],
)
def test_block_dict_round_trip(block):
    assert _block_from_dict(_block_to_dict(block)) == block


def test_message_dict_round_trip():
    msg = Message("user", [Text("q"), ToolResult("t1", "r")])
    assert Message.from_dict(msg.to_dict()) == msg


def test_unknown_block_serialisation_raises():
    with pytest.raises(TypeError):
        _block_to_dict(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        _block_from_dict({"type": "nope"})


def test_usage_add_and_to_dict():
    a = Usage(1, 2, 3, 4)
    b = Usage(10, 20, 30, 40)
    total = a + b
    assert (total.input_tokens, total.output_tokens) == (11, 22)
    assert total.to_dict() == {
        "input_tokens": 11,
        "output_tokens": 22,
        "cache_read_tokens": 33,
        "cache_write_tokens": 44,
    }


def test_convenience_constructors():
    assert user("hi") == Message("user", [Text("hi")])
    assert assistant("yo") == Message("assistant", [Text("yo")])


def test_blocks_are_immutable():
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        Text("x").text = "y"  # frozen dataclass
