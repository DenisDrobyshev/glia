"""Core data types: content blocks, messages, and token usage.

These are the atoms every other module works with. They are deliberately small,
immutable, and JSON-serialisable — a whole conversation is just a list of
:class:`Message`, and every message is a list of typed blocks you can log,
diff, snapshot, and replay. There is no hidden state anywhere in here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "assistant"]
"""Only two roles cross the wire. Tool results are carried as ``user`` messages
(the Anthropic convention); the system prompt lives on the trajectory, not in
the message list."""


@dataclass(frozen=True, slots=True)
class Text:
    """A plain text block."""

    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True, slots=True)
class Thinking:
    """A reasoning block returned by the model (may be empty when omitted)."""

    text: str
    type: Literal["thinking"] = "thinking"


@dataclass(frozen=True, slots=True)
class ToolUse:
    """The model's request to call a tool, with parsed arguments."""

    id: str
    name: str
    input: dict[str, Any]
    type: Literal["tool_use"] = "tool_use"


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The result of a tool call, sent back to the model."""

    tool_use_id: str
    content: str
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


Block = Text | Thinking | ToolUse | ToolResult
"""The closed set of things that can appear inside a message. A union, not a
base class — pattern-match or ``isinstance`` on it; there are no subclasses to
discover."""

_BLOCK_TYPES: dict[str, type] = {
    "text": Text,
    "thinking": Thinking,
    "tool_use": ToolUse,
    "tool_result": ToolResult,
}


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting for one model call. Add them up over a run to see cost."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
        }


@dataclass(frozen=True, slots=True)
class Message:
    """One turn in the conversation: a role plus an ordered list of blocks."""

    role: Role
    blocks: list[Block] = field(default_factory=list)

    # -- convenience accessors -------------------------------------------------

    def text(self) -> str:
        """Concatenate every :class:`Text` block (the human-readable answer)."""
        return "".join(b.text for b in self.blocks if isinstance(b, Text))

    def tool_uses(self) -> list[ToolUse]:
        """Every tool call the model requested in this message."""
        return [b for b in self.blocks if isinstance(b, ToolUse)]

    def thinking(self) -> str:
        return "".join(b.text for b in self.blocks if isinstance(b, Thinking))

    # -- serialisation ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "blocks": [_block_to_dict(b) for b in self.blocks]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(role=data["role"], blocks=[_block_from_dict(b) for b in data["blocks"]])


# -- module-level helpers so blocks stay pure dataclasses ----------------------


def _block_to_dict(block: Block) -> dict[str, Any]:
    # dataclasses.asdict would work, but this keeps the field order explicit and
    # avoids copying nested tool-input dicts through asdict's deepcopy.
    if isinstance(block, Text):
        return {"type": "text", "text": block.text}
    if isinstance(block, Thinking):
        return {"type": "thinking", "text": block.text}
    if isinstance(block, ToolUse):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if isinstance(block, ToolResult):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
            "is_error": block.is_error,
        }
    raise TypeError(f"unknown block type: {block!r}")


def _block_from_dict(data: dict[str, Any]) -> Block:
    kind = data.get("type")
    if kind == "text":
        return Text(data["text"])
    if kind == "thinking":
        return Thinking(data["text"])
    if kind == "tool_use":
        return ToolUse(id=data["id"], name=data["name"], input=data["input"])
    if kind == "tool_result":
        return ToolResult(
            tool_use_id=data["tool_use_id"],
            content=data["content"],
            is_error=data.get("is_error", False),
        )
    raise ValueError(f"unknown block type: {kind!r}")


def user(text: str) -> Message:
    """Build a user message from a string. A convenience, nothing more."""
    return Message(role="user", blocks=[Text(text)])


def assistant(text: str) -> Message:
    """Build an assistant message from a string."""
    return Message(role="assistant", blocks=[Text(text)])
