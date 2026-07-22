"""The trajectory: the single source of truth for a run.

A :class:`Trajectory` holds the system prompt, the full message list, an
append-only log of :class:`Event`\\s, and accumulated token usage. It is a plain
data structure — no behaviour hidden inside it — and it round-trips to JSON, so
"snapshot the agent and resume later" is just ``to_dict`` / ``from_dict``.

Every meaningful thing the agent does emits an event: a model call, a model
response, a tool call, a tool result, a compaction. Subscribe to the stream and
you can see the entire glass box working, live.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .types import Message, ToolResult, ToolUse, Usage, _block_from_dict, _block_to_dict

# --- Events -------------------------------------------------------------------
# Events are records, not commands. The agent emits them; hooks observe them.


@dataclass(frozen=True, slots=True)
class Event:
    """Base event. ``kind`` names the type; ``at`` is a wall-clock timestamp."""

    kind: str
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "at": self.at, **self._payload()}

    def _payload(self) -> dict[str, Any]:
        return {}


@dataclass(frozen=True, slots=True)
class RunStarted(Event):
    prompt: str | None = None
    kind: str = "run_started"

    def _payload(self) -> dict[str, Any]:
        return {"prompt": self.prompt}


@dataclass(frozen=True, slots=True)
class ModelCall(Event):
    step: int = 0
    message_count: int = 0
    tool_count: int = 0
    kind: str = "model_call"

    def _payload(self) -> dict[str, Any]:
        return {"step": self.step, "message_count": self.message_count, "tool_count": self.tool_count}


@dataclass(frozen=True, slots=True)
class ModelResponse(Event):
    step: int = 0
    text: str = ""
    stop_reason: str = ""
    tool_uses: tuple[str, ...] = ()
    usage: Usage = field(default_factory=Usage)
    kind: str = "model_response"

    def _payload(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "text": self.text,
            "stop_reason": self.stop_reason,
            "tool_uses": list(self.tool_uses),
            "usage": self.usage.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ToolCalled(Event):
    step: int = 0
    tool_use_id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    kind: str = "tool_called"

    def _payload(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "tool_use_id": self.tool_use_id,
            "name": self.name,
            "arguments": self.arguments,
        }


@dataclass(frozen=True, slots=True)
class ToolReturned(Event):
    step: int = 0
    tool_use_id: str = ""
    name: str = ""
    content: str = ""
    is_error: bool = False
    duration_s: float = 0.0
    kind: str = "tool_returned"

    def _payload(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "tool_use_id": self.tool_use_id,
            "name": self.name,
            "content": self.content,
            "is_error": self.is_error,
            "duration_s": self.duration_s,
        }


@dataclass(frozen=True, slots=True)
class Compacted(Event):
    freed_messages: int = 0
    summary: str = ""
    kind: str = "compacted"

    def _payload(self) -> dict[str, Any]:
        return {"freed_messages": self.freed_messages, "summary": self.summary}


@dataclass(frozen=True, slots=True)
class RunFinished(Event):
    step: int = 0
    output: str = ""
    stop_reason: str = ""
    kind: str = "run_finished"

    def _payload(self) -> dict[str, Any]:
        return {"step": self.step, "output": self.output, "stop_reason": self.stop_reason}


# --- Trajectory ---------------------------------------------------------------


@dataclass
class Trajectory:
    """The complete, inspectable state of one agent run."""

    system: str | None = None
    messages: list[Message] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    @classmethod
    def new(cls, system: str | None = None) -> Trajectory:
        return cls(system=system)

    # -- mutation (each returns nothing; the trajectory is the state) ----------

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    def add_user(self, text: str) -> None:
        from .types import user

        self.messages.append(user(text))

    def add_tool_results(self, results: Iterable[ToolResult]) -> None:
        """Append tool results as a single user turn (the provider convention)."""
        self.messages.append(Message(role="user", blocks=list(results)))

    def record(self, event: Event) -> None:
        self.events.append(event)

    def add_usage(self, usage: Usage) -> None:
        self.usage = self.usage + usage

    # -- inspection ------------------------------------------------------------

    def last_assistant_text(self) -> str:
        for message in reversed(self.messages):
            if message.role == "assistant":
                return message.text()
        return ""

    def pending_tool_uses(self) -> list[ToolUse]:
        """Tool calls in the final assistant message that have no result yet."""
        if not self.messages or self.messages[-1].role != "assistant":
            return []
        return self.messages[-1].tool_uses()

    def events_of(self, kind: str) -> list[Event]:
        return [e for e in self.events if e.kind == kind]

    # -- serialisation ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready snapshot. Messages are preserved exactly; events are
        recorded as dicts (they are a log, not replayed on load)."""
        return {
            "version": 1,
            "system": self.system,
            "messages": [m.to_dict() for m in self.messages],
            "usage": self.usage.to_dict(),
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trajectory:
        traj = cls(
            system=data.get("system"),
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            usage=Usage(**data.get("usage", {})),
        )
        # Events reload as generic records — enough to inspect a resumed run's
        # history without needing every concrete class.
        for raw in data.get("events", []):
            traj.events.append(_LoadedEvent(raw))
        return traj


class _LoadedEvent(Event):
    """A previously-serialised event, rehydrated as an opaque record."""

    __slots__ = ("_raw",)

    def __init__(self, raw: dict[str, Any]) -> None:
        object.__setattr__(self, "kind", raw.get("kind", "event"))
        object.__setattr__(self, "at", raw.get("at", 0.0))
        object.__setattr__(self, "_raw", raw)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._raw)


__all__ = [
    "Event",
    "RunStarted",
    "ModelCall",
    "ModelResponse",
    "ToolCalled",
    "ToolReturned",
    "Compacted",
    "RunFinished",
    "Trajectory",
]

# Re-exported so callers can round-trip blocks without importing types directly.
_ = (_block_to_dict, _block_from_dict)
