"""Context engineering: keep the working context small without losing the plot.

The finite context window is the scarcest resource a long-running agent has.
glia exposes compaction as an explicit, swappable primitive rather than baking
it in: a :class:`Compactor` decides *when* the trajectory is too big and *how*
to shrink it. The agent calls it once per step; you can watch it work via the
:class:`~glia.trajectory.Compacted` event.

Two implementations ship here:

* :class:`SummarizingCompactor` — folds old turns into an LLM-written summary,
  preserving provenance (this is "compaction").
* :class:`TrimmingCompactor` — drops the oldest turns outright (this is
  "context editing"). Cheaper, lossy, no model call.

Both keep the most recent turns verbatim so in-flight tool calls stay intact.
"""

from __future__ import annotations

from typing import Protocol

from .llm import LLMRequest
from .types import Message, Text, ToolResult, ToolUse, user


class Compactor(Protocol):
    """Decides whether and how to compact a trajectory."""

    def should_compact(self, trajectory: TrajectoryLike) -> bool:  # pragma: no cover - protocol
        ...

    async def compact(self, trajectory: TrajectoryLike, llm) -> str:  # pragma: no cover - protocol
        """Mutate the trajectory in place and return a short summary string
        describing what was folded away (for the event log)."""
        ...


class TrajectoryLike(Protocol):  # pragma: no cover - structural typing helper
    system: str | None
    messages: list[Message]


def _keeps_message_pairs_intact(messages: list[Message], keep_last: int) -> int:
    """Return a cut index that never splits an assistant tool_use from the user
    turn carrying its results — splitting those is an API error waiting to
    happen. We back the cut up until it lands on a clean boundary."""
    cut = max(0, len(messages) - keep_last)
    # If the message right before the cut is an assistant turn with tool calls,
    # its results live in the message at `cut`; move the cut back one so both
    # the calls and their results are compacted together.
    while cut > 0 and messages[cut - 1].role == "assistant" and messages[cut - 1].tool_uses():
        cut -= 1
    return cut


class TrimmingCompactor:
    """Drop the oldest turns once the message count crosses a threshold.

    No model call, no summary — purely mechanical. Good when old context is
    genuinely disposable (stateless tool chatter) and latency matters.
    """

    def __init__(self, *, max_messages: int = 40, keep_last: int = 20) -> None:
        self.max_messages = max_messages
        self.keep_last = keep_last

    def should_compact(self, trajectory: TrajectoryLike) -> bool:
        return len(trajectory.messages) > self.max_messages

    async def compact(self, trajectory: TrajectoryLike, llm) -> str:
        cut = _keeps_message_pairs_intact(trajectory.messages, self.keep_last)
        if cut <= 0:
            return ""
        dropped = cut
        trajectory.messages = trajectory.messages[cut:]
        return f"trimmed {dropped} oldest message(s)"


class SummarizingCompactor:
    """Summarise the oldest turns into a single note, keeping recent turns raw.

    Provenance-preserving: the summary replaces the dropped turns as a user
    message, so the model still knows what happened earlier — just compressed.
    """

    _PROMPT = (
        "Summarise the earlier part of this agent conversation into a compact "
        "note the assistant can rely on to keep working. Preserve concrete "
        "facts, decisions, tool results, and open threads. Be terse; omit "
        "pleasantries. Write only the note."
    )

    def __init__(
        self,
        *,
        max_messages: int = 40,
        keep_last: int = 12,
        max_tokens: int = 1024,
    ) -> None:
        self.max_messages = max_messages
        self.keep_last = keep_last
        self.max_tokens = max_tokens

    def should_compact(self, trajectory: TrajectoryLike) -> bool:
        return len(trajectory.messages) > self.max_messages

    async def compact(self, trajectory: TrajectoryLike, llm) -> str:
        cut = _keeps_message_pairs_intact(trajectory.messages, self.keep_last)
        if cut <= 0:
            return ""
        old = trajectory.messages[:cut]
        transcript = _render(old)
        request = LLMRequest(
            messages=[user(f"{self._PROMPT}\n\n<transcript>\n{transcript}\n</transcript>")],
            max_tokens=self.max_tokens,
        )
        response = await llm.generate(request)
        summary = response.message.text().strip() or "(summary unavailable)"
        note = Message(
            role="user",
            blocks=[Text(f"[Earlier conversation, compacted]\n{summary}")],
        )
        trajectory.messages = [note, *trajectory.messages[cut:]]
        return summary


def _render(messages: list[Message]) -> str:
    """A plain-text rendering of messages for the summariser to read."""
    lines: list[str] = []
    for message in messages:
        for block in message.blocks:
            if isinstance(block, Text):
                lines.append(f"{message.role}: {block.text}")
            elif isinstance(block, ToolUse):
                lines.append(f"{message.role}: [calls {block.name}({block.input})]")
            elif isinstance(block, ToolResult):
                flag = " (error)" if block.is_error else ""
                lines.append(f"tool{flag}: {block.content}")
    return "\n".join(lines)
