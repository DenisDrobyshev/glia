"""Human-in-the-loop tool approval — an inspectable gate before a tool runs.

An approval policy is any callable ``(ApprovalRequest) -> Decision`` (sync or
async; a bare ``bool`` is accepted too). The agent consults it for every tool
call, emits :class:`~glia.trajectory.ApprovalRequested` /
:class:`~glia.trajectory.ApprovalResolved` events around the decision, and turns
a denial into an error :class:`~glia.types.ToolResult` the model can react to.
Denied tools never execute.

Because the policy is just a function, "human in the loop" is literally a
function that prompts a human — or an allowlist, or an LLM judge. A few common
policies ship here; the interactive one (:func:`prompt_in_terminal`) is a
reference, not the only way.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """What the policy is asked to rule on: a single pending tool call."""

    step: int
    tool_use_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Decision:
    """The policy's verdict. ``reason`` is fed back to the model on a denial so
    it can adapt (e.g. try a safer approach)."""

    allow: bool
    reason: str = ""


ApprovalPolicy = Callable[[ApprovalRequest], Decision | bool | Awaitable[Decision | bool]]
"""Return ``True``/``Decision(True)`` to allow, ``False``/``Decision(False, reason)`` to deny."""


async def resolve(policy: ApprovalPolicy, request: ApprovalRequest) -> Decision:
    """Call a policy and normalise its result to a :class:`Decision`."""
    outcome = policy(request)
    if inspect.isawaitable(outcome):
        outcome = await outcome
    if isinstance(outcome, bool):
        return Decision(allow=outcome, reason="" if outcome else "denied by policy")
    return outcome


# -- built-in policies ---------------------------------------------------------


def approve_all(request: ApprovalRequest) -> Decision:
    """Allow every tool. (The default when no policy is set is also 'allow all',
    but with no approval events emitted.)"""
    return Decision(True)


def deny_all(request: ApprovalRequest) -> Decision:
    return Decision(False, "denied by policy")


def allow_only(*names: str) -> ApprovalPolicy:
    """Allow only the named tools; deny the rest."""
    allowed = set(names)

    def policy(request: ApprovalRequest) -> Decision:
        if request.name in allowed:
            return Decision(True)
        return Decision(False, f"tool {request.name!r} is not on the allowlist")

    return policy


def deny(*names: str) -> ApprovalPolicy:
    """Deny the named tools; allow the rest."""
    blocked = set(names)

    def policy(request: ApprovalRequest) -> Decision:
        if request.name in blocked:
            return Decision(False, f"tool {request.name!r} is blocked by policy")
        return Decision(True)

    return policy


def prompt_in_terminal(request: ApprovalRequest) -> Decision:  # pragma: no cover - interactive
    """A reference interactive policy: ask a human on the terminal.

    Not used in tests (it blocks on ``input``). Shows the shape of a real
    human-in-the-loop gate — swap in your own UI prompt.
    """
    answer = input(f"Approve tool {request.name}({request.arguments})? [y/N] ").strip().lower()
    if answer in ("y", "yes"):
        return Decision(True)
    return Decision(False, "denied by human reviewer")
