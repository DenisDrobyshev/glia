"""Exception hierarchy for glia.

Everything that can go wrong raises a subclass of :class:`GliaError`, so callers
can catch the whole family with one ``except`` or narrow to a specific failure.
Nothing here is magic — these are plain exceptions you can read and reason about.
"""

from __future__ import annotations


class GliaError(Exception):
    """Base class for every error raised by glia."""


class ToolError(GliaError):
    """A tool could not be found, called, or completed successfully."""


class GuardrailTripped(GliaError):
    """An input or output guardrail rejected the content.

    Carries the offending value and the guardrail name so a caller can decide
    whether to retry, surface the message, or abort.
    """

    def __init__(self, guardrail: str, message: str, value: object | None = None) -> None:
        super().__init__(f"guardrail {guardrail!r} tripped: {message}")
        self.guardrail = guardrail
        self.message = message
        self.value = value


class MaxStepsExceeded(GliaError):
    """The agent loop hit ``max_steps`` without the model ending its turn."""

    def __init__(self, steps: int) -> None:
        super().__init__(
            f"agent did not finish within {steps} steps "
            f"(raise max_steps, or inspect the trajectory to see why it kept going)"
        )
        self.steps = steps


class ProviderError(GliaError):
    """An LLM provider adapter failed (bad request, transport error, refusal)."""


class StructuredOutputError(GliaError):
    """The model's output could not be coerced into the requested schema."""
