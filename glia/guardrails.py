"""Guardrails: small, composable validators that run on the way in and out.

A guardrail is any callable ``(text) -> None`` that raises
:class:`~glia.errors.GuardrailTripped` to reject. That's the whole interface —
so a regex check, a length cap, a PII scrubber, or an LLM-judge are all just
functions. The agent runs input guardrails on each user prompt and output
guardrails on each final answer.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from .errors import GuardrailTripped

Guardrail = Callable[[str], None]
"""Raise :class:`GuardrailTripped` to reject; return ``None`` to pass."""


def run_guardrails(guardrails: Iterable[Guardrail], text: str) -> None:
    """Run each guardrail in order; the first to raise stops the chain."""
    for guard in guardrails:
        guard(text)


def max_length(limit: int, *, name: str = "max_length") -> Guardrail:
    """Reject text longer than ``limit`` characters."""

    def guard(text: str) -> None:
        if len(text) > limit:
            raise GuardrailTripped(name, f"length {len(text)} exceeds limit {limit}", text)

    return guard


def block_pattern(pattern: str | re.Pattern[str], *, name: str = "block_pattern") -> Guardrail:
    """Reject text that matches a regular expression."""
    compiled = re.compile(pattern)

    def guard(text: str) -> None:
        if compiled.search(text):
            raise GuardrailTripped(name, f"matched forbidden pattern {compiled.pattern!r}", text)

    return guard


def require_pattern(pattern: str | re.Pattern[str], *, name: str = "require_pattern") -> Guardrail:
    """Reject text that does *not* match a regular expression."""
    compiled = re.compile(pattern)

    def guard(text: str) -> None:
        if not compiled.search(text):
            raise GuardrailTripped(name, f"missing required pattern {compiled.pattern!r}", text)

    return guard


# A deliberately conservative set of secret-shaped patterns. Extend for your
# threat model — this is a starting point, not a compliance product.
_SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{36}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]


def no_secrets(*, name: str = "no_secrets") -> Guardrail:
    """Reject text that appears to contain an API key or private key."""

    def guard(text: str) -> None:
        for pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                raise GuardrailTripped(name, "text appears to contain a secret/credential", text)

    return guard
