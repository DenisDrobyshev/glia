"""OpenTelemetry exporter — turn the glia event stream into spans.

Drop :class:`OTelExporter` into an agent's ``hooks`` and every run becomes a
trace: a root ``glia.run`` span with child ``glia.model_call`` and ``glia.tool``
spans, carrying model/tool/token/stop-reason attributes and marking tool errors.
Approvals and compaction land as span events.

    from glia import Agent
    from glia.integrations.otel import OTelExporter

    agent = Agent(llm, tools=[...], hooks=[OTelExporter()])   # uses the global tracer

The exporter only needs an OTel *tracer* (``pip install "glia-agents[otel]"``, then
configure a TracerProvider + exporter as usual). It's a plain hook — synchronous,
no async — and it never raises into your run.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..trajectory import Event


class OTelExporter:
    """A glia hook that emits OpenTelemetry spans from agent events."""

    def __init__(
        self,
        tracer: Any = None,
        *,
        service_name: str = "glia",
        context_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        if tracer is None:
            from opentelemetry import trace  # lazy: only needed for the default tracer

            tracer = trace.get_tracer(service_name)
        self.tracer = tracer
        self._ctx = context_factory or _default_context_factory()

        self._run: Any = None
        self._run_ctx: Any = None
        self._model: dict[int, Any] = {}
        self._tool: dict[str, Any] = {}

    def __call__(self, event: Event) -> None:
        try:
            self._handle(event)
        except Exception:  # noqa: BLE001 - telemetry must never break a run
            pass

    def _handle(self, event: Event) -> None:
        kind = event.kind
        if kind == "run_started":
            self._run = self.tracer.start_span("glia.run")
            self._run.set_attribute("glia.prompt", getattr(event, "prompt", None) or "")
            self._run_ctx = self._ctx(self._run)

        elif kind == "model_call":
            span = self.tracer.start_span("glia.model_call", context=self._run_ctx)
            span.set_attribute("glia.step", getattr(event, "step", 0))
            span.set_attribute("glia.tool_count", getattr(event, "tool_count", 0))
            self._model[getattr(event, "step", 0)] = span

        elif kind == "model_response":
            span = self._model.pop(getattr(event, "step", 0), None)
            if span is not None:
                usage = getattr(event, "usage", None)
                span.set_attribute("glia.stop_reason", getattr(event, "stop_reason", ""))
                if usage is not None:
                    span.set_attribute("glia.input_tokens", usage.input_tokens)
                    span.set_attribute("glia.output_tokens", usage.output_tokens)
                tool_uses = getattr(event, "tool_uses", ())
                if tool_uses:
                    span.set_attribute("glia.tool_uses", list(tool_uses))
                span.end()

        elif kind == "tool_called":
            span = self.tracer.start_span("glia.tool", context=self._run_ctx)
            span.set_attribute("glia.tool.name", getattr(event, "name", ""))
            self._tool[getattr(event, "tool_use_id", "")] = span

        elif kind == "tool_returned":
            span = self._tool.pop(getattr(event, "tool_use_id", ""), None)
            if span is not None:
                span.set_attribute("glia.tool.name", getattr(event, "name", ""))
                span.set_attribute("glia.tool.duration_s", getattr(event, "duration_s", 0.0))
                if getattr(event, "is_error", False):
                    _set_error(span, getattr(event, "content", ""))
                span.end()

        elif kind == "approval_resolved" and self._run is not None:
            self._run.add_event(
                "approval",
                {"tool": getattr(event, "name", ""), "allowed": getattr(event, "allowed", True)},
            )

        elif kind == "compacted" and self._run is not None:
            self._run.add_event("compacted", {"freed_messages": getattr(event, "freed_messages", 0)})

        elif kind == "run_finished" and self._run is not None:
            self._run.set_attribute("glia.stop_reason", getattr(event, "stop_reason", ""))
            self._run.set_attribute("glia.steps", getattr(event, "step", 0))
            self._run.end()
            self._run = None


def _default_context_factory() -> Callable[[Any], Any]:
    try:
        from opentelemetry import trace

        return trace.set_span_in_context
    except ImportError:  # pragma: no cover - only when otel isn't installed
        return lambda span: span


def _set_error(span: Any, message: str) -> None:
    try:
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.ERROR, message))
    except Exception:  # noqa: BLE001 - Status types unavailable; attribute is enough
        pass
    span.set_attribute("glia.error", True)
