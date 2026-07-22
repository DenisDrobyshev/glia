from __future__ import annotations

from conftest import run

from glia import Agent, tool
from glia.approval import allow_only, deny
from glia.llm import LLMResponse
from glia.providers import EchoLLM, call
from glia.types import Message, Text

# --- streaming ----------------------------------------------------------------


def test_streaming_emits_deltas_that_reconstruct_the_answer():
    agent = Agent(EchoLLM(["The quick brown fox."]), stream=True)

    async def collect():
        deltas, output = [], None
        async for event in agent.run_events("go"):
            if event.kind == "model_delta":
                deltas.append(event.text)  # type: ignore[attr-defined]
            elif event.kind == "run_finished":
                output = event.output  # type: ignore[attr-defined]
        return deltas, output

    deltas, output = run(collect())
    assert len(deltas) > 1  # streamed in pieces
    assert "".join(deltas) == "The quick brown fox."
    assert output == "The quick brown fox."


def test_streaming_falls_back_to_generate_when_provider_cannot_stream():
    class GenerateOnly:
        async def generate(self, request):
            return LLMResponse(message=Message("assistant", [Text("no stream here")]), stop_reason="end_turn")

    agent = Agent(GenerateOnly(), stream=True)  # stream requested, provider can't
    result = run(agent.run("hi"))
    assert result.output == "no stream here"
    assert result.trajectory.events_of("model_delta") == []  # no deltas emitted


# --- parallel tool execution --------------------------------------------------


def test_parallel_tools_all_run_and_results_stay_in_call_order():
    order: list[str] = []

    @tool
    async def slow(label: str) -> str:
        """Record a label."""
        order.append(label)
        return f"did-{label}"

    llm = EchoLLM(
        [
            [call("slow", {"label": "a"}, id="1"), call("slow", {"label": "b"}, id="2")],
            "both done",
        ]
    )
    agent = Agent(llm, tools=[slow], parallel_tools=True)
    result = run(agent.run("run both"))

    assert result.output == "both done"
    # Both tools executed...
    assert set(order) == {"a", "b"}
    # ...and results are emitted in the original call order.
    returned = result.trajectory.events_of("tool_returned")
    assert [e.content for e in returned] == ["did-a", "did-b"]  # type: ignore[attr-defined]


# --- human-in-the-loop approval -----------------------------------------------


def test_approval_denies_tool_which_never_executes():
    executed: list[str] = []

    @tool
    async def danger(x: str) -> str:
        """A tool we want to gate."""
        executed.append(x)
        return "ran"

    llm = EchoLLM([call("danger", {"x": "boom"}), "understood, I won't do that"])
    agent = Agent(llm, tools=[danger], approval=deny("danger"))
    result = run(agent.run("please do the dangerous thing"))

    # The tool was gated and never ran.
    assert executed == []
    # The model saw a denial result it could react to.
    returned = result.trajectory.events_of("tool_returned")[0]
    assert returned.is_error is True  # type: ignore[attr-defined]
    assert "not approved" in returned.content.lower()  # type: ignore[attr-defined]
    # And the decision is visible in the stream.
    resolved = result.trajectory.events_of("approval_resolved")[0]
    assert resolved.allowed is False  # type: ignore[attr-defined]


def test_approval_allows_listed_tool():
    @tool
    async def safe(x: str) -> str:
        """An allowed tool."""
        return f"ok-{x}"

    llm = EchoLLM([call("safe", {"x": "hi"}), "done"])
    agent = Agent(llm, tools=[safe], approval=allow_only("safe"))
    result = run(agent.run("use the safe tool"))

    returned = result.trajectory.events_of("tool_returned")[0]
    assert returned.is_error is False  # type: ignore[attr-defined]
    assert returned.content == "ok-hi"  # type: ignore[attr-defined]
    assert result.trajectory.events_of("approval_requested")  # gate was consulted


def test_async_approval_policy_supported():
    @tool
    async def t(x: str) -> str:
        """tool."""
        return x

    async def policy(request):
        return request.name == "t"  # bool return is normalised to a Decision

    llm = EchoLLM([call("t", {"x": "v"}), "ok"])
    agent = Agent(llm, tools=[t], approval=policy)
    result = run(agent.run("go"))
    assert result.trajectory.events_of("tool_returned")[0].is_error is False  # type: ignore[attr-defined]
