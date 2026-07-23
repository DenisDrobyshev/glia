"""Tests for the MCP bridge adapter, using a fake session (no mcp package)."""

from __future__ import annotations

from conftest import run

from glia import Agent, ToolRegistry
from glia.integrations.mcp import tools_from_mcp
from glia.providers import EchoLLM
from glia.providers import call as tcall


class FakeContent:
    def __init__(self, text):
        self.text = text


class FakeResult:
    def __init__(self, content, isError=False):
        self.content = content
        self.isError = isError


class FakeToolDef:
    def __init__(self, name, description, inputSchema):
        self.name = name
        self.description = description
        self.inputSchema = inputSchema


class FakeSession:
    def __init__(self, results):
        self.results = results
        self.calls = []

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        return self.results[name]


def test_tools_from_mcp_carries_name_desc_schema():
    schema = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    tools = tools_from_mcp(FakeSession({}), [FakeToolDef("search", "Search things", schema)])
    t = tools[0]
    assert t.name == "search"
    assert t.description == "Search things"
    assert t.parameters == schema
    assert t.schema().parameters == schema  # wire-level view matches


def test_calling_an_mcp_tool_invokes_the_session():
    session = FakeSession({"m": FakeResult([FakeContent("a"), FakeContent("b")])})
    tools = tools_from_mcp(session, [FakeToolDef("m", "", {"type": "object"})])
    out = run(tools[0].call({"x": 1}))
    assert out == "a\nb"
    assert session.calls == [("m", {"x": 1})]


def test_error_result_is_marked_as_error():
    session = FakeSession({"bad": FakeResult([FakeContent("boom")], isError=True)})
    reg = ToolRegistry(tools_from_mcp(session, [FakeToolDef("bad", "", {"type": "object"})]))
    result = run(reg.invoke("id", "bad", {}))
    assert result.is_error and "boom" in result.content


def test_dict_shaped_tool_def():
    tools = tools_from_mcp(FakeSession({}), [{"name": "x", "description": "d", "inputSchema": {"type": "object"}}])
    assert tools[0].name == "x" and tools[0].description == "d"


def test_mcp_tools_run_inside_an_agent():
    session = FakeSession({"echo": FakeResult([FakeContent("pong")])})
    defs = [FakeToolDef("echo", "Echo tool", {"type": "object", "properties": {"msg": {"type": "string"}}})]
    agent = Agent(EchoLLM([tcall("echo", {"msg": "hi"}), "done"]), tools=tools_from_mcp(session, defs))
    result = run(agent.run("use echo"))
    assert result.output == "done"
    assert session.calls == [("echo", {"msg": "hi"})]
    assert result.trajectory.events_of("tool_returned")[0].content == "pong"  # type: ignore[attr-defined]
