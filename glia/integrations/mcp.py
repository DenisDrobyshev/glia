"""MCP bridge: use any Model Context Protocol server's tools as glia tools.

The heart of it is a tiny, pure adapter — :func:`tools_from_mcp` turns MCP tool
definitions + an open session into ordinary :class:`~glia.tools.Tool` objects.
That part has no dependencies and is fully tested.

Two convenience connectors open a real session for you (they need the optional
``mcp`` package: ``pip install "glia-agents[mcp]"``):

    from glia import Agent
    from glia.integrations.mcp import mcp_stdio_tools

    async with mcp_stdio_tools("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]) as tools:
        agent = Agent(llm, tools=tools)
        await agent.run("List the files in /tmp")

The session stays open for the duration of the ``async with`` block, so the
tools are callable while the agent runs.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from ..errors import ProviderError, ToolError
from ..tools import Tool


def tools_from_mcp(session: Any, tool_defs: list[Any]) -> list[Tool]:
    """Adapt MCP tool definitions to glia tools bound to an open ``session``.

    ``session`` needs an async ``call_tool(name, arguments)`` method; each entry
    in ``tool_defs`` needs ``name``, ``description``, and ``inputSchema``
    attributes (the shape the ``mcp`` package returns). Pure and dependency-free.
    """
    return [_wrap(session, td) for td in tool_defs]


def _wrap(session: Any, td: Any) -> Tool:
    raw_name = getattr(td, "name", None) or (td.get("name") if isinstance(td, dict) else None)
    name: str = raw_name or "mcp_tool"
    description: str = (getattr(td, "description", None) or (td.get("description") if isinstance(td, dict) else "")) or ""
    schema = getattr(td, "inputSchema", None)
    if schema is None and isinstance(td, dict):
        schema = td.get("inputSchema")
    parameters = schema or {"type": "object", "properties": {}}

    async def call(**kwargs: Any) -> str:
        result = await session.call_tool(name, kwargs)
        return _render_result(result)

    call.__name__ = name or "mcp_tool"
    call.__doc__ = description
    return Tool(name=name, description=description.strip(), parameters=parameters, func=call, is_async=True)


def _render_result(result: Any) -> str:
    """Flatten an MCP ``CallToolResult`` into text; raise on tool errors so the
    registry records an error result the model can see."""
    parts: list[str] = []
    for item in getattr(result, "content", None) or []:
        text = getattr(item, "text", None)
        parts.append(text if text is not None else str(item))
    rendered = "\n".join(parts)
    if getattr(result, "isError", False):
        raise ToolError(rendered or "MCP tool reported an error")
    return rendered


def _import_mcp():
    try:
        import mcp  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ProviderError(
            "the 'mcp' package is required for MCP connectors — "
            "install it with: pip install 'glia-agents[mcp]'"
        ) from exc


@asynccontextmanager
async def mcp_stdio_tools(command: str, args: list[str] | None = None, env: dict | None = None):
    """Open an MCP server over stdio and yield its tools as glia tools."""
    _import_mcp()
    from mcp import ClientSession, StdioServerParameters  # pragma: no cover - needs mcp
    from mcp.client.stdio import stdio_client  # pragma: no cover

    params = StdioServerParameters(command=command, args=args or [], env=env)  # pragma: no cover
    async with stdio_client(params) as (read, write):  # pragma: no cover
        async with ClientSession(read, write) as session:  # pragma: no cover
            await session.initialize()
            listed = await session.list_tools()
            yield tools_from_mcp(session, listed.tools)


@asynccontextmanager
async def mcp_http_tools(url: str, headers: dict | None = None):
    """Open an MCP server over Streamable HTTP and yield its tools as glia tools."""
    _import_mcp()
    from mcp import ClientSession  # pragma: no cover - needs mcp
    from mcp.client.streamable_http import streamablehttp_client  # pragma: no cover

    async with streamablehttp_client(url, headers=headers) as (read, write, _):  # pragma: no cover
        async with ClientSession(read, write) as session:  # pragma: no cover
            await session.initialize()
            listed = await session.list_tools()
            yield tools_from_mcp(session, listed.tools)
