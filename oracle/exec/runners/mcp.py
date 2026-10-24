"""Call one tool on a stdio MCP server and return its result as text for `parse_output`.

One server process per call: simple and isolated; a cached session is an optimisation for
later. The `mcp` package is an optional extra, imported only when an MCP op runs.
"""

import json
import os
from typing import Any, Literal

from oracle.registry.schema import McpServer


class McpError(RuntimeError):
    def __init__(self, status: Literal["error", "not_configured"], message: str) -> None:
        super().__init__(message)
        self.status: Literal["error", "not_configured"] = status


async def call_mcp(
    server: McpServer, tool: str, arguments: dict[str, Any], env: dict[str, str], timeout: float
) -> str:
    try:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except ImportError:  # pragma: no cover - depends on the install
        raise McpError("not_configured", "MCP tools need the extra: uv sync --extra mcp") from None
    params = StdioServerParameters(command=server.command, args=server.args, env=env)
    with open(os.devnull, "w") as quiet:
        try:
            async with (
                stdio_client(params, errlog=quiet) as (read, write),
                ClientSession(read, write, read_timeout_seconds=timeout) as session,
            ):
                await session.initialize()
                result: Any = await session.call_tool(tool, arguments, read_timeout_seconds=timeout)
        except FileNotFoundError:
            raise McpError("not_configured", f"{server.command!r} is not on PATH") from None
        except BaseExceptionGroup as group:  # anyio task groups wrap the real failure
            raise _cause(group, server) from None
    return _text(result)


def _cause(group: BaseExceptionGroup, server: McpServer) -> Exception:
    """The first real error inside a (possibly nested) exception group, as an McpError or a
    TimeoutError the executor already reports cleanly."""
    leaves: list[BaseException] = []
    stack: list[BaseException] = [group]
    while stack:
        exc = stack.pop()
        if isinstance(exc, BaseExceptionGroup):
            stack.extend(exc.exceptions)
        else:
            leaves.append(exc)
    for exc in leaves:
        if isinstance(exc, TimeoutError):
            return exc
        if isinstance(exc, FileNotFoundError):
            return McpError("not_configured", f"{server.command!r} is not on PATH")
    first = leaves[0] if leaves else group
    return McpError("error", f"{type(first).__name__}: {first}")


def _text(result: Any) -> str:
    if getattr(result, "is_error", False):
        raise McpError("error", _join(result) or "the MCP tool reported an error")
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict) and set(structured) == {"result"}:
        inner = structured["result"]  # SDKs wrap a tool's return value; unwrap it
        return inner if isinstance(inner, str) else json.dumps(inner)
    return json.dumps(structured) if structured is not None else _join(result)


def _join(result: Any) -> str:
    return "\n".join(getattr(c, "text", "") for c in getattr(result, "content", []) or [])
