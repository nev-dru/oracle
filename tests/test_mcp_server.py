"""The oracle's own MCP server, called over real MCP stdio."""

import json
import os
import sys

from oracle.config import REPO_ROOT
from oracle.mcp_server import ask_args


def test_ask_args_pass_only_what_is_set() -> None:
    assert ask_args("q", None, "quick", None, False, None) == [
        "ask", "--format", "json", "--level", "quick", "--", "q",
    ]  # fmt: skip
    args = ask_args("-1d deploys", "why", "standard", "acli", True, "s1")
    assert args[-2:] == ["--", "-1d deploys"] and "--breadth" in args and "--scope" in args


async def test_server_lists_tools_and_expands_over_stdio(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    env = {**os.environ, "ORACLE_HOME": str(tmp_path), "ORACLE_PROFILE": "home"}
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "oracle.mcp_server"], env=env, cwd=str(REPO_ROOT)
    )
    with open(os.devnull, "w") as quiet:
        async with (
            stdio_client(params, errlog=quiet) as (r, w),
            ClientSession(r, w, read_timeout_seconds=60) as s,
        ):
            await s.initialize()
            names = {t.name for t in (await s.list_tools()).tools}
            result = await s.call_tool("expand", {"pointer": "jira:NOPE-1"})
    assert names == {"oracle", "expand"}
    structured = getattr(result, "structured_content", None)
    text = structured["result"] if structured else getattr(result.content[0], "text", "")
    assert "no record for jira:NOPE-1" in json.loads(text)["error"]
