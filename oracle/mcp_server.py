"""The oracle as an MCP server: one `oracle` tool (and `expand`) for clients that prefer MCP to
Bash. A thin wrapper that runs the CLI with `--format json`; nothing in the core depends on it.

Run: `oracle-mcp` (needs the `mcp` extra). Each call is a fresh CLI process, so the server holds
no state and a slow call never blocks the next.
"""

import asyncio
import contextlib
import json
import sys

CLI = [sys.executable, "-m", "oracle"]
TIMEOUT_S = 180.0


async def run_cli(args: list[str]) -> str:
    """One CLI run. stdin is closed (the server's stdin is the protocol stream), and the child
    is killed if the call times out or the client cancels it."""
    proc = await asyncio.create_subprocess_exec(
        *CLI,
        *args,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), TIMEOUT_S)
    except TimeoutError:
        return json.dumps({"error": f"oracle gave no answer in {TIMEOUT_S:g} s"})
    finally:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
    if proc.returncode:
        return json.dumps({"error": err.decode("utf-8", "replace").strip()})
    return out.decode("utf-8", "replace")


def ask_args(
    question: str,
    intent: str | None,
    level: str,
    scope: str | None,
    breadth: bool,
    session: str | None,
) -> list[str]:
    args = ["ask", "--format", "json", "--level", level]
    args += ["--intent", intent] if intent else []
    args += ["--scope", scope] if scope else []
    args += ["--session", session] if session else []
    args += ["--breadth"] if breadth else []
    return [*args, "--", question]  # a question that starts with '-' is still the question


def build_server():  # type: ignore[no-untyped-def]
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("oracle", instructions=__doc__)

    @server.tool()
    async def oracle(
        question: str,
        intent: str | None = None,
        level: str = "standard",
        scope: str | None = None,
        breadth: bool = False,
        session: str | None = None,
    ) -> str:
        """Find things across every registered source (code, tickets, docs, chat, cluster,
        logs). Ask in plain language and say why. Returns findings with pointers, coverage
        (what was and was not searched) and ready-to-run refinements."""
        return await run_cli(ask_args(question, intent, level, scope, breadth, session))

    @server.tool()
    async def expand(pointer: str, session: str | None = None) -> str:
        """The full record behind a pointer from an earlier oracle answer."""
        return await run_cli(["expand", pointer, *(["--session", session] if session else [])])

    return server


def main() -> None:
    build_server().run("stdio")


if __name__ == "__main__":
    main()
