"""The script runner end to end with a fake handler: the real child process, no tools, no models."""

import ast
import asyncio
from pathlib import Path
from typing import Any

import pytest

from oracle.deep.runner import CallRefused, ScriptLimits, run_script
from oracle.registry.codegen import build_library, ident, sources
from oracle.registry.loader import Registry


@pytest.fixture
def lib(tmp_path: Path, fake_registry: Registry) -> Path:
    return build_library(fake_registry, tmp_path).path


class Recorder:
    def __init__(self, delay: float = 0.0) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.delay = delay
        self.active = self.peak = 0

    async def __call__(self, method: str, args: dict[str, Any]) -> Any:
        self.calls.append((method, args))
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(self.delay)
        self.active -= 1
        if args["params"].get("key") == "BAD":
            raise CallRefused("param 'key' is not a ticket key")
        return [{"key": args["params"].get("key", "x"), "_pointer": "jira:OPS-1"}]


async def test_calls_reports_and_print(lib: Path) -> None:
    handler = Recorder()
    code = """
from oracle_tools import acli, report
import json
tickets = await acli.jira_get(key="OPS-1206")
print("got", len(tickets))
report({"tickets": tickets, "dumped": json.dumps(tickets[0])})
"""
    result = await run_script(code, handler, lib)
    assert result.ok, result.error
    assert handler.calls == [("op", {"op": "acli.jira.get", "params": {"key": "OPS-1206"}})]
    assert result.reports[0]["tickets"][0]["_pointer"] == "jira:OPS-1"
    assert "got 1" in result.printed and result.calls == 1


async def test_gather_runs_calls_concurrently(lib: Path) -> None:
    handler = Recorder(delay=0.2)
    code = """
from oracle_tools import acli, gather, report
keys = ["A-1", "A-2", "A-3", "A-4"]
found = await gather(*(acli.jira_get(key=k) for k in keys))
report([f[0]["key"] for f in found])
"""
    result = await run_script(code, handler, lib)
    assert result.ok, result.error
    assert result.reports == [["A-1", "A-2", "A-3", "A-4"]] and handler.peak == 4


async def test_refused_call_raises_in_the_script(lib: Path) -> None:
    code = """
from oracle_tools import acli, report
try:
    await acli.jira_get(key="BAD")
except ValueError as e:
    report(str(e))
"""
    result = await run_script(code, Recorder(), lib)
    assert result.reports == ["param 'key' is not a ticket key"]


async def test_traceback_points_at_the_script(lib: Path) -> None:
    result = await run_script("x = 1\ny = {}['missing']\n", Recorder(), lib)
    assert result.error and '"<script>", line 2' in result.error and "KeyError" in result.error


async def test_wall_clock_stops_a_loop(lib: Path) -> None:
    code = "n = 0\nwhile True:\n    n += 1\n"
    result = await run_script(code, Recorder(), lib, ScriptLimits(wall_s=1.0, cpu_s=30))
    assert result.error and "past 1 s" in result.error


async def test_cpu_limit_kills_a_busy_child(lib: Path) -> None:
    code = "n = 0\nwhile True:\n    n += 1\n"
    result = await run_script(code, Recorder(), lib, ScriptLimits(wall_s=20, cpu_s=1))
    assert result.error and "CPU limit" in result.error


async def test_call_cap(lib: Path) -> None:
    code = """
from oracle_tools import acli, report
n = 0
for i in range(5):
    try:
        await acli.jira_get(key=str(i))
        n += 1
    except ValueError as e:
        report(str(e))
report(n)
"""
    result = await run_script(code, Recorder(), lib, ScriptLimits(max_calls=3))
    assert result.reports[-1] == 3 and "more than 3 tool calls" in result.reports[0]


async def test_static_check_refuses_before_running(lib: Path) -> None:
    handler = Recorder()
    result = await run_script("import os\nos.system('true')", handler, lib)
    assert result.refused and not handler.calls and not result.ok


RUNTIME_WALLS = {
    # each passes the static check; the child's own restrictions stop it
    "submodule through a facade": "import json\njson.decoder",
    "dotted import": "import collections.abc",
    "builtins outside the safe set": "exit()",
    "files through a module's imports": "import json\njson.codecs.open('/tmp/x', 'w')",
}


async def test_runtime_walls(lib: Path) -> None:
    out = {}
    for name, code in RUNTIME_WALLS.items():
        out[name] = await run_script(code, Recorder(), lib)
    assert "AttributeError" in (out["submodule through a facade"].error or "")
    assert "not allowed" in (out["dotted import"].error or "")
    assert "NameError" in (out["builtins outside the safe set"].error or "")
    assert "AttributeError" in (out["files through a module's imports"].error or "")


async def test_printed_output_is_capped(lib: Path) -> None:
    code = "for i in range(20000):\n    print('x' * 50)\n"
    result = await run_script(code, Recorder(), lib, ScriptLimits(printed_kb=4))
    assert result.ok and len(result.printed) == 4096


def test_stubs_compile_and_match_manifests(fake_registry: Registry) -> None:
    files, problems = sources(fake_registry)
    assert not problems
    for rel, text in files.items():
        ast.parse(text, rel)
    tree = ast.parse(files["oracle_tools/acli.py"])
    functions = {n.name: n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)}
    tool = fake_registry.tools["acli"]
    assert set(functions) == {ident(op.name) for op in tool.ops}
    for op in tool.ops:
        args = {a.arg for a in functions[ident(op.name)].args.kwonlyargs}
        assert args == {ident(p) for p in op.params}


def test_library_is_keyed_by_its_sources(tmp_path: Path, fake_registry: Registry) -> None:
    first = build_library(fake_registry, tmp_path)
    again = build_library(fake_registry, tmp_path)
    assert first.path == again.path and (first.path / "oracle_tools" / "acli.py").exists()
    assert "async def jira_get" in first.stubs["acli"]


async def test_a_closed_channel_fails_the_pending_call(lib: Path) -> None:
    """When the oracle's end of the pipe closes mid-call, the script's await fails at once
    instead of waiting for the wall clock."""
    import json
    import sys

    from oracle.deep.runner import CHILD
    from oracle.deep.sandbox import ALLOWED_IMPORTS, SAFE_BUILTINS

    code = "from oracle_tools import acli\nawait acli.jira_get(key='OPS-1')\n"
    hello = {"code": code, "allowed": sorted(ALLOWED_IMPORTS), "builtins": sorted(SAFE_BUILTINS),
             "cpu_s": 5, "memory_mb": 256}  # fmt: skip
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-I", "-S", "-B", str(CHILD), str(lib),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL, env={},
    )  # fmt: skip
    assert proc.stdin and proc.stdout
    proc.stdin.write(json.dumps(hello).encode() + b"\n")
    await proc.stdin.drain()
    assert json.loads(await proc.stdout.readline())["t"] == "call"
    proc.stdin.close()
    last = json.loads(await asyncio.wait_for(proc.stdout.readline(), 5))
    await proc.wait()
    assert last["t"] == "error" and "closed the channel" in last["error"]


def test_a_bad_manifest_leaves_out_its_tool_not_the_package(
    tmp_path: Path, fake_registry: Registry
) -> None:
    import os

    acli = fake_registry.tools["acli"]
    get = acli.ops[0]
    params = dict(get.params) | {"a-b": get.params[next(iter(get.params))]}
    params["a_b"] = params["a-b"]
    clashing = acli.model_copy(update={"ops": [get.model_copy(update={"params": params})]})
    tools = dict(fake_registry.tools) | {
        "acli": clashing,
        "report": acli.model_copy(update={"name": "report"}),
    }
    registry = Registry(tools=tools)
    library = build_library(registry, tmp_path)
    assert any("params become the same name a_b" in p for p in library.problems)
    assert any(p.startswith("report: its module name report") for p in library.problems)
    assert "report" not in library.stubs and "acli" in library.stubs
    stale = tmp_path / "lib" / "0123456789ab"
    stale.mkdir()
    os.utime(stale, (0, 0))
    build_library(fake_registry, tmp_path)
    assert not stale.exists()  # an unused library is removed after a week
