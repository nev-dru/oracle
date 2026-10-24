import os
import sys
from pathlib import Path

from oracle.exec.executor import Executor
from oracle.registry.loader import Registry
from tests.conftest import script_tool

PAGED = """
import json, sys
start = int(sys.argv[sys.argv.index("--start") + 1]) if "--start" in sys.argv else 0
items = [{"id": f"i{n}", "name": f"item {n}"} for n in range(5)]
nxt = start + 2 if start + 2 < len(items) else None
print(json.dumps({"items": items[start:start + 2], "next": nxt}))
"""


def executor_for(tool, tmp_path: Path, env: dict[str, str] | None = None) -> Executor:
    reg = Registry(tools={tool.name: tool})
    return Executor(reg, env if env is not None else dict(os.environ), tmp_path / "spill")


async def test_pagination_collects_every_page(tmp_path: Path) -> None:
    tool = script_tool(
        tmp_path,
        PAGED,
        op={
            "argv": [sys.executable, str(tmp_path / "tool.py"), "{?cursor:--start,cursor}"],
            "paginate": {"param": "cursor", "next": "$.next", "max_pages": 5},
            "returns": {"path": "$.items[*]", "id_field": "id", "pointer": "t:{id}"},
        },
    )
    result = await executor_for(tool, tmp_path).run("t.run", {})
    assert result.ok, result.error
    assert result.pages == 3
    assert result.ids == ["i0", "i1", "i2", "i3", "i4"]
    assert result.pointers[0] == "t:i0"


async def test_nonzero_exit_reports_stderr(tmp_path: Path) -> None:
    tool = script_tool(tmp_path, "import sys\nsys.stderr.write('bad jql')\nsys.exit(3)\n")
    result = await executor_for(tool, tmp_path).run("t.run", {})
    assert result.status == "error"
    assert "exit 3" in (result.error or "") and "bad jql" in (result.error or "")


async def test_timeout_kills_the_process(tmp_path: Path) -> None:
    tool = script_tool(tmp_path, "import time\ntime.sleep(5)\n", limits={"timeout_s": 0.3})
    result = await executor_for(tool, tmp_path).run("t.run", {})
    assert result.status == "error" and "timed out" in (result.error or "")


async def test_missing_binary_and_env_are_not_configured(tmp_path: Path) -> None:
    tool = script_tool(tmp_path, "", op={"argv": ["definitely-not-a-binary-xyz"]})
    assert (await executor_for(tool, tmp_path).run("t.run", {})).status == "not_configured"
    tool = script_tool(tmp_path, "print('{}')", auth={"env": ["NEEDED_TOKEN"]})
    result = await executor_for(tool, tmp_path, env={"PATH": os.environ["PATH"]}).run("t.run", {})
    assert result.status == "not_configured" and "NEEDED_TOKEN" in (result.error or "")


async def test_invalid_params_never_run(tmp_path: Path) -> None:
    tool = script_tool(tmp_path, "raise SystemExit('should not run')")
    result = await executor_for(tool, tmp_path).run("t.run", {"surprise": 1})
    assert result.status == "invalid"
    assert (await executor_for(tool, tmp_path).run("nope.run", {})).status == "invalid"


async def test_oversized_output_spills_to_file(tmp_path: Path) -> None:
    body = (
        "import json\n"
        "print(json.dumps({'items': [{'id': i, 'blob': 'x' * 400} for i in range(20)]}))\n"
    )
    tool = script_tool(tmp_path, body, limits={"max_output_kb": 2})
    result = await executor_for(tool, tmp_path).run("t.run", {})
    assert result.ok and result.truncated
    assert result.spill and Path(result.spill.removeprefix("file:")).stat().st_size > 2048
    assert 0 < len(result.records) < 20
    assert len(result.ids) == len(result.records) == len(result.pointers)


async def test_non_json_output_is_an_error(tmp_path: Path) -> None:
    tool = script_tool(tmp_path, "print('not json')")
    result = await executor_for(tool, tmp_path).run("t.run", {})
    assert result.status == "error" and "not JSON" in (result.error or "")


async def test_fake_tools_through_home_profile(home_cfg, fake_registry) -> None:
    ex = Executor(fake_registry, home_cfg.tool_env(), home_cfg.spill_dir)
    jira = await ex.run("acli.jira.search", {"jql": 'text ~ "ownership transfer"', "limit": 2})
    assert jira.ok and jira.pages >= 2
    assert all(p.startswith("jira:") for p in jira.pointers)
    people = await ex.run("workiq.people.lookup", {"name": "andrew neville"})
    assert [r["display_name"] for r in people.records] == ["Andrew Neville"]
