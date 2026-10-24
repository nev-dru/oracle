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


async def test_unexecutable_binary_is_an_error_not_a_crash(tmp_path: Path) -> None:
    script = tmp_path / "noexec"
    script.write_text("#!/bin/sh\necho {}\n")
    script.chmod(0o644)
    tool = script_tool(tmp_path, "", op={"argv": [str(script)]})
    result = await executor_for(tool, tmp_path).run("t.run", {})
    assert result.status == "error" and "cannot run" in (result.error or "")


async def test_binary_output_is_spilled_with_a_hex_head(tmp_path: Path) -> None:
    tool = script_tool(tmp_path, "import sys\nsys.stdout.buffer.write(bytes(range(256)) * 8)\n")
    result = await executor_for(tool, tmp_path).run("t.run", {})
    assert result.ok and result.truncated
    assert result.records[0]["binary"] is True and len(result.records[0]["head_hex"]) == 2048
    assert result.spill is not None and result.spill.startswith("file:")
    assert result.pointers[0] == result.spill


async def test_fake_jira_honours_operators(home_cfg, fake_registry) -> None:
    ex = Executor(fake_registry, home_cfg.tool_env(), home_cfg.spill_dir)
    open_mine = await ex.run(
        "acli.jira.search", {"jql": "assignee = currentUser() AND status != Done", "limit": 50}
    )
    assert open_mine.ok and open_mine.ids
    assert all(r["status"] != "Done" for r in open_mine.records)
    assert all(r["assignee"] == "Andrew Neville" for r in open_mine.records)


async def test_large_binary_output_spills_once(tmp_path: Path) -> None:
    body = "import sys\nsys.stdout.buffer.write(bytes(range(256)) * 400)\n"
    tool = script_tool(tmp_path, body, limits={"max_output_kb": 4})
    result = await executor_for(tool, tmp_path).run("t.run", {})
    assert result.ok and result.pointers[0] == result.spill
    assert len(list((tmp_path / "spill").iterdir())) == 1


LATER_PAGE_BINARY = """
import json, sys
if "--start" in sys.argv:
    sys.stdout.buffer.write(bytes([255, 254]) * 10)
else:
    print(json.dumps({"items": [{"id": "a"}], "next": 1}))
"""


async def test_binary_on_a_later_page_is_an_error(tmp_path: Path) -> None:
    tool = script_tool(
        tmp_path,
        LATER_PAGE_BINARY,
        op={
            "argv": [sys.executable, str(tmp_path / "tool.py"), "{?cursor:--start,cursor}"],
            "paginate": {"param": "cursor", "next": "$.next"},
        },
    )
    result = await executor_for(tool, tmp_path).run("t.run", {})
    assert result.status == "error" and "page 2 was not UTF-8" in (result.error or "")


async def test_fake_jira_refuses_clauses_it_cannot_check(home_cfg, fake_registry) -> None:
    ex = Executor(fake_registry, home_cfg.tool_env(), home_cfg.spill_dir)
    result = await ex.run("acli.jira.search", {"jql": "project = PLAT AND reporter was andrew"})
    assert result.status == "error" and "unsupported clause" in (result.error or "")
    quoted = await ex.run("acli.jira.search", {"jql": 'text ~ "order by" AND status = Done'})
    assert quoted.ok and quoted.records == []


async def test_spill_head_is_redacted_on_expand(tmp_path: Path) -> None:
    from oracle.exec.pointers import expand_spill

    body = (
        "import json\n"
        "items = [{'id': i, 'env': 'password=hunter2 ' + 'x' * 300} for i in range(20)]\n"
        "print(json.dumps({'items': items}))\n"
    )
    tool = script_tool(tmp_path, body, limits={"max_output_kb": 2})
    result = await executor_for(tool, tmp_path).run("t.run", {})
    assert result.spill and "hunter2" not in str(result.records)
    shown = expand_spill(result.spill, tmp_path / "spill")
    assert "hunter2" not in shown["head"] and "[REDACTED]" in shown["head"]
