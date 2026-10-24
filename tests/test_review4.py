"""Regression tests for the fourth code review: one per finding, safety first."""

import os
import stat
import time
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from oracle import cli
from oracle.ask import AskRequest, Models, ask
from oracle.config import REPO_ROOT, Config, load_config
from oracle.exec.executor import Executor, OpResult
from oracle.exec.runners.mcp import McpError, _cause
from oracle.log.report import build_report
from oracle.log.store import CallRecord, open_store
from oracle.plan.distill import refine_problems
from oracle.registry.importers.cli_draft import op_problems
from oracle.registry.importers.infer import infer_returns
from oracle.registry.importers.mcp_import import McpTool
from oracle.registry.loader import Registry, load_registry
from oracle.registry.params import ParamError, validate_params
from oracle.registry.schema import McpServer, Op, Param, Tool
from oracle.skills.loader import load_skills, select
from oracle.workflows.ingest import IngestError, _empty, _start, activate, staging_dir
from tests.conftest import DONE, REGISTRY, echo_distiller, plan, scripted


def marker_tool(tmp_path: Path, argv_tail: list[str], params: dict) -> tuple[Tool, Path]:
    marker = tmp_path / "RAN"
    script = tmp_path / "kubectl"
    script.write_text(f"#!/bin/sh\necho ran > {marker}\necho '{{}}'\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    tool = Tool.model_validate(
        {"name": "k", "description": "d",
         "ops": [{"name": "run", "when": "w", "argv": [str(script), *argv_tail], "params": params}]}
    )  # fmt: skip
    return tool, marker


@pytest.mark.parametrize(
    ("tail", "params", "values"),
    [
        (
            ["app", "{action}", "{name}"],
            {"action": {}, "name": {}},
            {"action": "sync", "name": "b"},
        ),
        (["{?confirm:delete}", "pods"], {"confirm": {"type": "bool"}}, {"confirm": True}),
    ],
)
async def test_rendered_command_is_checked_before_it_runs(
    tmp_path: Path, tail: list[str], params: dict, values: dict
) -> None:
    tool, marker = marker_tool(tmp_path, tail, params)
    ex = Executor(Registry(tools={"k": tool}), dict(os.environ), tmp_path / "spill")
    result = await ex.run("k.run", values)
    assert result.status == "invalid" and "read-only" in (result.error or "")
    assert not marker.exists()


def test_honestly_declared_mutating_ops_are_dropped_not_fatal(tmp_path: Path) -> None:
    data = {
        "name": "jira",
        "description": "d",
        "ops": [
            {"name": "search", "when": "w", "argv": ["acli", "jira", "search"]},
            {
                "name": "transition",
                "when": "w",
                "argv": ["acli", "jira", "transition"],
                "mutating": True,
            },
        ],
    }
    (tmp_path / "jira.yaml").write_text(yaml.safe_dump(data))
    reg = load_registry(tmp_path)
    assert [op.name for op in reg.tools["jira"].ops] == ["search"] and reg.problems == []


def test_each_import_path_drafts_one_transport() -> None:
    example = [{"params": {}}]
    mcp_op = Op.model_validate(
        {"name": "ls", "when": "w", "mcp_tool": "list_files", "examples": example}
    )
    cli_op = Op.model_validate(
        {"name": "ls", "when": "w", "argv": ["python", "-c", "x"], "examples": example}
    )
    assert op_problems(mcp_op, "kubectl")  # a CLI draft cannot become an MCP server
    assert op_problems(cli_op, "python", mcp=True)  # an MCP repair cannot switch to argv
    assert not op_problems(mcp_op, "python", mcp=True)


def test_server_hints_cannot_lift_a_name_refusal() -> None:
    assert McpTool("clear_cache", "", {}, read_only=True).refused
    assert McpTool("search_notes", "", {}, read_only=False).refused


def test_mcp_exception_groups_report_the_real_cause() -> None:
    server = McpServer(command="srv")
    assert isinstance(_cause(BaseExceptionGroup("x", [TimeoutError()]), server), TimeoutError)
    nested = BaseExceptionGroup("x", [BaseExceptionGroup("y", [RuntimeError("boom")])])
    err = _cause(nested, server)
    assert isinstance(err, McpError) and "boom" in str(err)


def test_a_malformed_skill_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "SKILL.md").write_text("---\nname: bad\nno closing fence\n")
    (tmp_path / "good").mkdir()
    (tmp_path / "good" / "SKILL.md").write_text(
        "---\nname: good\napplies_to: [planner]\n---\nbody\n"
    )
    assert [s.name for s in load_skills(tmp_path)] == ["good"]


def test_secrets_in_identifiers_refused_prose_allowed() -> None:
    op = Op(name="get", when="w", argv=["k", "get", "{r}"], params={"r": Param(required=True)})
    for value in ("secret/db-creds", "secrets,configmaps", "Secret.v1"):
        with pytest.raises(ParamError, match="secrets"):
            validate_params(op, {"r": value})
    assert validate_params(op, {"r": "where do we store secrets"})


def test_code_operators_are_refused_in_json_params() -> None:
    op = Op(name="find", when="w", argv=["mongo", "find", "{f}"], params={"f": Param(type="json")})
    with pytest.raises(ParamError, match=r"\$where"):
        validate_params(op, {"f": {"$or": [{"$where": "sleep(1)"}]}})
    assert validate_params(op, {"f": '{"account_id": 42}'})["f"] == {"account_id": 42}


def test_int_enums_and_floats_validate() -> None:
    op = Op(
        name="q", when="w", argv=["t", "{n}", "{s}"],
        params={"n": Param(type="int", enum=["1", "2"]), "s": Param(type="float")},
    )  # fmt: skip
    assert validate_params(op, {"n": 2, "s": "0.75"}) == {"n": 2, "s": 0.75}


def test_staging_is_cleared_and_existing_files_refused(home_cfg: Config, tmp_path: Path) -> None:
    cfg = load_config({"ORACLE_HOME": str(tmp_path), "ORACLE_REGISTRY": str(tmp_path / "reg")})
    (tmp_path / "reg").mkdir()
    staged = staging_dir(cfg, "fresh")
    staged.mkdir(parents=True)
    (staged / "manifest.yaml").write_text("stale")
    _start("fresh", cfg, Registry())
    assert not (staged / "manifest.yaml").exists()
    (tmp_path / "reg" / "broken.yaml").write_text("not: [valid")
    with pytest.raises(IngestError, match="already has a manifest"):
        _start("broken", cfg, Registry())
    with pytest.raises(IngestError, match="valid tool name"):
        _start("My_Tool", cfg, Registry())


def test_activation_repeats_the_loader_checks(tmp_path: Path) -> None:
    cfg = load_config({"ORACLE_HOME": str(tmp_path), "ORACLE_REGISTRY": str(tmp_path / "reg")})
    staged = staging_dir(cfg, "edited")
    staged.mkdir(parents=True)
    bad = {
        "name": "edited",
        "description": "d",
        "ops": [{"name": "x", "when": "w", "argv": ["k", "delete", "all"]}],
    }
    (staged / "manifest.yaml").write_text(yaml.safe_dump(bad))
    with pytest.raises(IngestError, match="not read-only"):
        activate(cfg, "edited")
    (staged / "manifest.yaml").write_text("name: [broken")
    with pytest.raises(IngestError, match="not valid"):
        activate(cfg, "edited")


def test_inference_skips_fields_some_records_leave_empty() -> None:
    records = [
        {"id": "a", "title": "", "status": "open"},
        {"id": "b", "title": "t", "status": "done"},
    ]
    returns = infer_returns({"items": records}, "t")
    assert returns is not None and "title" not in returns.summary_fields
    assert _empty([]) and _empty({"items": []}) and not _empty({"items": [1]})


async def test_report_ignores_breadth_and_recipe_stages(tmp_path: Path) -> None:
    async with open_store(tmp_path / "log.sqlite") as store:
        ranked = [{"tool": "workiq", "score": 1}]
        await store.record_call(
            CallRecord("c1", "s", "today", None, "standard", ranked, None, wall_ms=1, breadth=True)
        )
        await store.record_call(
            CallRecord("c2", "s", "who", None, "standard", ranked, None, wall_ms=1)
        )
        hit = OpResult(op="acli.jira.search", params={})
        hit.records, hit.pointers = [{"key": "A"}], ["jira:A"]
        await store.record_op("c1", "s", "s1", hit)
        await store.record_op("c2", "s", "s1/tickets", hit)
        report = await build_report(store.db, time.time() - 60)
    assert report.ranker_misses == []


async def test_showing_a_pointer_again_keeps_its_expanded_mark(tmp_path: Path) -> None:
    hit = OpResult(op="acli.jira.get", params={})
    hit.records, hit.pointers = [{"key": "A"}], ["jira:A"]
    async with open_store(tmp_path / "log.sqlite") as store:
        await store.record_call(CallRecord("c1", "s", "q", None, "quick", [], None, wall_ms=1))
        await store.record_op("c1", "s", "s1", hit)
        await store.lookup("jira:A", "s")
        await store.record_op("c1", "s", "s2", hit)
        async with store.db.execute(
            "SELECT expanded_at FROM pointers WHERE pointer = 'jira:A'"
        ) as cur:
            assert (await cur.fetchone())[0] is not None  # type: ignore[index]


def test_mcp_import_needs_a_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ORACLE_HOME", str(tmp_path))
    result = CliRunner().invoke(cli.app, ["tool", "add", "--mcp", "uvx some-server"])
    assert result.exit_code == 2 and "needs --name" in result.output


async def test_quick_ignores_breadth_with_a_note(home_cfg: Config, fake_registry: Registry) -> None:
    step = {"id": "s1", "op": "workiq.calendar.today"}
    m = Models(scripted(plan(step)), scripted(DONE), echo_distiller())
    answer = await ask(
        AskRequest("meetings", level="quick", breadth=True), home_cfg, fake_registry, m
    )
    assert any("ignored at quick" in n for n in answer.notes) and not answer.breadth


def test_refine_scope_is_checked_when_quoted() -> None:
    assert refine_problems(['oracle "x" --scope "cluster"'], {"kubectl"})


def test_every_shortlisted_tool_gets_a_skill_before_any_gets_two() -> None:
    skills = load_skills(REPO_ROOT / "skills")
    chosen = [
        s.name
        for s in select(skills, "planner", ["acli", "es", "kubectl"], load_registry(REGISTRY), "x")
    ]
    by_tool = {s.name: s.tool for s in skills}
    assert {by_tool[n] for n in chosen[:2]} == {"acli", "es"}  # one per tool, not jql and cql
