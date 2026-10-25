"""Workflows as ops, and the recursion guard: depth, cycles and the tier clamp."""

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from oracle.ask import AskRequest, Models, ask
from oracle.config import REPO_ROOT, Config, load_config
from oracle.exec.executor import Executor
from oracle.log.store import open_store
from oracle.registry.loader import Registry, load_registry
from oracle.workflows.protocol import GuardError, Trace, clamp
from oracle.workflows.runner import WorkflowHost
from tests.conftest import DONE, REGISTRY, echo_distiller, plan, scripted

TOOLS = ["acli", "workiq", "kubectl", "argocd", "es"]


@pytest.fixture
def wf_cfg(tmp_path: Path) -> Config:
    tools = tmp_path / "registry" / "tools"
    tools.mkdir(parents=True)
    for name in TOOLS:
        shutil.copy(REGISTRY / f"{name}.yaml", tools)
    fakes = REPO_ROOT / "tests" / "fakes" / "workflows"
    shutil.copy(fakes / "incident.yaml", tools)
    shutil.copytree(
        fakes, tmp_path / "registry" / "workflows", ignore=shutil.ignore_patterns("*.yaml")
    )
    env = {"ORACLE_HOME": str(tmp_path / "home"), "ORACLE_PROFILE": "home",
           "ORACLE_REGISTRY": str(tools)}  # fmt: skip
    return load_config(env)


@pytest.fixture
def wf_registry(wf_cfg: Config) -> Registry:
    reg = load_registry(wf_cfg.registry_dir)
    assert not reg.problems, reg.problems
    return reg


def executor(cfg: Config, registry: Registry, trace: Trace | None = None) -> Executor:
    host = WorkflowHost(cfg, registry, trace or Trace.root("deep"))
    return Executor(registry, cfg.tool_env(), cfg.spill_dir, host)


async def test_workflow_gathers_from_several_tools(wf_cfg: Config, wf_registry: Registry) -> None:
    result = await executor(wf_cfg, wf_registry).run(
        "incident.investigate", {"service": "billing-api"}
    )
    assert result.ok, result.error
    sources = {r["source"] for r in result.records}
    assert {"argocd.app.history", "kubectl.events"} & sources
    assert all(not p.startswith("incident:") for p in result.pointers)  # the tools' own pointers


async def test_a_workflow_that_calls_itself_is_refused(
    wf_cfg: Config, wf_registry: Registry
) -> None:
    result = await executor(wf_cfg, wf_registry).run("incident.again", {"service": "x"})
    assert result.ok and "would repeat incident.again" in result.records[0]["text"]


async def test_depth_is_capped(wf_cfg: Config, wf_registry: Registry) -> None:
    deep = Trace("t", depth=3, chain=(("oracle", "ask", "a"),) * 0, level="deep")
    result = await executor(wf_cfg, wf_registry, deep).run("incident.investigate", {"service": "a"})
    assert result.status == "invalid" and "nesting deeper than 3" in (result.error or "")


async def test_nested_ask_runs_one_tier_lower(wf_cfg: Config, wf_registry: Registry) -> None:
    """outer standard call -> incident.nested -> inner oracle call at quick, one trace."""
    step = {"id": "s1", "op": "incident.nested", "params": {"question": "who owns OPS-1206"}}
    inner = {"id": "s1", "op": "acli.jira.get", "params": {"key": "OPS-1206"}}
    models = Models(
        planner=scripted(plan(step), plan(inner)),  # shared by both calls, in this order
        checker=scripted(DONE),
        distiller=echo_distiller(),
    )
    req = AskRequest("run the nested workflow", level="standard", scope=["incident"])
    answer = await ask(req, wf_cfg, wf_registry, models)
    assert answer.ops_run == ["incident.nested"], answer.failures
    async with (
        open_store(wf_cfg.log_path) as store,
        store.db.execute("SELECT query, ran_level, depth, trace_id FROM calls ORDER BY ts") as cur,
    ):
        rows = [tuple(r) for r in await cur.fetchall()]
    (inner_q, inner_level, inner_depth, inner_trace), outer_row = rows
    assert inner_q == "who owns OPS-1206" and inner_level == "quick" and inner_depth == 3
    assert inner_trace == outer_row[3] and outer_row[2] == 1  # the root is the first link


def test_clamp() -> None:
    assert clamp("deep", None) == ("standard", None)
    assert clamp("standard", None) == ("quick", None)
    assert clamp("quick", None) == ("quick", None)
    assert clamp("standard", "deep")[0] == "standard" and clamp("standard", "deep")[1]
    assert clamp("deep", "quick") == ("quick", None)


def test_trace_round_trips_and_guards() -> None:
    t = Trace.root("deep").enter("oracle", "ask", "Who owns   X?")
    back = Trace.from_env(t.to_env())
    assert back is not None and back == t
    with pytest.raises(GuardError, match="would repeat"):
        back.enter("oracle", "ask", "who owns x?")  # normalised: the same question
    assert Trace.from_env("{not json") is None


def test_cli_refuses_a_nested_cycle(wf_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    from oracle.cli import app

    parent = Trace.root("standard").enter("oracle", "ask", "loop question")
    env = {
        "ORACLE_TRACE": parent.to_env(),
        "ORACLE_HOME": str(wf_cfg.home),
        "ORACLE_PROFILE": "home",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    result = CliRunner().invoke(app, ["ask", "loop question"])
    assert result.exit_code == 2 and "would repeat" in result.output


def test_cli_run_prints_records(wf_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    from oracle.cli import app

    monkeypatch.setenv("ORACLE_HOME", str(wf_cfg.home))
    monkeypatch.setenv("ORACLE_PROFILE", "home")
    monkeypatch.setenv("ORACLE_REGISTRY", str(wf_cfg.registry_dir))
    monkeypatch.delenv("ORACLE_TRACE", raising=False)
    result = CliRunner().invoke(app, ["run", "incident.investigate", "service=billing-api"])
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert out["status"] == "ok" and out["results"]


def test_cli_run_explains_bad_input(wf_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    from oracle.cli import app

    monkeypatch.setenv("ORACLE_HOME", str(wf_cfg.home))
    monkeypatch.setenv("ORACLE_PROFILE", "home")
    monkeypatch.setenv("ORACLE_REGISTRY", str(wf_cfg.registry_dir))
    monkeypatch.delenv("ORACLE_TRACE", raising=False)
    several = CliRunner().invoke(app, ["run", "incident"])
    assert several.exit_code != 0 and "several ops; name one: incident." in several.output
    bad = CliRunner().invoke(app, ["run", "acli.jira.search", "jql=x", "limit=10,"])
    assert bad.exit_code != 0 and "limit is int: '10,' is not JSON" in bad.output


def test_manifest_rules() -> None:
    from pydantic import ValidationError

    from oracle.registry.schema import Tool

    base = {"name": "w", "description": "d", "kind": "workflow"}
    op = {"name": "go", "when": "w", "entry": "oracle_workflows.w:go"}
    Tool.model_validate(base | {"ops": [op]})
    with pytest.raises(ValidationError, match="entry exactly when"):
        Tool.model_validate(base | {"kind": "tool", "ops": [op]})
    with pytest.raises(ValidationError, match="exactly one of"):
        Tool.model_validate(base | {"ops": [op | {"argv": ["x"]}]})
    with pytest.raises(ValidationError):
        Tool.model_validate(base | {"ops": [op | {"entry": "os:system"}]})
