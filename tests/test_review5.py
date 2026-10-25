"""Fixes from the fifth review: the nested guard, script retries, the sandbox's gather, stub
docstrings, budgets, the crawl's counts and the CLI's guard."""

import ast
import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from oracle.ask import AskRequest, Models, ask
from oracle.config import Config
from oracle.deep.runner import ScriptLimits, run_script
from oracle.deep.sandbox import script_problems
from oracle.exec.dag import PlanRunner, StepRun
from oracle.exec.executor import OpResult
from oracle.log.store import open_store
from oracle.plan.models import Step
from oracle.registry.codegen import build_library, sources
from oracle.registry.loader import Registry
from oracle.registry.schema import Tool
from oracle.render import findings_from
from oracle.sdk.budget import parse_budget
from oracle.workflows.protocol import GuardError, Trace, WorkflowBudget
from tests.conftest import DONE, echo_distiller, plan, scripted
from tests.test_workflows import executor, wf_cfg, wf_registry  # noqa: F401 (fixtures)


async def test_a_nested_ask_carries_its_own_trace(wf_cfg: Config, wf_registry: Registry) -> None:  # noqa: F811
    """root (1) -> workflow (2) -> nested ask (3) -> workflow (4): refused as too deep. Before
    the fix the nested ask saw the workflow's trace, so the fourth level ran at depth 3."""
    inner_step = {"id": "s1", "op": "incident.nested_scoped",
                  "params": {"question": "second question", "scope": ["incident"]}}  # fmt: skip
    outer_step = {"id": "s1", "op": "incident.nested_scoped",
                  "params": {"question": "first question", "scope": ["incident"]}}  # fmt: skip
    models = Models(
        planner=scripted(plan(outer_step), plan(inner_step)),
        checker=scripted(DONE),
        distiller=echo_distiller(),
    )
    req = AskRequest("start", level="standard", scope=["incident"])
    answer = await ask(req, wf_cfg, wf_registry, models)
    assert any("inner call ran at quick" in n for n in answer.notes), answer.notes
    async with (
        open_store(wf_cfg.log_path) as store,
        store.db.execute("SELECT error FROM ops WHERE error LIKE 'refused:%'") as cur,
    ):
        errors = [r[0] for r in await cur.fetchall()]
    assert errors and "nesting deeper than 3" in errors[0]


async def test_a_nested_script_request_stays_under_its_parent(
    home_cfg: Config, fake_registry: Registry
) -> None:
    trace = Trace.root("standard").enter("oracle", "ask", "outer", level="standard")
    step = {"id": "s1", "op": "acli.jira.get", "params": {"key": "OPS-1206"}}
    models = Models(
        planner=scripted(plan(step)), checker=scripted(DONE), distiller=echo_distiller()
    )
    req = AskRequest("who owns OPS-1206", level="standard", script=True)
    answer = await ask(req, home_cfg, fake_registry, models, trace=trace)
    assert answer.level == "standard" and answer.script is None
    assert any("--script needs deep" in n for n in answer.notes)


class SlowExecutor:
    def __init__(self) -> None:
        self.started = 0

    async def run(self, ref: str, params: dict[str, Any]) -> OpResult:
        self.started += 1
        await asyncio.sleep(0.2)
        return OpResult(op=ref, params=params, records=[{"key": "OPS-1"}], pointers=["jira:OPS-1"])


async def test_a_cancelled_caller_leaves_the_shared_lookup_running(
    fake_registry: Registry,
) -> None:
    """A failed script attempt cancels its in-flight calls; the corrected attempt that asks for
    the same lookup must get the result, not a CancelledError (or a hang), and own it: the
    cancelled step never finished, so a reuse marker pointing at it would log no pointers."""
    slow = SlowExecutor()
    runner = PlanRunner(fake_registry, slow)  # type: ignore[arg-type]
    step = Step(id="x1_1", op="acli.jira.get", params={"key": "OPS-1"})
    first = asyncio.ensure_future(runner.run_literal(step))
    await asyncio.sleep(0.05)
    first.cancel()
    again = await asyncio.wait_for(runner.run_literal(step.model_copy(update={"id": "x2_1"})), 2)
    assert again.result.ok and again.same_as is None and slow.started == 1
    assert set(runner.runs) == {"x2_1"} and again.result.pointers == ["jira:OPS-1"]
    reused = await runner.run_literal(step.model_copy(update={"id": "x2_2"}))
    assert reused.same_as == "x2_1"


async def test_every_call_gets_a_reply(tmp_path: Path, fake_registry: Registry) -> None:
    lib = build_library(fake_registry, tmp_path).path

    async def broken(method: str, args: dict[str, Any]) -> Any:
        raise RuntimeError("boom")

    code = (
        "from oracle_tools import acli, report\n"
        "try:\n"
        '    await acli.jira_get(key="OPS-1")\n'
        "except ValueError as e:\n"
        "    report(str(e))\n"
    )
    result = await run_script(code, broken, lib, ScriptLimits(wall_s=10))
    assert result.ok and "the oracle failed this call" in result.reports[0]


async def test_gather_hands_scripts_no_futures(tmp_path: Path, fake_registry: Registry) -> None:
    lib = build_library(fake_registry, tmp_path).path

    async def handler(method: str, args: dict[str, Any]) -> Any:
        return [{"key": args["params"]["key"]}]

    code = """
from oracle_tools import acli, gather, report
pending = gather(acli.jira_get(key="A-1"), acli.jira_get(key="A-2"))
report(str(pending).split()[0])
found = await pending
report(isinstance(found, list))
"""
    result = await run_script(code, handler, lib)
    assert result.reports == ["<coroutine", True], result.error
    assert script_problems("pending.get_loop()")  # and a future's methods are refused anyway


async def test_an_oversized_message_is_an_error_not_a_crash(
    tmp_path: Path, fake_registry: Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    import oracle.deep.runner as runner_module

    monkeypatch.setattr(runner_module, "LINE_LIMIT", 2048)
    lib = build_library(fake_registry, tmp_path).path

    async def handler(method: str, args: dict[str, Any]) -> Any:
        return []

    result = await run_script(
        "from oracle_tools import report\nreport('x' * 100000)\n", handler, lib
    )
    assert result.error and "unreadable message" in result.error


def test_stub_docstrings_survive_quotes() -> None:
    tool = Tool.model_validate({
        "name": "quoted", "description": 'says "hi"',
        "ops": [
            {"name": "ask", "when": 'questions like "who owns X"', "argv": ["x"]},
            {"name": "path", "when": "a path like C:\\\\temp\\\\", "argv": ["x"],
             "params": {"q": {"desc": 'ends in "quote"'}}},
        ],
    })  # fmt: skip
    files, _ = sources(Registry(tools={"quoted": tool}))
    for rel, text in files.items():
        ast.parse(text, rel)


def test_raw_findings_tolerate_the_script_report(fake_registry: Registry) -> None:
    result = OpResult(
        op="script.report", params={}, records=[{"count": 3}], pointers=["script:c#1"]
    )
    assert (
        findings_from([StepRun("script/report", result)], fake_registry)[0].pointer == "script:c#1"
    )


def test_a_numeric_config_budget() -> None:
    assert parse_budget(150000, 1) == 150000 and parse_budget("unlimited", 1) is None


def test_the_workflow_budget_reserves_before_calls_start() -> None:
    budget = WorkflowBudget(model_calls=8)
    first = budget.reserve("standard")
    second = budget.reserve("standard")
    with pytest.raises(GuardError, match="needs about 4"):
        budget.reserve("standard")  # three started together would pass a check-after cap
    budget.settle(first, 2)
    budget.settle(second, 3)
    assert budget.used == 5 and budget.reserved == 0
    assert budget.reserve("quick") == 1


async def test_a_bug_in_a_workflow_is_an_error(wf_cfg: Config, wf_registry: Registry) -> None:  # noqa: F811
    result = await executor(wf_cfg, wf_registry).run("incident.broken", {})
    assert result.status == "error" and "no_such_param" in (result.error or "")


def _cli_env(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORACLE_HOME", str(cfg.home))
    monkeypatch.setenv("ORACLE_PROFILE", "home")
    monkeypatch.setenv("ORACLE_REGISTRY", str(cfg.registry_dir))


def test_oracle_run_enters_the_guard(wf_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    from oracle.cli import app

    _cli_env(wf_cfg, monkeypatch)
    values = {"service": "billing-api"}
    parent = Trace.root("deep").enter("run", "incident.investigate", values)
    monkeypatch.setenv("ORACLE_TRACE", parent.to_env())
    result = CliRunner().invoke(app, ["run", "incident.investigate", "service=billing-api"])
    assert result.exit_code == 2 and "would repeat incident.investigate" in result.output


def test_tool_test_runs_workflow_examples(wf_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    from oracle.cli import app

    _cli_env(wf_cfg, monkeypatch)
    monkeypatch.delenv("ORACLE_TRACE", raising=False)
    result = CliRunner().invoke(app, ["tool", "test", "incident"])
    # the test-only ops have no examples; the real one runs instead of "not_configured"
    assert "PASS  incident.investigate[0]" in result.output, result.output
    assert "not_configured" not in result.output


def test_claude_concurrency_ignores_a_bad_value(monkeypatch: pytest.MonkeyPatch) -> None:
    from oracle.llm.claude_code import _concurrency

    monkeypatch.setenv("ORACLE_CLAUDE_CONCURRENCY", "auto")
    assert _concurrency() == 5
    monkeypatch.setenv("ORACLE_CLAUDE_CONCURRENCY", "0")
    assert _concurrency() == 1


async def test_terms_the_hop_plan_skips_are_not_counted_as_searched(
    home_cfg: Config, fake_registry: Registry
) -> None:
    from pydantic_ai.models.function import FunctionModel

    from tests.conftest import _final, prompt_text

    terms = [
        {"value": "OPS-1206", "description": "a ticket key", "relevance": 0.9},
        {"value": "the moon", "description": "nothing any tool holds", "relevance": 0.8},
    ]

    def extract(messages, info):  # type: ignore[no-untyped-def]
        marker = "Already searched (do not return):"
        text = prompt_text(messages)
        already = text.split(marker, 1)[1].split("\n", 1)[0] if marker in text else ""
        return _final(info, {"terms": [] if "ops-1206" in already else terms})

    def hop(messages, info):  # type: ignore[no-untyped-def]
        text = prompt_text(messages)
        if "Terms to look up:" not in text:
            first = {"id": "s1", "op": "workiq.teams.search", "params": {"query": "gateway"}}
            return _final(info, plan(first))
        step = {
            "id": "h2s1",
            "op": "acli.jira.get",
            "params": {"key": "OPS-1206"},
            "term": "OPS-1206",
        }
        return _final(info, {"steps": [step]})

    models = Models(
        planner=FunctionModel(hop),
        extractor=FunctionModel(extract),
        distiller=echo_distiller(),
        reducer=echo_distiller(),
    )
    answer = await ask(
        AskRequest("gateway 5xx", level="deep", depth=2), home_cfg, fake_registry, models
    )
    assert answer.terms_searched == 1
    assert "the moon (no op can look it up)" in answer.unexpanded
    assert json.dumps(answer.unexpanded).count("OPS-1206") == 0
