"""Each strategy through its own interface: a Context over the fake tools, scripted models, and
the Outcome it returns. No summary, no answer, no log."""

from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel

from oracle.ask import AskRequest, Models
from oracle.budget import Budget
from oracle.config import Config
from oracle.exec.executor import Executor
from oracle.registry.loader import Registry
from oracle.strategy import Context, strategy_for
from tests.conftest import DONE, plan, scripted
from tests.test_deep import deep_models
from tests.test_deep_script import CHAIN_SCRIPT, QUESTION, scripter
from tests.test_deep_script import models as script_models


def context(
    cfg: Config, registry: Registry, models: Models, req: AskRequest, tokens: int | None = None
) -> Context:
    executor = Executor(registry, cfg.tool_env(), cfg.spill_dir)
    return Context.build(req, cfg, registry, models, Budget.for_tier(req.level, tokens), executor)


async def test_a_failed_plan_is_a_failure_not_a_traceback(
    home_cfg: Config, fake_registry: Registry
) -> None:
    def down(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError("model down")

    req = AskRequest("who is jon", level="quick")
    ctx = context(home_cfg, fake_registry, Models(planner=FunctionModel(down)), req)
    out = await strategy_for("quick", False).run(ctx)
    assert out.runs == [] and out.plan is None
    assert out.failures == ["planner: RuntimeError: model down"]


async def test_checked_replans_and_keeps_the_last_checks_gaps(
    home_cfg: Config, fake_registry: Registry
) -> None:
    retry = {
        "complete": False,
        "gaps": ["no person named jon"],
        "next_steps": [
            {"id": "s2", "op": "workiq.people.lookup", "params": {"name": "andrew neville"}}
        ],
    }
    first = plan({"id": "s1", "op": "workiq.people.lookup", "params": {"name": "jon"}})
    models = Models(planner=scripted(first), checker=scripted(retry, DONE))
    ctx = context(home_cfg, fake_registry, models, AskRequest("who is jon"))
    out = await strategy_for("standard", False).run(ctx)
    assert out.replans == 1 and out.round_name == "replan"
    assert [(r.op, r.round) for r in out.runs] == [
        ("workiq.people.lookup", 0),
        ("workiq.people.lookup", 1),
    ]
    assert out.gaps == []  # the second check found none; the first one's gap is not kept
    assert out.interpretation == "i" and out.hops == 0 and out.stopped is None


async def test_crawled_says_what_it_searched_and_why_it_stopped(
    home_cfg: Config, fake_registry: Registry
) -> None:
    req = AskRequest(QUESTION, level="deep", depth=4)
    ctx = context(home_cfg, fake_registry, deep_models(), req, tokens=200_000)
    out = await strategy_for("deep", False).run(ctx)
    assert (out.hops, out.stopped, out.round_name) == (4, "depth", "hop")
    assert out.searched == ["OPS-1206", "ingestion gateway 5xx runbook", "Priya Raman"]
    assert [r.op for r in out.runs] == [
        "workiq.teams.search",
        "acli.jira.get",
        "acli.confluence.search",
        "workiq.people.lookup",
    ]
    assert out.summary_runs == [r for r in out.runs if r.found]


async def test_scripted_hands_the_summary_what_the_script_reported(
    home_cfg: Config, fake_registry: Registry
) -> None:
    req = AskRequest(QUESTION, level="deep", script=True)
    models = script_models(scripter=scripter(CHAIN_SCRIPT))
    out = await strategy_for("deep", True).run(context(home_cfg, fake_registry, models, req))
    assert out.script == CHAIN_SCRIPT and out.plan is None and not out.failures
    assert out.interpretation == "the gateway 5xx incident"
    assert out.trace[0].startswith("script: attempt 1,") and out.round_name == "hop"
    assert [r.op for r in out.logged] == ["script.report"]  # logged, but not a tool's result
    assert "script.report" not in {r.op for r in out.runs}
    reported = {p for r in out.summary_runs for p in r.result.pointers}
    assert {"jira:OPS-1206", "conf:90011", "people:u-1003"} <= reported
