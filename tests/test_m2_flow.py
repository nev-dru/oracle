"""The standard tier end to end on the fake tools, with scripted models (no model spend)."""

import shutil
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel

from oracle.ask import AskRequest, Models, ask
from oracle.config import REPO_ROOT, Config, ConfigError, load_config
from oracle.log.store import open_store
from oracle.registry.loader import Registry, load_registry
from oracle.render import to_markdown
from tests.conftest import DONE, _final, echo_distiller, plan, scripted


def models(planner: dict | FunctionModel, *checks: dict, distiller: Any = None) -> Models:
    return Models(
        planner=planner if isinstance(planner, FunctionModel) else scripted(planner),
        checker=scripted(*(checks or (DONE,))),
        distiller=distiller or echo_distiller(),
    )


async def test_people_profile_recipe_is_breadth_grouped_by_source(
    home_cfg: Config, fake_registry: Registry
) -> None:
    m = models(plan({"id": "s1", "op": "people.profile", "params": {"name": "Andrew Neville"}}))
    answer = await ask(AskRequest("who is andrew neville"), home_cfg, fake_registry, m)
    assert answer.recipes == ["people.profile"] and answer.breadth
    assert {"workiq.people.lookup", "acli.jira.search", "workiq.teams.search"} <= set(
        answer.ops_run
    )
    md = to_markdown(answer)
    assert md.index("### workiq") < md.index("### acli") or md.index("### acli") < md.index(
        "### workiq"
    )
    sources = [f.source.split(".")[0] for f in answer.findings]
    assert sources == sorted(sources, key=sources.index)  # contiguous groups
    assert "## Refine" in md and "[people.profile] s1/directory" in md
    async with (
        open_store(home_cfg.log_path) as store,
        store.db.execute("SELECT step_id FROM ops WHERE call_id = ?", (answer.call_id,)) as cur,
    ):
        steps = [row[0] for row in await cur.fetchall()]
    assert set(steps) == {"s1/directory", "s1/tickets", "s1/messages"}


async def test_today_events_names_tools_not_here_and_the_window(
    home_cfg: Config, fake_registry: Registry
) -> None:
    m = models(plan({"id": "s1", "op": "today.events"}))
    answer = await ask(AskRequest("important events from today"), home_cfg, fake_registry, m)
    assert answer.time_window and answer.time_window.startswith("today = ")
    assert any(line.startswith("time: today = ") for line in answer.trace)
    assert {"argocd", "kubectl"} <= set(answer.unavailable)
    ran = set(answer.ops_run)
    assert {"workiq.calendar.today", "acli.jira.search", "workiq.teams.search"} <= ran
    assert "argocd.app.history" not in ran


async def test_check_replans_with_new_steps_then_stops(
    home_cfg: Config, fake_registry: Registry
) -> None:
    retry = {
        "complete": False,
        "gaps": ["no person named jon"],
        "next_steps": [
            {"id": "s2", "op": "workiq.people.lookup", "params": {"name": "andrew neville"}}
        ],
    }
    m = models(
        plan({"id": "s1", "op": "workiq.people.lookup", "params": {"name": "jon"}}), retry, DONE
    )
    answer = await ask(AskRequest("who is jon"), home_cfg, fake_registry, m)
    assert answer.replans == 1 and "replan 1:" in answer.trace
    assert answer.ops_run == ["workiq.people.lookup", "workiq.people.lookup"]
    assert answer.no_match == []  # the source did match once
    assert answer.findings[0].pointer == "people:u-1001"


async def test_replans_stop_at_the_limit(home_cfg: Config, fake_registry: Registry) -> None:
    def always_more(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        n = sum(1 for m in messages for _ in m.parts)  # unique ids per call
        step = {"id": f"x{n}{id(messages)}", "op": "workiq.calendar.today"}
        return _final(info, {"complete": False, "gaps": ["more"], "next_steps": [step]})

    m = Models(
        planner=scripted(plan({"id": "s1", "op": "workiq.calendar.today"})),
        checker=FunctionModel(always_more),
        distiller=echo_distiller(),
    )
    answer = await ask(AskRequest("meetings"), home_cfg, fake_registry, m)
    assert answer.replans == 2
    assert any("replan limit" in n for n in answer.notes)


async def test_distill_must_cite_real_pointers_and_falls_back(
    home_cfg: Config, fake_registry: Registry
) -> None:
    bad_then_good = scripted(
        {"findings": [{"text": "x", "pointer": "jira:FAKE-1"}], "confidence": "high"},
        {"findings": [{"text": "Andrew", "pointer": "people:u-1001"}], "confidence": "high"},
    )
    step = {"id": "s1", "op": "workiq.people.lookup", "params": {"name": "andrew neville"}}
    answer = await ask(
        AskRequest("who is andrew neville"),
        home_cfg,
        fake_registry,
        models(plan(step), distiller=bad_then_good),
    )
    assert [f.pointer for f in answer.findings] == ["people:u-1001"]

    def broken(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError("haiku down")

    fallback = await ask(
        AskRequest("who is andrew neville"),
        home_cfg,
        fake_registry,
        models(plan(step), distiller=FunctionModel(broken)),
    )
    assert fallback.findings[0].text.startswith("Andrew Neville")
    assert any("distill failed" in n for n in fallback.notes) and fallback.confidence == "low"


async def test_check_steps_are_validated_and_retried(
    home_cfg: Config, fake_registry: Registry
) -> None:
    bogus = {"complete": False, "next_steps": [{"id": "s2", "op": "kubectl.get"}]}
    m = models(plan({"id": "s1", "op": "workiq.calendar.today"}), bogus, DONE)
    answer = await ask(AskRequest("meetings today"), home_cfg, fake_registry, m)
    assert answer.replans == 0 and "kubectl.get" not in answer.ops_run


async def test_missing_model_config_fails_before_tools_run(
    fake_registry: Registry, tmp_path: Path
) -> None:
    cfg = load_config({"ORACLE_HOME": str(tmp_path), "ORACLE_PROFILE": "home"})
    m = Models(planner=scripted(plan({"id": "s1", "op": "workiq.calendar.today"})))
    with pytest.raises(ConfigError, match=r"ORACLE_MODEL_(PLANNER|EXTRACT)"):
        await ask(AskRequest("meetings"), cfg, fake_registry, m)


def _fixtures_ready() -> bool:
    if not shutil.which("codebase-memory-mcp") or not (REPO_ROOT / ".fixtures/vex.db").exists():
        return False
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1)
    except OSError:
        return False
    return True


needs_fixtures = pytest.mark.skipif(
    not _fixtures_ready(), reason="needs `make fixtures`, codebase-memory-mcp and Ollama"
)


@pytest.fixture
def home_registry(home_cfg: Config) -> Registry:
    assert home_cfg.profile and home_cfg.profile.tools
    return load_registry(home_cfg.registry_dir, set(home_cfg.profile.tools))


@needs_fixtures
async def test_find_and_expand_recipe_on_real_code_tools(
    home_cfg: Config, home_registry: Registry
) -> None:
    step = {"id": "s1", "op": "code.find_and_expand", "params": {"concept": "ownership transfer"}}
    answer = await ask(
        AskRequest("where ownership transfer happens in code"),
        home_cfg,
        home_registry,
        models(plan(step)),
    )
    assert {"code-search.search", "code-graph.symbols_at", "code-graph.expand"} <= set(
        answer.ops_run
    )
    assert any(
        f.pointer.startswith("code:") and "handlers.py" in f.pointer for f in answer.findings
    )


@needs_fixtures
async def test_salesforce_to_cloud_needs_a_replan(
    home_cfg: Config, home_registry: Registry
) -> None:
    """No acme symbol is named *cloud*, so the recipe's path is skipped; the check searches by
    meaning, then joins the two ends by name."""
    search = {
        "complete": False,
        "gaps": ["no symbol named cloud"],
        "next_steps": [
            {"id": "s2", "op": "code-search.search", "params": {"query": "cloud ingestion gateway"}}
        ],
    }
    join = {
        "complete": False,
        "gaps": ["no path yet"],
        "next_steps": [
            {
                "id": "s3",
                "op": "code-graph.path_between",
                "params": {"source": "sync_salesforce_changes", "target": "publish"},
            }
        ],
    }
    first = {
        "id": "s1",
        "op": "code.trace_path",
        "params": {"source": "salesforce", "target": "cloud"},
    }
    answer = await ask(
        AskRequest("how salesforce talks to cloud"),
        home_cfg,
        home_registry,
        models(plan(first), search, join, DONE),
    )
    assert answer.replans == 2
    first_path = next(line for line in answer.trace if "s1/path" in line)
    assert "skipped" in first_path or "0 results" in first_path
    assert {"code-graph.find_symbol", "code-graph.path_between"} <= set(answer.ops_run)
    assert any(
        line.startswith("s3 code-graph.path_between") and "4 results" in line
        for line in answer.trace
    )
