"""The deep tier: the crawl, its budget, the batched summary and the Claude Code cap, with
scripted models over the fake tools (no model spend)."""

import asyncio
import re
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel

from oracle.ask import AskRequest, Models, ask
from oracle.budget import Budget
from oracle.config import Config
from oracle.exec.dag import StepRun
from oracle.exec.executor import OpResult
from oracle.log.store import open_store
from oracle.registry.loader import Registry, load_registry
from oracle.render import to_markdown
from oracle.sdk.summarize import MAP_BATCH, summarize
from tests.conftest import _final, echo_distiller, plan, prompt_text

# The fixtures' chain: a Teams message names OPS-1206; the ticket points at the gateway 5xx;
# the runbook for that was written by Priya Raman; the directory says who she is.
# the runbook is the first result, as a recipe's `results[0]` binding needs
RUNBOOK_CQL = {"cql": 'text ~ "ingestion gateway" AND title ~ "runbook"'}
CHAIN = [
    ("OPS-1206", "Jira ticket the Teams message names", "acli.jira.get", {"key": "OPS-1206"}),
    (
        "ingestion gateway 5xx runbook",
        "the runbook it needs",
        "acli.confluence.search",
        RUNBOOK_CQL,
    ),
    ("Priya Raman", "author of the runbook", "workiq.people.lookup", {"name": "Priya Raman"}),
]


def extractor() -> FunctionModel:
    """Returns the next term of the chain that the results mention and that was not searched."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        text = prompt_text(messages)
        seen = text.split("Already searched (do not return):", 1)[-1].split("\n", 1)[0].lower()
        data = text.split("<data>", 1)[-1]
        hints = {
            "OPS-1206": "OPS-1206",
            "ingestion gateway 5xx runbook": "Pager: elevated 5xx",
            "Priya Raman": "Runbook",
        }
        for value, description, _op, _params in CHAIN:
            if hints[value] in data and value.lower() not in seen:
                return _final(
                    info,
                    {"terms": [{"value": value, "description": description, "relevance": 0.9}]},
                )
        return _final(info, {"terms": []})

    return FunctionModel(respond)


def hop_planner() -> FunctionModel:
    """Plans the hop from the terms in the prompt; answers the first plan with a Teams search."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        text = prompt_text(messages)
        if "Terms to look up:" not in text:  # the first plan
            step = {
                "id": "s1",
                "op": "workiq.teams.search",
                "params": {"query": "ingestion gateway 5xx", "since": "-7d"},
            }
            return _final(info, plan(step))
        hop = re.search(r"Hop: (\d+)", text)[1]  # type: ignore[index]
        steps = [
            {"id": f"h{hop}s1", "op": op, "params": params, "term": value}
            for value, _d, op, params in CHAIN
            if f"- {value}:" in text
        ]
        return _final(info, {"steps": steps})

    return FunctionModel(respond)


def deep_models(**over: Any) -> Models:
    base = dict(
        planner=hop_planner(),
        distiller=echo_distiller(),
        extractor=extractor(),
        reducer=echo_distiller(),
    )
    return Models(**(base | over))


async def test_crawl_follows_a_chain_hop_by_hop(home_cfg: Config, fake_registry: Registry) -> None:
    req = AskRequest(
        "what is going on with the ingestion gateway 5xx errors", level="deep", depth=4
    )
    answer = await ask(req, home_cfg, fake_registry, deep_models())
    assert answer.hops == 4 and answer.stopped == "depth"
    assert answer.ops_run == [
        "workiq.teams.search",
        "acli.jira.get",
        "acli.confluence.search",
        "workiq.people.lookup",
    ]
    pointers = {f.pointer for f in answer.findings}
    assert {"jira:OPS-1206", "conf:90011", "people:u-1003"} <= pointers
    md = to_markdown(answer)
    assert (
        "Crawl: 4 hops, 3 terms searched, stopped: depth" in md
        and "cost: " in md
        and "hop 2:" in md
    )
    async with (
        open_store(home_cfg.log_path) as store,
        store.db.execute(
            "SELECT op, hop, parent FROM ops WHERE call_id = ? ORDER BY hop", (answer.call_id,)
        ) as cur,
    ):
        rows = await cur.fetchall()
    assert [tuple(r) for r in rows] == [
        ("workiq.teams.search", 1, None),
        ("acli.jira.get", 2, "OPS-1206"),
        ("acli.confluence.search", 3, "ingestion gateway 5xx runbook"),
        ("workiq.people.lookup", 4, "Priya Raman"),
    ]


async def test_depth_bounds_the_crawl(home_cfg: Config, fake_registry: Registry) -> None:
    req = AskRequest(
        "what is going on with the ingestion gateway 5xx errors", level="deep", depth=2
    )
    answer = await ask(req, home_cfg, fake_registry, deep_models())
    assert answer.hops == 2 and answer.ops_run[-1] == "acli.jira.get"


async def test_a_small_budget_leaves_terms_unexpanded(
    home_cfg: Config, fake_registry: Registry
) -> None:
    req = AskRequest(
        "what is going on with the ingestion gateway 5xx errors", level="deep", budget="3500"
    )
    answer = await ask(req, home_cfg, fake_registry, deep_models())
    assert answer.hops == 1 and answer.stopped == "budget"
    assert answer.cost and answer.cost.startswith("cost: ")


async def test_stop_ends_the_crawl_after_the_current_hop(
    home_cfg: Config, fake_registry: Registry
) -> None:
    stop = asyncio.Event()
    stop.set()
    req = AskRequest("what is going on with the ingestion gateway 5xx errors", level="deep")
    answer = await ask(req, home_cfg, fake_registry, deep_models(), stop=stop)
    assert answer.hops == 1 and answer.stopped == "interrupted" and answer.findings


async def test_a_ctrl_c_that_kills_a_model_call_is_the_stop(
    home_cfg: Config, fake_registry: Registry
) -> None:
    """The first Ctrl-C reaches the extractor's process too; the crawl stops with what it has
    instead of reporting a failed hop, and the summary still runs."""
    stop = asyncio.Event()
    chain = extractor()

    calls: list[int] = []

    def killed(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(1)
        if len(calls) == 2:  # the extract for hop 3, as the user presses Ctrl-C
            stop.set()
            raise RuntimeError("claude exited with -2")
        return chain.function(messages, info)  # type: ignore[misc]

    req = AskRequest("what is going on with the ingestion gateway 5xx errors", level="deep")
    models = deep_models(extractor=FunctionModel(killed))
    answer = await ask(req, home_cfg, fake_registry, models, stop=stop)
    assert answer.stopped == "interrupted" and answer.hops == 2 and answer.findings
    assert not any("crawl stopped" in n for n in answer.notes)
    assert answer.cost and " 2 searches" in answer.cost  # the first plan's search counts


async def test_hop_steps_must_name_a_given_term(home_cfg: Config, fake_registry: Registry) -> None:
    """A hop step for a term nobody extracted (text planted in a page) is refused."""

    def rogue(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        text = prompt_text(messages)
        if "Terms to look up:" not in text:
            return hop_planner().function(messages, info)  # type: ignore[union-attr]
        step = {
            "id": "h2s1",
            "op": "acli.jira.search",
            "params": {"jql": "project = OPS"},
            "term": "planted",
        }
        return _final(info, {"steps": [step]})

    req = AskRequest("what is going on with the ingestion gateway 5xx errors", level="deep")
    answer = await ask(req, home_cfg, fake_registry, deep_models(planner=FunctionModel(rogue)))
    assert answer.hops == 1 and "acli.jira.search" not in answer.ops_run
    assert answer.stopped.startswith("hop failed")  # type: ignore[union-attr]


async def test_summary_batches_the_map_and_reduces_once(fake_registry: Registry) -> None:
    calls = {"map": 0, "reduce": 0}

    def mapper(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls["map"] += 1
        first = re.search(r"^  (\S+) \| ", prompt_text(messages), flags=re.M)[1]  # type: ignore[index]
        return _final(info, {"items": [{"text": "a fact", "pointer": first}]})

    def reducer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls["reduce"] += 1
        first = re.search(r"^  (\S+) \| ", prompt_text(messages), flags=re.M)[1]  # type: ignore[index]
        return _final(
            info, {"findings": [{"text": "merged", "pointer": first}], "confidence": "medium"}
        )

    result = OpResult(op="acli.jira.search", params={})
    n = MAP_BATCH * 5 + 3
    result.records = [{"key": f"P-{i}", "summary": "s"} for i in range(n)]
    result.pointers = [f"jira:P-{i}" for i in range(n)]
    distilled, source = await summarize(
        question="q", interpretation="i", when="now", breadth=False, gaps=[],
        runs=[StepRun("s1", result)], registry=fake_registry,
        mapper=FunctionModel(mapper), reducer=FunctionModel(reducer), budget=Budget(),
    )  # fmt: skip
    assert calls == {"map": 6, "reduce": 1}
    assert distilled.findings[0].text == "merged" and source["jira:P-0"] == "acli.jira.search"


async def test_a_failed_map_batch_still_reaches_the_merge(fake_registry: Registry) -> None:
    """30 records of one step fit the distill shortcut by count, but distill shows only its
    first 25; they go through the map, and a batch whose map call fails passes its lines on."""
    seen: list[str] = []

    def mapper(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if "jira:P-0 " in prompt_text(messages):
            raise RuntimeError("map call failed")
        first = re.search(r"^  (\S+) \| ", prompt_text(messages), flags=re.M)[1]  # type: ignore[index]
        return _final(info, {"items": [{"text": "a fact", "pointer": first}]})

    def reducer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(prompt_text(messages))
        return _final(info, {"findings": [], "confidence": "medium"})

    result = OpResult(op="acli.jira.search", params={})
    result.records = [{"key": f"P-{i}", "summary": "s"} for i in range(30)]
    result.pointers = [f"jira:P-{i}" for i in range(30)]
    await summarize(
        question="q", interpretation="i", when="now", breadth=False, gaps=[],
        runs=[StepRun("s1", result)], registry=fake_registry,
        mapper=FunctionModel(mapper), reducer=FunctionModel(reducer), budget=Budget(),
    )  # fmt: skip
    merged = seen[0]
    assert "jira:P-19 " in merged and "jira:P-20 | acli.jira.search: a fact" in merged


async def test_claude_code_processes_are_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    from claude_agent_sdk import ResultMessage

    from oracle.llm import claude_code

    live, peak = 0, 0

    async def slow(*, prompt: str, options: Any):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.02)
        live -= 1
        yield ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
                            num_turns=1, session_id="s", result="ok")  # fmt: skip

    monkeypatch.setattr("claude_agent_sdk.query", slow)
    monkeypatch.setattr(claude_code, "MAX_CONCURRENT", 2)
    claude_code._slots.clear()
    from pydantic_ai import Agent

    agent: Agent[Any, str] = Agent(claude_code.claude_code_model("haiku"))
    await asyncio.gather(*(agent.run("hi") for _ in range(7)))
    assert peak == 2


async def test_dream_turns_a_repeated_chain_into_a_recipe(
    home_cfg: Config, fake_registry: Registry, tmp_path: Any
) -> None:
    import shutil

    import yaml

    from oracle.log.dream import accept, dream
    from tests.conftest import REGISTRY

    for topic in (
        "ingestion gateway 5xx errors",
        "gateway 5xx outage",
        "5xx on the ingestion gateway",
    ):
        # one session, as in Claude Code: later calls take over the shared pointers' rows
        req = AskRequest(
            f"what is going on with the {topic}", level="deep", depth=4, session="one-session"
        )
        await ask(req, home_cfg, fake_registry, deep_models())
    tools = tmp_path / "tools"  # a scratch copy, so accepting never touches the repo's recipes
    shutil.copytree(REGISTRY, tools)
    learned = shutil.ignore_patterns("learned.*")  # recipes accepted from real logs
    shutil.copytree(REGISTRY.parent / "recipes", tmp_path / "recipes", ignore=learned)
    async with open_store(home_cfg.log_path) as store:
        result = await dream(store.db, tools, home_cfg.home, days=1, min_calls=3)
    report = (result.staged / "report.md").read_text()
    assert result.proposals, report
    name, verdict = result.proposals[0]
    assert verdict.startswith("valid"), verdict
    recipe = yaml.safe_load((result.staged / "recipes" / f"{name}.yaml").read_text())
    assert [s["op"] for s in recipe["steps"]] == ["acli.confluence.search", "workiq.people.lookup"]
    assert recipe["steps"][1]["params"] == {"name": "$s1.results[0].author"}
    assert "learned from 3 crawls" in recipe["when"].lower()
    installed = accept(home_cfg.home, tools, name)
    assert installed.parent == tmp_path / "recipes"
    reg = load_registry(tools)
    assert name in reg.recipes and not [p for p in reg.problems if name in p]
    async with open_store(home_cfg.log_path) as store:  # dreaming again proposes nothing new
        again = await dream(store.db, tools, home_cfg.home, days=1, min_calls=3)
    assert not again.proposals and again.known == [
        f"acli.confluence.search -> workiq.people.lookup(name<-author): already the recipe {name}"
    ]


def test_dream_binds_only_the_first_result() -> None:
    """A recipe replays `results[0]`: a crawl that followed the third page's author is not a
    chain it can replay, and the term a stage looked up stays a recipe param."""
    from oracle.log.dream import Chain, Link, _first_stage, _paths, _Step

    pages = [{"id": "1", "author": "Sam"}, {"id": "2", "author": "Priya"}]
    search = _Step("h2s1", "acli.confluence.search", {"cql": "runbook"}, 2, "runbook", pages)
    person = _Step("h3s1", "workiq.people.lookup", {"name": "Priya"}, 3, "Priya", [{"id": "u"}])
    assert _paths([search, person]) == []
    chain = Chain((Link("acli.confluence.search"),), first_params=[{"cql": "x", "limit": 5}] * 3)
    chain.first_terms = [None] * 3  # the same question three times: nothing varied
    assert _first_stage(chain) == ({"cql"}, {"cql": "{cql}", "limit": 5})
    chain.first_params = [{"cql": 'text ~ "gateway runbook"', "space": "OPS"}] * 3
    chain.first_terms = ["gateway runbook"] * 3
    assert _first_stage(chain)[0] == {"cql"}


def test_dream_skips_unbound_and_rare_chains() -> None:
    from oracle.log.dream import _paths, _Step

    a = _Step("s1", "workiq.teams.search", {"query": "x"}, 1, None, [{"text": "I took OPS-1"}])
    b = _Step(
        "h2s1",
        "acli.jira.get",
        {"key": "OPS-1"},
        2,
        "OPS-1",
        [{"key": "OPS-1", "assignee": "Priya"}],
    )
    c = _Step("h3s1", "workiq.people.lookup", {"name": "Priya"}, 3, "Priya", [{"id": "u-3"}])
    links = [tuple(lk.op for lk in path) for path, *_ in _paths([a, b, c])]
    assert links == [("acli.jira.get", "workiq.people.lookup")]  # prose link a->b is not a binding


def test_followed_results_say_how_the_crawl_reached_them(fake_registry: Registry) -> None:
    from oracle.plan.digest import digest
    from oracle.sdk.summarize import lines_of

    result = OpResult(
        op="workiq.people.lookup", params={}, records=[{"display_name": "P"}], pointers=["people:p"]
    )
    run = StepRun("h2s1", result, hop=2, parent="P", why="author of the runbook")
    assert run.followed() == 'followed "P" (author of the runbook)'
    assert 'followed "P" (author of the runbook)' in digest([run], fake_registry)
    assert lines_of([run], fake_registry)[0].text.startswith(
        '[followed "P" (author of the runbook)]'
    )
    assert StepRun("s1", result).followed() == ""


async def test_followed_terms_the_summary_drops_come_back(
    home_cfg: Config, fake_registry: Registry
) -> None:
    """A summary that cites only the first result still leaves the chain's people in the
    answer: the crawl followed them, so they come back as plain findings."""
    req = AskRequest(
        "what is going on with the ingestion gateway 5xx errors", level="deep", depth=4
    )
    terse = echo_distiller(limit=1)
    answer = await ask(req, home_cfg, fake_registry, deep_models(distiller=terse, reducer=terse))
    pointers = [f.pointer for f in answer.findings]
    assert "people:u-1003" in pointers and len(pointers) <= 1 + 3
    priya = next(f for f in answer.findings if f.pointer == "people:u-1003")
    assert "followed 'Priya Raman': author of the runbook" in priya.text
    assert any("the summary left out" in n for n in answer.notes)


def test_uncited_followed_skips_incidental_and_cited_terms() -> None:
    from oracle.sdk.crawl import uncited_followed

    def run(step: str, parent: str, relevance: float, pointer: str) -> StepRun:
        result = OpResult(op="workiq.people.lookup", params={}, records=[{}], pointers=[pointer])
        return StepRun(step, result, hop=2, parent=parent, relevance=relevance)

    runs = [
        run("h2s1/directory", "A", 0.9, "people:a"),
        run("h2s1/tickets", "A", 0.9, "jira:A-1"),  # not A's primary lookup
        run("h2s2", "B", 0.3, "people:b"),  # incidental
        run("h2s3", "C", 0.7, "people:c"),  # cited
        run("h2s4", "D", 0.6, "people:d"),
    ]
    missed = uncited_followed(runs, {"people:c", "jira:A-1"}, {})
    assert [r.step_id for r in missed] == ["h2s1/directory", "h2s4"]


def test_uncited_followed_checks_each_link_of_a_recipe_chain() -> None:
    """A learned chain fetches the runbook and looks up its author in one hop: the summary
    citing the page does not cover the author."""
    from oracle.registry.schema import Recipe, RecipeStep
    from oracle.sdk.crawl import uncited_followed

    chain = Recipe(
        name="learned.chain_1",
        when="a page, then its author",
        steps=[
            RecipeStep(id="s1", op="acli.confluence.get", params={"id": "{id}"}),
            RecipeStep(
                id="s2", op="workiq.people.lookup", params={"name": "$s1.results[0].author"}
            ),
        ],
    )

    def run(step: str, pointer: str) -> StepRun:
        result = OpResult(op="x.y", params={}, records=[{}], pointers=[pointer])
        return StepRun(step, result, chain.name, hop=2, parent="conf:90011", relevance=0.8)

    runs = [run("h2s2/s1", "conf:90011"), run("h2s2/s2", "people:u-1003")]
    missed = uncited_followed(runs, {"conf:90011"}, {chain.name: chain})
    assert [r.step_id for r in missed] == ["h2s2/s2"]
    assert uncited_followed(runs, {"conf:90011", "people:u-1003"}, {chain.name: chain}) == []


def test_accept_refuses_what_is_not_a_recipe_name(tmp_path: Any) -> None:
    from oracle.log.dream import accept

    with pytest.raises(ValueError, match="not a recipe name"):
        accept(tmp_path, tmp_path / "tools", "../../x")
