"""M3: breadth, oracle report and the skills loader."""

import time
from pathlib import Path

from oracle.ask import AskRequest, Models, ask
from oracle.config import REPO_ROOT, Config
from oracle.exec.executor import OpResult
from oracle.log.report import build_report, write_candidates
from oracle.log.store import CallRecord, open_store
from oracle.rank.index import Ranker
from oracle.registry.loader import Registry, load_registry
from oracle.skills.loader import load_skills, render, select
from tests.conftest import DONE, REGISTRY, echo_distiller, plan, scripted


def test_breadth_shortlists_every_tool_in_a_small_registry() -> None:
    reg = load_registry(REGISTRY)
    ranked = Ranker(reg).rank("everything about the ingestion gateway", breadth=True)
    assert {t.tool for t in ranked} == set(reg.tools)
    assert len(Ranker(reg).rank("everything about the ingestion gateway")) == 5


async def test_breadth_flag_forces_a_breadth_plan(
    home_cfg: Config, fake_registry: Registry
) -> None:
    step = {"id": "s1", "op": "workiq.people.lookup", "params": {"name": "andrew neville"}}
    m = Models(scripted(plan(step)), scripted(DONE), echo_distiller())
    answer = await ask(AskRequest("andrew", breadth=True), home_cfg, fake_registry, m)
    assert answer.breadth


async def test_report_finds_requeries_misses_and_feedback(tmp_path: Path) -> None:
    async with open_store(tmp_path / "log.sqlite") as store:
        now = time.time()
        ranked = [{"tool": "workiq", "score": 1}, {"tool": "code-graph", "score": 0}]
        await store.record_call(
            CallRecord(
                "c1", "conv", "who owns the gateway", None, "standard", ranked, None, wall_ms=1
            )
        )
        await store.record_call(
            CallRecord(
                "c2",
                "conv",
                "gateway owner team",
                None,
                "standard",
                ranked,
                None,
                wall_ms=1,
                replans=2,
            )
        )
        hit = OpResult(op="acli.jira.search", params={})
        hit.records, hit.pointers = [{"key": "A-1"}], ["jira:A-1"]
        await store.record_op("c2", "conv", "s1", hit)
        await store.record_feedback("c1", False, "missed the jira ticket", "agent")
        report = await build_report(store.db, now - 60)
    assert report.calls == 2 and report.requeries == [
        ("who owns the gateway", "gateway owner team")
    ]
    assert report.replanned_twice == ["gateway owner team"]
    assert report.ranker_misses[0]["expect"] == ["acli"]
    assert report.feedback == {"not useful": 1} and report.notes == ["missed the jira ticket"]
    path = tmp_path / "candidates.yaml"
    assert write_candidates(report, path) == 1 and "acli" in path.read_text()
    assert not (REPO_ROOT / "ranker_eval.yaml").read_text().count("gateway owner team")


def test_skills_load_and_select_by_role_tool_and_question() -> None:
    skills = load_skills(REPO_ROOT / "skills")
    reg = load_registry(REGISTRY)
    names = lambda chosen: [s.name for s in chosen]  # noqa: E731
    incident = select(
        skills, "planner", ["argocd", "kubectl", "es"], reg, "billing-api is down after the deploy"
    )
    assert "kubectl-selectors" in names(incident) and "incident-triage" in names(incident)
    assert len(incident) <= 4
    assert names(select(skills, "ingest", [], reg, "x")) == ["manifest-authoring"]
    assert names(select(skills, "planner", ["workiq"], reg, "who is jon")) == ["people-lookup"]
    assert all(len(s.body) <= 3200 for s in skills)
    assert render([]) == "" and "### jql" in render(
        select(skills, "planner", ["acli"], reg, "tickets")
    )


def test_refine_must_name_real_tools() -> None:
    from oracle.plan.distill import refine_problems

    tools = {"kubectl", "acli"}
    assert refine_problems(['oracle "x" --scope kubectl'], tools) == []
    assert refine_problems(['oracle "x" --scope cluster'], tools)
    assert refine_problems(["search jira"], tools)
