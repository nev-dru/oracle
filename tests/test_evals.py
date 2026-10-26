from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from oracle import cli
from oracle.ask import Models
from oracle.config import REPO_ROOT, Config
from oracle.evals import (
    CLAUSES,
    NO_PROFILE,
    clause_failures,
    judge,
    load_cases,
    ranker_hit,
    run_case,
)
from oracle.registry.loader import Registry
from oracle.render import Answer, Finding
from tests.conftest import DONE, echo_distiller, plan, scripted


def answer(**over: object) -> Answer:
    base = Answer(
        call_id="c",
        session="s",
        level="standard",
        findings=[
            Finding("a", "people:u-1", "workiq.people.lookup"),
            Finding("b", "teams:19:x", "workiq.teams.search"),
            Finding("c", "jira:P-1", "acli.jira.search"),
        ],
        searched=["acli", "workiq"],
        not_searched=[],
        unavailable=["argocd", "kubectl"],
        failures=[],
        confidence="medium",
        gaps=[],
        trace=["time: today = a .. b (PDT)", "s1 workiq.calendar.today → 3 results"],
        breadth=True,
        ops_run=["workiq.people.lookup", "workiq.teams.search", "acli.jira.search"],
        recipes=["people.profile"],
        time_window="today = a .. b (PDT)",
        no_match=["workiq.mail.search"],
        shortlist=["workiq", "acli", "code-graph"],
        scored=["workiq", "acli"],
    )
    return replace(base, **over)  # type: ignore[arg-type]


def test_every_golden_clause_has_a_checker() -> None:
    golden = REPO_ROOT / "golden.yaml"
    named = {p for c in yaml.safe_load(golden.read_text())["cases"] for p in c.get("profiles", {})}
    for profile in named | {NO_PROFILE}:
        for case in load_cases(golden, profile=profile):
            assert set(case["expect"]) <= set(CLAUSES), (profile, case["id"])


def test_the_fixture_pointers_are_checked_at_home_only() -> None:
    golden = REPO_ROOT / "golden.yaml"
    home = {c["id"]: c for c in load_cases(golden, profile="home")}["crawl-gateway-5xx"]
    work = {c["id"]: c for c in load_cases(golden)}["crawl-gateway-5xx"]
    assert "jira:OPS-1206" in home["expect"]["pointers"] and "Priya" in home["judge"]
    assert "pointers" not in work["expect"] and "Priya" not in work["judge"]
    assert work["unset_profile"] == NO_PROFILE and "unset_profile" not in home


def golden_with(tmp_path: Path, **case: object) -> Path:
    path = tmp_path / "golden.yaml"
    base = {"id": "c", "query": "q", "expect": {"findings_min": 1}, "judge": "shared"}
    path.write_text(yaml.safe_dump({"cases": [{**base, **case}]}))
    return path


def test_a_profile_block_adds_its_clauses_and_replaces_the_rubric(tmp_path: Path) -> None:
    block = {"expect": {"pointers": ["jira:P-1"]}, "judge": "home rubric"}
    golden = golden_with(tmp_path, profiles={"home": block})
    (case,) = load_cases(golden, profile="home")
    assert case["expect"] == {"findings_min": 1, "pointers": ["jira:P-1"]}
    assert case["judge"] == "home rubric"
    (bare,) = load_cases(golden_with(tmp_path, profiles={"home": {"expect": {}}}), profile="home")
    assert bare["judge"] == "shared"


def test_a_typo_in_a_profile_block_is_refused(tmp_path: Path) -> None:
    golden = golden_with(tmp_path, profiles={"home": {"pointers": ["jira:P-1"]}})
    with pytest.raises(ValueError, match=r"profiles\.home has \['pointers'\]"):
        load_cases(golden, profile="other")


def test_clauses_pass_on_a_matching_answer(fake_registry: Registry) -> None:
    expect = {
        "breadth": True,
        "tools_any": ["workiq.teams.search", "argocd.app.history"],
        "tools_all": ["workiq.people.lookup"],
        "tools_min": 1,
        "findings_min": 3,
        "grouped_by_source": True,
        "coverage_names_unconfigured": True,
        "resolved_time_range_in_trace": True,
        "pointers": ["jira:P-1"],
        "pointer_prefixes_any": ["teams:"],
        "recipe": "people.profile",
        "confidence_at_least": "medium",
        "coverage_names_no_match_sources": True,
    }
    assert clause_failures(expect, answer(), fake_registry) == []


@pytest.mark.parametrize(
    ("expect", "over", "needle"),
    [
        ({"breadth": True}, {"breadth": False}, "breadth"),
        ({"tools_all": ["code-graph.expand"]}, {}, "did not run"),
        ({"tools_any": ["x.y"]}, {}, "none of"),
        ({"tools_any": ["acli.jira.search", "x.y"], "tools_min": 2}, {}, "only 1"),
        ({"findings_min": 5}, {}, "3 findings"),
        ({"recipe": "today.events"}, {}, "not used"),
        ({"confidence_at_least": "high"}, {}, "confidence medium"),
        ({"pointers": ["jira:NOPE"]}, {}, "missing pointers"),
        ({"pointer_prefixes_any": ["code:"]}, {}, "no finding pointer"),
        ({"resolved_time_range_in_trace": True}, {"time_window": None}, "time range"),
        (
            {"coverage_names_unconfigured": True, "tools_any": ["argocd.app.history"]},
            {"unavailable": []},
            "unavailable",
        ),
        ({"bogus_clause": 1}, {}, "unknown clause"),
    ],
)
def test_clauses_fail_on_a_clear_mismatch(
    fake_registry: Registry, expect: dict, over: dict, needle: str
) -> None:
    failures = clause_failures(expect, answer(**over), fake_registry)
    assert failures and needle in failures[0]


def test_grouping_is_checked_on_the_findings_order(fake_registry: Registry) -> None:
    mixed = [
        Finding("a", "people:u-1", "workiq.people.lookup"),
        Finding("c", "jira:P-1", "acli.jira.search"),
        Finding("b", "teams:19:x", "workiq.teams.search"),
    ]
    assert clause_failures({"grouped_by_source": True}, answer(findings=mixed), fake_registry)


def test_ranker_hit_counts_scored_registered_tools_not_padding(fake_registry: Registry) -> None:
    case = {"expect": {"tools_any": ["workiq.calendar.today", "kubectl.events"]}}
    assert ranker_hit(case, answer(), fake_registry)
    assert not ranker_hit(case, answer(scored=["acli"]), fake_registry)
    every = {"expect": {"tools_all": ["workiq.people.lookup", "acli.jira.search"]}}
    assert ranker_hit(every, answer(), fake_registry)
    assert not ranker_hit(
        every, answer(scored=["workiq"], shortlist=["workiq", "acli"]), fake_registry
    )


async def test_judge_and_case_run_with_scripted_models(
    home_cfg: Config, fake_registry: Registry, tmp_path: Path
) -> None:
    verdict = await judge(scripted({"verdict": "pass", "reason": "ok"}), "q", "r", "answer")
    assert verdict.verdict == "pass"
    case = next(
        c for c in load_cases(REPO_ROOT / "golden.yaml") if c["id"] == "who-is-andrew-neville"
    )
    models = Models(
        planner=scripted(
            plan({"id": "s1", "op": "people.profile", "params": {"name": "Andrew Neville"}})
        ),
        checker=scripted(DONE),
        distiller=echo_distiller(),
    )
    result = await run_case(
        case,
        home_cfg,
        fake_registry,
        models,
        scripted({"verdict": "fail", "reason": "invented role"}),
    )
    assert result.failures == ["judge: invented role"]
    passing = await run_case(case, home_cfg, fake_registry, models, None)
    assert passing.passed, passing.failures
    assert passing.ranker_hit
    home_only = golden_with(tmp_path, **case, profiles={"home": {"expect": {}}})
    (at_work,) = load_cases(home_only)
    unset = await run_case(at_work, home_cfg, fake_registry, models, None)
    assert unset.failures == [
        "profiles: no 'default' block, so this profile's data checks are unset"
    ]
    assert unset.answer, "the answer is kept, to write the profile's pointers from"


def test_feedback_cli_records_and_rejects_unknown_calls(home_cfg: Config, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import asyncio

    from oracle.log.store import CallRecord, open_store

    monkeypatch.setenv("ORACLE_HOME", str(home_cfg.home))
    monkeypatch.setenv("ORACLE_PROFILE", "home")

    async def seed() -> None:
        async with open_store(home_cfg.log_path) as store:
            await store.record_call(CallRecord("c1", "s", "q", None, "quick", [], None, wall_ms=1))

    asyncio.run(seed())
    runner = CliRunner()
    ok = runner.invoke(cli.app, ["feedback", "c1", "--useful", "no", "--note", "missed jira"])
    assert ok.exit_code == 0 and "recorded" in ok.output
    missing = runner.invoke(cli.app, ["feedback", "nope", "--useful", "yes"])
    assert missing.exit_code == 2
