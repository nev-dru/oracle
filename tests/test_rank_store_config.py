from pathlib import Path

import pytest

from oracle.config import ConfigError, load_config
from oracle.exec.executor import OpResult
from oracle.log.store import CallRecord, open_store
from oracle.rank.index import Ranker, tokens
from oracle.registry.loader import load_registry
from tests.conftest import REGISTRY


@pytest.fixture(scope="module")
def ranker() -> Ranker:
    return Ranker(load_registry(REGISTRY))


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("who is andrew neville", "workiq"),
        ("important events from today", "workiq"),
        ("get ticket PLAT-412", "acli"),
        ("design doc for the gateway", "acli"),
        ("callers of transfer_ownership", "code-graph"),
        ("callers of x", "code-graph"),
        ("who calls transfer_ownership", "code-graph"),
    ],
)
def test_top_tool_for_seed_shapes(ranker: Ranker, query: str, expected: str) -> None:
    assert ranker.rank(query)[0].tool == expected


def test_unknown_person_still_reaches_people_source(ranker: Ranker) -> None:
    assert ranker.rank("who is jane smith")[0].tool == "workiq"


def test_code_question_shortlists_both_code_tools(ranker: Ranker) -> None:
    tools = [t.tool for t in ranker.rank("where ownership transfer happens in code")]
    assert {"code-search", "code-graph"} <= set(tools[:3])


def test_shortlist_is_padded_and_grouped(ranker: Ranker) -> None:
    ranked = ranker.rank("how salesforce talks to cloud")
    assert len(ranked) == 5  # padded to top=5 with unmatched tools
    graph = next(t for t in ranked if t.tool == "code-graph")
    assert "code-graph.path_between" in graph.ops and len(graph.ops) == 5


def test_scope_restricts_and_exact_name_pins(ranker: Ranker) -> None:
    assert [t.tool for t in ranker.rank("anything at all", scope=["acli"])] == ["acli"]
    pinned = ranker.rank("use code-search for the token refresh")
    assert pinned[0].tool == "code-search" and pinned[0].pinned


def test_tokens_drop_stopwords_and_fold_plurals() -> None:
    assert tokens("Who are the owners of tickets?") == ["who", "owner", "ticket"]
    assert tokens("when is my meeting") == ["meeting"]


async def test_store_round_trip_and_session_scoping(tmp_path: Path) -> None:
    result = OpResult(op="acli.jira.get", params={"key": "PLAT-1"})
    result.records, result.pointers = [{"key": "PLAT-1", "summary": "s"}], ["jira:PLAT-1"]
    async with open_store(tmp_path / "log.sqlite") as store:
        await store.record_call(CallRecord("c1", "sess-a", "q", None, "quick", [], None, wall_ms=5))
        await store.record_op("c1", "sess-a", "s1", result)
        hit = await store.lookup("jira:PLAT-1", "sess-a")
        assert hit is not None and hit.record["summary"] == "s" and hit.call_id == "c1"
        assert await store.lookup("jira:PLAT-1", "sess-b") is None
        assert await store.lookup("jira:PLAT-1", None) is not None
        assert await store.lookup("jira:NOPE", None) is None


def test_config_env_overrides_file_and_profile_paths(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.toml").write_text(
        '[models]\nplanner = "claude-code:sonnet"\nextract = "claude-code:haiku"\n'
    )
    cfg = load_config(
        {"ORACLE_HOME": str(home), "ORACLE_MODEL_PLANNER": "test", "ORACLE_PROFILE": "home"}
    )
    assert cfg.model("planner") == "test"
    assert cfg.model("extract") == "claude-code:haiku"
    with pytest.raises(ConfigError, match="ORACLE_MODEL_DEEP"):
        cfg.model("deep")
    assert cfg.profile is not None and cfg.profile.registry is not None
    env = cfg.tool_env({"PATH": "/usr/bin"})
    assert env["PATH"].split(":")[0].endswith("adapters/bin")
    assert env["VEX_DB"].endswith(".fixtures/vex.db") and "${" not in env["VEX_DB"]


def test_unknown_profile_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config({"ORACLE_HOME": str(tmp_path), "ORACLE_PROFILE": "nowhere"})
