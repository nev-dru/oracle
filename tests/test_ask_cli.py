import json
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from typer.testing import CliRunner

from oracle import cli
from oracle.ask import AskRequest, Models, ask
from oracle.config import Config
from oracle.log.store import open_store
from oracle.registry.loader import Registry
from oracle.render import to_json, to_markdown
from tests.conftest import DONE, echo_distiller, scripted


def planner_choosing(op: str, **params: Any) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        body = {"interpretation": "i", "steps": [{"id": "s1", "op": op, "params": params}]}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, body)])

    return FunctionModel(respond)


async def test_quick_ask_end_to_end(home_cfg: Config, fake_registry: Registry) -> None:
    model = planner_choosing("acli.jira.search", jql="assignee = currentUser() AND updated >= -1d")
    req = AskRequest("what am I working on today", level="quick", session="conv-1")
    answer = await ask(req, home_cfg, fake_registry, model)

    assert answer.searched == ["acli"] and answer.not_searched == ["workiq"]
    assert {"code-graph", "code-search"} <= set(answer.unavailable)  # inactive in this registry
    assert answer.findings and all(f.pointer.startswith("jira:") for f in answer.findings)
    md = to_markdown(answer)
    for section in ("## Findings", "## Coverage", "## Trace", "Not searched: workiq"):
        assert section in md
    assert "Not available here: " in md and "code-graph" in md
    assert json.loads(to_json(answer))["coverage"]["searched"] == ["acli"]

    async with open_store(home_cfg.log_path) as store:
        hit = await store.lookup(answer.findings[0].pointer, "conv-1")
    assert hit is not None and hit.op == "acli.jira.search"


async def test_logged_level_is_the_level_that_ran(
    home_cfg: Config, fake_registry: Registry
) -> None:
    models = Models(
        planner=planner_choosing("workiq.people.lookup", name="andrew neville"),
        checker=scripted(DONE),
        distiller=echo_distiller(),
    )
    answer = await ask(AskRequest("who is andrew neville"), home_cfg, fake_registry, models)
    assert answer.findings[0].pointer == "people:u-1001"
    async with (
        open_store(home_cfg.log_path) as store,
        store.db.execute(
            "SELECT level, ran_level FROM calls WHERE call_id = ?", (answer.call_id,)
        ) as cur,
    ):
        row = await cur.fetchone()
    assert row is not None and (row[0], row[1]) == ("standard", "standard")


async def test_failed_op_is_reported_in_coverage(home_cfg: Config, fake_registry: Registry) -> None:
    model = planner_choosing("acli.jira.get", key="NOPE-1")
    answer = await ask(AskRequest("ticket NOPE-1", level="quick"), home_cfg, fake_registry, model)
    assert answer.searched == [] and answer.confidence == "low"
    assert answer.failures and "not found" in answer.failures[0]
    assert "no results" in answer.gaps


@pytest.fixture
def runner_env(home_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    for var in ("ORACLE_REGISTRY", "ORACLE_TZ", "ORACLE_MODEL_PLANNER", "ORACLE_MODEL_EXTRACT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ORACLE_HOME", str(home_cfg.home))
    monkeypatch.setenv("ORACLE_PROFILE", "home")
    return CliRunner()


def test_cli_tool_list_and_test_on_fakes(runner_env: CliRunner) -> None:
    listed = runner_env.invoke(cli.app, ["tool", "list"])
    assert listed.exit_code == 0 and "acli.jira.search" in listed.output
    tested = runner_env.invoke(cli.app, ["tool", "test", "acli", "workiq"])
    assert tested.exit_code == 0, tested.output
    assert "8/8 examples passed" in tested.output


def test_cli_tool_test_refuses_unknown_names(
    runner_env: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = runner_env.invoke(cli.app, ["tool", "test", "acl"])
    assert result.exit_code == 2 and "unknown tools ['acl']" in result.output
    monkeypatch.setenv("ORACLE_REGISTRY", "/nope")
    missing = runner_env.invoke(cli.app, ["tool", "test", "acli"])
    assert missing.exit_code == 2 and "does not exist" in missing.output


def test_cli_ask_without_model_is_a_clear_error(runner_env: CliRunner) -> None:
    result = runner_env.invoke(cli.app, ["ask", "who is andrew neville"])
    assert result.exit_code == 2
    assert "ORACLE_MODEL_PLANNER" in result.output


def test_cli_expand_unknown_pointer(runner_env: CliRunner) -> None:
    result = runner_env.invoke(cli.app, ["expand", "jira:NOPE-1"])
    assert result.exit_code == 2 and "no record" in result.output


def test_main_defaults_to_ask(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []
    monkeypatch.setattr(cli, "app", lambda: seen.append(list(cli.sys.argv)))
    monkeypatch.setattr(cli.sys, "argv", ["oracle", "who is x", "--level", "quick"])
    cli.main()
    monkeypatch.setattr(cli.sys, "argv", ["oracle", "tool", "list"])
    cli.main()
    assert seen == [["oracle", "ask", "who is x", "--level", "quick"], ["oracle", "tool", "list"]]


async def test_planner_failure_becomes_coverage_not_traceback(
    home_cfg: Config, fake_registry: Registry
) -> None:
    def broken(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError("proxy returned 503")

    answer = await ask(
        AskRequest("who is andrew neville", level="quick"),
        home_cfg,
        fake_registry,
        FunctionModel(broken),
    )
    assert answer.failures == ["planner: RuntimeError: proxy returned 503"]
    assert answer.searched == [] and answer.not_searched == ["acli", "workiq"]
    assert answer.confidence == "low"


async def test_empty_scope_answers_without_a_model_call(
    home_cfg: Config, fake_registry: Registry
) -> None:
    def never(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError("planner must not be called")

    req = AskRequest("anything", level="quick", scope=["no-such-tool"])
    answer = await ask(req, home_cfg, fake_registry, FunctionModel(never))
    assert "no active tool matches the question or --scope" in answer.gaps


async def test_spilled_output_pointer_expands(
    home_cfg: Config, tmp_path, runner_env: CliRunner
) -> None:
    from oracle.registry.loader import Registry as Reg
    from tests.conftest import script_tool

    body = (
        "import json\n"
        "print(json.dumps({'items': [{'id': i, 'blob': 'y' * 600} for i in range(30)]}))"
    )
    tool = script_tool(tmp_path, body, limits={"max_output_kb": 2})
    answer = await ask(
        AskRequest("big", level="quick"),
        home_cfg,
        Reg(tools={"t": tool}),
        planner_choosing("t.run"),
    )
    spill_gap = next(g for g in answer.gaps if "truncated" in g)
    pointer = spill_gap.split("`")[1]
    assert pointer.startswith("file:")
    result = runner_env.invoke(cli.app, ["expand", pointer])
    assert result.exit_code == 0, result.output
    shown = json.loads(result.output)
    assert shown["bytes"] > 2048 and shown["truncated"] is True
    assert shown["head"].startswith('{"items"')
    outside = runner_env.invoke(cli.app, ["expand", "file:/etc/hosts"])
    assert outside.exit_code == 2 and "not a spill file" in outside.output
