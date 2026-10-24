from datetime import datetime
from typing import Any

import pytest
from claude_agent_sdk import ResultMessage
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from oracle.llm import claude_code
from oracle.llm.models import resolve_model
from oracle.plan.models import Plan, Step, plan_errors
from oracle.plan.planner import Shortlist, build_planner, make_plan, render_request
from oracle.rank.index import Ranker
from oracle.registry.loader import Registry


def scripted(*plans: dict[str, Any]) -> tuple[FunctionModel, list[list[ModelMessage]]]:
    """A model that answers with each plan in turn, recording what it was sent."""
    seen: list[list[ModelMessage]] = []
    queue = list(plans)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, queue.pop(0))])

    return FunctionModel(respond), seen


def plan(op: str, **params: Any) -> dict[str, Any]:
    return {"interpretation": "i", "steps": [{"id": "s1", "op": op, "params": params}]}


def test_plan_errors_catch_unknown_ops_params_and_limits(fake_registry: Registry) -> None:
    allowed = set(fake_registry.op_refs())
    bad = Plan(
        interpretation="x",
        steps=[
            Step(id="s1", op="acli.jira.search", params={"jql": "x", "bogus": 1}),
            Step(id="s1", op="kubectl.get", after=["s9"]),
        ],
    )
    errors = plan_errors(bad, fake_registry, allowed, max_steps=1)
    assert any("at most 1" in e for e in errors)
    assert any("unique" in e for e in errors)
    assert any("unknown params" in e for e in errors)
    assert any("not in the shortlist" in e for e in errors)
    dash = Plan(interpretation="x", steps=[Step(id="s1", op="workiq.people.lookup",
                                               params={"name": "-rf"})])  # fmt: skip
    assert any("cannot start with '-'" in e for e in plan_errors(dash, fake_registry, allowed, 1))


async def test_planner_retries_with_validation_feedback(fake_registry: Registry) -> None:
    model, seen = scripted(
        plan("acli.jira.search", query="oops"),
        plan("acli.jira.search", jql='text ~ "transfer"'),
    )
    shortlist = Shortlist(Ranker(fake_registry).rank("tickets about transfer"))
    outcome = await make_plan(
        build_planner(model),
        question="tickets about transfer",
        intent=None,
        shortlist=shortlist,
        registry=fake_registry,
        max_steps=1,
        now=datetime.now().astimezone(),
    )
    assert outcome.plan.steps[0].params == {"jql": 'text ~ "transfer"'}
    assert outcome.usage.model_calls == 2
    retry_prompt = claude_code.transcript(seen[1])
    assert "rejected" in retry_prompt and "unknown params ['query']" in retry_prompt


def test_request_lists_shortlist_ops_and_params(fake_registry: Registry) -> None:
    ranker = Ranker(fake_registry)
    q = "who is andrew neville"
    shortlist = Shortlist(ranker.rank(q), ranker.rank_recipes(q))
    text = render_request(
        "who is andrew neville", "onboarding", shortlist, fake_registry, 1, datetime(2026, 1, 2)
    )
    assert "Why they are asking: onboarding" in text
    assert "- workiq.people.lookup:" in text
    assert "name: str, required" in text
    assert "Step limit: 1" in text
    assert text.index("- people.profile:") < text.index("Tools:")  # recipes come first


def test_resolve_model_routes_claude_code_prefix() -> None:
    model = resolve_model("claude-code:sonnet")
    assert isinstance(model, FunctionModel) and model.model_name == "claude-code:sonnet"
    assert resolve_model("openai:bedrock/anthropic.claude") == "openai:bedrock/anthropic.claude"


def result_message(**over: Any) -> ResultMessage:
    base: dict[str, Any] = dict(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="s",
        usage={"input_tokens": 10, "cache_read_input_tokens": 5, "output_tokens": 3},
    )
    return ResultMessage(**(base | over))


async def test_claude_code_structured_output_becomes_validated_plan(
    monkeypatch: pytest.MonkeyPatch, fake_registry: Registry
) -> None:
    sent: list[Any] = []

    async def fake_query(*, prompt: str, options: Any):
        sent.append((prompt, options))
        yield result_message(structured_output=plan("workiq.calendar.today"))

    monkeypatch.setattr("claude_agent_sdk.query", fake_query)
    agent: Agent[Any, Plan] = Agent(claude_code.claude_code_model("haiku"), output_type=Plan)
    result = await agent.run("what is on today?", instructions="plan it")
    assert result.output.steps[0].op == "workiq.calendar.today"
    prompt, options = sent[0]
    assert prompt == "what is on today?"
    assert options.model == "haiku" and options.tools == [] and options.setting_sources == []
    assert options.output_format["type"] == "json_schema"
    assert "steps" in options.output_format["schema"]["properties"]
    assert result.usage.input_tokens == 15 and result.usage.output_tokens == 3


async def test_claude_code_error_result_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_query(*, prompt: str, options: Any):
        yield result_message(is_error=True, result="rate limited")

    monkeypatch.setattr("claude_agent_sdk.query", fake_query)
    agent: Agent[Any, str] = Agent(claude_code.claude_code_model("sonnet"))
    with pytest.raises(claude_code.ClaudeCodeError, match="rate limited"):
        await agent.run("hi")


def test_transcript_includes_prior_answer_and_ignores_plain_tool_returns() -> None:
    from pydantic_ai.messages import ToolReturnPart, UserPromptPart

    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart("plan this")]),
        ModelResponse(parts=[ToolCallPart("final_result", {"steps": []})]),
        ModelRequest(parts=[ToolReturnPart("final_result", "Final result processed.")]),
    ]
    text = claude_code.transcript(messages)
    assert text.startswith("plan this")
    assert '{"steps": []}' in text
    assert "processed" not in text


async def test_claude_code_call_times_out_instead_of_hanging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    async def stuck(*, prompt: str, options: Any):
        await asyncio.sleep(10)
        yield result_message()

    monkeypatch.setattr("claude_agent_sdk.query", stuck)
    monkeypatch.setattr(claude_code, "CALL_TIMEOUT_S", 0.05)
    agent: Agent[Any, str] = Agent(claude_code.claude_code_model("sonnet"))
    with pytest.raises(claude_code.ClaudeCodeError, match="no result in"):
        await agent.run("hi")
