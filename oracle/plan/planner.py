"""Planner agent: question + ranked shortlist -> validated Plan (PydanticAI, model from config)."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

from oracle.plan.models import Plan, plan_errors
from oracle.rank.index import RankedTool
from oracle.registry.loader import Registry
from oracle.registry.schema import Op, Param, Tool

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
PLAN_RETRIES = 2


@dataclass
class PlanContext:
    registry: Registry
    allowed: set[str]
    max_steps: int


@dataclass
class PlanOutcome:
    plan: Plan
    tokens_in: int
    tokens_out: int
    model_calls: int


def build_planner(model: Model | str) -> Agent[PlanContext, Plan]:
    agent = Agent(
        model,
        deps_type=PlanContext,
        output_type=Plan,
        instructions=(PROMPTS / "planner.md").read_text(),
        retries=PLAN_RETRIES,
    )

    @agent.output_validator
    def _valid(ctx: RunContext[PlanContext], plan: Plan) -> Plan:
        errors = plan_errors(plan, ctx.deps.registry, ctx.deps.allowed, ctx.deps.max_steps)
        if errors:
            raise ModelRetry("Fix these problems:\n- " + "\n- ".join(errors))
        return plan

    return agent


async def make_plan(
    agent: Agent[PlanContext, Plan],
    *,
    question: str,
    intent: str | None,
    shortlist: list[RankedTool],
    registry: Registry,
    max_steps: int,
    now: datetime,
) -> PlanOutcome:
    allowed = {ref for t in shortlist for ref in t.ops}
    prompt = render_request(question, intent, shortlist, registry, max_steps, now)
    result = await agent.run(prompt, deps=PlanContext(registry, allowed, max_steps))
    usage = result.usage
    return PlanOutcome(
        plan=result.output,
        tokens_in=usage.input_tokens,
        tokens_out=usage.output_tokens,
        model_calls=usage.requests,
    )


def render_request(
    question: str,
    intent: str | None,
    shortlist: list[RankedTool],
    registry: Registry,
    max_steps: int,
    now: datetime,
) -> str:
    lines = [f"Question: {question}"]
    if intent:
        lines.append(f"Why they are asking: {intent}")
    lines += [
        f"Current time: {now.isoformat(timespec='minutes')} ({now.tzname()})",
        f"Step limit: {max_steps}",
        "",
        "Shortlist:",
    ]
    for ranked in shortlist:
        lines += render_tool(registry.tools[ranked.tool])
    return "\n".join(lines)


def render_tool(tool: Tool) -> list[str]:
    lines = ["", f"### {tool.name}: {tool.description}"]
    if tool.when:
        lines.append(f"when: {tool.when}")
    if tool.not_when:
        lines.append(f"not_when: {tool.not_when}")
    for op in tool.ops:
        lines += render_op(tool.name, op)
    return lines


def render_op(tool: str, op: Op) -> list[str]:
    lines = [f"- {tool}.{op.name}: {op.when}"]
    lines += [f"    {name}: {describe(p)}" for name, p in op.params.items()]
    return lines


def describe(p: Param) -> str:
    bits = [p.type, "required" if p.required else "optional"]
    if p.default is not None:
        bits.append(f"default {p.default}")
    if p.max is not None:
        bits.append(f"max {p.max}")
    if p.enum:
        bits.append("one of " + "|".join(p.enum))
    return ", ".join(bits) + (f". {p.desc}" if p.desc else "")
