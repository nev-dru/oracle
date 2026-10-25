"""Planner agent: question + ranked shortlist -> validated Plan (PydanticAI, model from config)."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pydantic_ai
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

from oracle.budget import Budget
from oracle.plan.models import Plan, plan_errors
from oracle.rank.index import RankedTool
from oracle.registry.loader import Registry
from oracle.registry.schema import Op, Param, Recipe, Tool
from oracle.timewin import TimeWindow

# The oracle owns its stdout (the main agent reads it), so pydantic-ai's first-run banner is off
# for every agent, whichever command builds it.
pydantic_ai.BANNER_ENABLED = False

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
PLAN_RETRIES = 2


@dataclass
class Shortlist:
    tools: list[RankedTool]
    recipes: list[str] = field(default_factory=list)

    @property
    def allowed(self) -> set[str]:
        return {ref for t in self.tools for ref in t.ops} | set(self.recipes)


@dataclass
class PlanContext:
    registry: Registry
    allowed: set[str]
    max_steps: int


def prompt(name: str) -> str:
    return (PROMPTS / f"{name}.md").read_text()


def build_planner(model: Model | str) -> Agent[PlanContext, Plan]:
    agent = Agent(
        model,
        deps_type=PlanContext,
        output_type=Plan,
        instructions=prompt("planner"),
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
    shortlist: Shortlist,
    registry: Registry,
    max_steps: int,
    now: datetime,
    window: TimeWindow | None = None,
    budget: Budget,
    breadth: bool = False,
    guidance: str = "",
) -> Plan:
    text = render_request(question, intent, shortlist, registry, max_steps, now, window)
    if guidance:
        text += "\n\n" + guidance
    if breadth:
        text += (
            "\n\nBreadth requested: one step per listed tool that could hold part of the answer."
        )
    deps = PlanContext(registry, shortlist.allowed, max_steps)
    result = await budget.model(agent, text, deps=deps)
    return result.output.model_copy(update={"breadth": True}) if breadth else result.output


def render_request(
    question: str,
    intent: str | None,
    shortlist: Shortlist,
    registry: Registry,
    max_steps: int,
    now: datetime,
    window: TimeWindow | None = None,
) -> str:
    lines = [f"Question: {question}"]
    if intent:
        lines.append(f"Why they are asking: {intent}")
    lines.append(f"Current time: {now.isoformat(timespec='minutes')} ({now.tzname()})")
    if window:
        lines.append(f"Time window in the question: {window.describe()}")
    lines += [f"Step limit: {max_steps}", ""]
    lines += render_shortlist(shortlist, registry)
    return "\n".join(lines)


def render_shortlist(shortlist: Shortlist, registry: Registry) -> list[str]:
    lines: list[str] = []
    if shortlist.recipes:
        lines += ["Recipes (tested chains; prefer one when its `when` fits the question):"]
        for name in shortlist.recipes:
            lines += render_recipe(registry.recipes[name])
        lines.append("")
    lines.append("Tools:")
    for ranked in shortlist.tools:
        lines += render_tool(registry.tools[ranked.tool])
    return lines


def render_recipe(recipe: Recipe) -> list[str]:
    lines = [f"- {recipe.name}: {recipe.when}"]
    lines += [f"    {name}: {describe(p)}" for name, p in recipe.params.items()]
    return lines


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
