"""Completeness check (Sonnet): do the results answer the question, and if not, which steps
would close the gaps? The within-question judgment: a synonym retry, a second naming
convention, a sibling tool when one returned nothing.
"""

from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

from oracle.budget import Budget
from oracle.exec.dag import StepRun
from oracle.plan.digest import DATA_RULE, digest
from oracle.plan.models import Plan, Step, step_errors
from oracle.plan.planner import PLAN_RETRIES, Shortlist, prompt, render_shortlist
from oracle.registry.loader import Registry
from oracle.timewin import TimeWindow


class Check(BaseModel):
    complete: bool = Field(description="the results answer every part of the question")
    sufficient: bool = Field(
        default=False, description="plainly answered already; stop even if more could be found"
    )
    gaps: list[str] = Field(
        default_factory=list, description="parts of the question no result covers, briefly"
    )
    next_steps: list[Step] = Field(
        default_factory=list,
        description="new steps that would close the gaps; new ids; may bind to earlier steps",
    )


@dataclass
class CheckContext:
    registry: Registry
    allowed: set[str]
    max_steps: int
    earlier: set[str]


def build_checker(model: Model | str) -> Agent[CheckContext, Check]:
    agent = Agent(
        model,
        deps_type=CheckContext,
        output_type=Check,
        instructions=prompt("check") + "\n" + DATA_RULE,
        retries=PLAN_RETRIES,
    )

    @agent.output_validator
    def _valid(ctx: RunContext[CheckContext], check: Check) -> Check:
        d = ctx.deps
        if d.max_steps == 0:  # the last check: steps would not run, so do not retry over them
            return check
        errors = step_errors(check.next_steps, d.registry, d.allowed, d.max_steps, d.earlier)
        if errors:
            raise ModelRetry("Fix these problems in next_steps:\n- " + "\n- ".join(errors))
        return check

    return agent


async def run_check(
    agent: Agent[CheckContext, Check],
    *,
    question: str,
    intent: str | None,
    plan: Plan,
    window: TimeWindow | None,
    runs: list[StepRun],
    shortlist: Shortlist,
    registry: Registry,
    max_steps: int,
    budget: Budget,
    earlier: set[str],
    guidance: str = "",
) -> Check:
    """`earlier` is every step id used so far: plan steps, recipe steps and their stages."""
    lines = [f"Question: {question}"]
    if intent:
        lines.append(f"Why they are asking: {intent}")
    lines += [f"Interpretation: {plan.interpretation}", f"A complete answer has: {plan.expect}"]
    if window:
        lines.append(f"Time window: {window.describe()}")
    lines += ["", "Steps run so far and their results:", digest(runs, registry), ""]
    if max_steps:
        lines.append(f"You may add at most {max_steps} steps.")
    else:
        lines.append("No more steps can run: judge completeness and list gaps; next_steps empty.")
    lines += render_shortlist(shortlist, registry)
    if guidance:
        lines += ["", guidance]
    deps = CheckContext(registry, shortlist.allowed, max_steps, earlier | {r.step_id for r in runs})
    result = await budget.model(agent, "\n".join(lines), deps=deps)
    return result.output
