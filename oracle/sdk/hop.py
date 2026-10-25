"""The crawl's hop planner (Sonnet): one call turns every admitted term into lookups. Terms are
tool output, so they arrive inside the data block, defanged, and every step is validated by the
same `step_errors` as a standard plan: planted text cannot steer the crawl outside the registry
or past read-only."""

from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

from oracle.plan.digest import DATA_CLOSE, DATA_OPEN, DATA_RULE, defang
from oracle.plan.models import Step, step_errors
from oracle.plan.planner import PLAN_RETRIES, Shortlist, Usage, prompt, render_shortlist
from oracle.registry.loader import Registry
from oracle.sdk.extract import Term, key

STEPS_PER_TERM = 2


class HopStep(Step):
    term: str = Field(description="the term this step looks up, copied exactly")


class HopPlan(BaseModel):
    steps: list[HopStep] = Field(default_factory=list)


@dataclass
class HopContext:
    registry: Registry
    allowed: set[str]
    terms: set[str]
    earlier: set[str]
    max_steps: int


def build_hop_planner(model: Model | str) -> Agent[HopContext, HopPlan]:
    agent = Agent(
        model,
        deps_type=HopContext,
        output_type=HopPlan,
        instructions=prompt("hop") + "\n" + DATA_RULE,
        retries=PLAN_RETRIES,
    )

    @agent.output_validator
    def _valid(ctx: RunContext[HopContext], plan: HopPlan) -> HopPlan:
        d = ctx.deps
        steps = [Step(id=s.id, op=s.op, params=s.params, after=s.after) for s in plan.steps]
        errors = step_errors(steps, d.registry, d.allowed, d.max_steps, d.earlier)
        errors += [
            f"{s.id}: term {s.term!r} is not one of the terms given"
            for s in plan.steps
            if key(s.term) not in d.terms
        ]
        if errors:
            raise ModelRetry("Fix these problems:\n- " + "\n- ".join(errors))
        return plan

    return agent


async def plan_hop(
    agent: Agent[HopContext, HopPlan],
    *,
    hop: int,
    question: str,
    terms: list[Term],
    shortlist: Shortlist,
    registry: Registry,
    earlier: set[str],
    usage: Usage,
) -> HopPlan:
    listed = "\n".join(f"- {defang(t.value)}: {defang(t.description)}" for t in terms)
    lines = [
        f"Question: {question}",
        f"Hop: {hop}",
        "",
        "Terms to look up:",
        DATA_OPEN,
        listed,
        DATA_CLOSE,
        "",
        *render_shortlist(shortlist, registry),
    ]
    deps = HopContext(
        registry,
        shortlist.allowed,
        {key(t.value) for t in terms},
        earlier,
        STEPS_PER_TERM * len(terms),
    )
    result = await agent.run("\n".join(lines), deps=deps)
    usage.add(result.usage)
    return result.output
