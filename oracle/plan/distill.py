"""Distill (Haiku): results -> 3-15 findings with pointers, confidence, gaps and Refine commands.

Coverage (searched, not searched, no match, unavailable) is computed from execution, not by the
model; distill only writes finding text and order, gaps, confidence and refinements. A pointer
that is not one of this call's results is rejected, so no finding cites something that was not
found.
"""

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

from oracle.budget import Budget
from oracle.exec.dag import StepRun
from oracle.plan.digest import DATA_RULE, digest
from oracle.plan.planner import PLAN_RETRIES, prompt
from oracle.registry.loader import Registry

MAX_FINDINGS = 15


PER_STEP = 25  # records of one step that distill sees


class Item(BaseModel):
    text: str = Field(description="one short sentence: what this result says that matters")
    pointer: str = Field(description="exactly one pointer copied from the data")


class Distilled(BaseModel):
    findings: list[Item] = Field(default_factory=list, max_length=MAX_FINDINGS)
    confidence: Literal["high", "medium", "low"]
    gaps: list[str] = Field(
        default_factory=list, description="what the question asked that no result covers"
    )
    refine: list[str] = Field(
        default_factory=list,
        max_length=3,
        description='ready-to-run follow-ups, each `oracle "..." [--scope tool]`',
    )


@dataclass
class DistillContext:
    pointers: set[str]
    tools: set[str]


def refine_problems(refine: list[str], tools: set[str]) -> list[str]:
    """Each refinement must be a runnable `oracle` command naming only real tools."""
    out = [
        f"refine must start with `oracle `: {r!r}" for r in refine if not r.startswith("oracle ")
    ]
    for r in refine:
        scope = re.search(r"--scope[ =]+[\"']?([\w,-]+)", r)
        unknown = sorted(set(scope[1].split(",")) - tools) if scope else []
        if unknown:
            out.append(f"--scope names tools that do not exist {unknown}; tools: {sorted(tools)}")
    return out


def build_distiller(model: Model | str) -> Agent[DistillContext, Distilled]:
    agent = Agent(
        model,
        deps_type=DistillContext,
        output_type=Distilled,
        instructions=prompt("distill") + "\n" + DATA_RULE,
        retries=PLAN_RETRIES,
    )

    @agent.output_validator
    def _valid(ctx: RunContext[DistillContext], out: Distilled) -> Distilled:
        bad = [f.pointer for f in out.findings if f.pointer not in ctx.deps.pointers]
        errors = [f"pointers not in the data: {bad}; copy pointers exactly"] if bad else []
        errors += refine_problems(out.refine, ctx.deps.tools)
        if errors:
            raise ModelRetry("Fix these problems:\n- " + "\n- ".join(errors))
        return out

    return agent


async def run_distill(
    agent: Agent[DistillContext, Distilled],
    *,
    question: str,
    interpretation: str,
    when: str,
    breadth: bool,
    check_gaps: list[str],
    runs: list[StepRun],
    registry: Registry,
    budget: Budget,
) -> Distilled:
    lines = [f"Question: {question}", f"Interpretation: {interpretation}", f"Now: {when}"]
    if breadth:
        lines.append("Breadth: cover every source that returned something, most important first.")
    if check_gaps:
        lines.append("Gaps the check found: " + "; ".join(check_gaps))
    lines += ["", digest(runs, registry, per_step=PER_STEP, slice_chars=300)]
    pointers = {p for run in runs if run.result.ok for p in run.result.pointers}
    deps = DistillContext(pointers, set(registry.tools))
    result = await budget.model(agent.run("\n".join(lines), deps=deps))
    return result.output
