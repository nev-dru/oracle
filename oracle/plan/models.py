"""The plan the planner emits, and its validation against the registry and the shortlist."""

from typing import Any

from pydantic import BaseModel, Field

from oracle.exec.runners.argv import ArgvError, LeadingDashError, build_argv
from oracle.registry.loader import Registry
from oracle.registry.params import ParamError, validate_params


class Step(BaseModel):
    id: str = Field(description="short unique id, e.g. s1")
    op: str = Field(description="op reference exactly as listed, e.g. acli.jira.search")
    params: dict[str, Any] = Field(default_factory=dict, description="params for the op")
    after: list[str] = Field(default_factory=list, description="ids of steps this one waits for")


class Plan(BaseModel):
    interpretation: str = Field(description="one sentence: what the question is really asking")
    steps: list[Step] = Field(min_length=1)
    breadth: bool = False
    expect: str = Field(default="", description="what a complete answer contains")


def plan_errors(plan: Plan, registry: Registry, allowed: set[str], max_steps: int) -> list[str]:
    """Everything wrong with a plan, phrased so the planner can fix it on retry."""
    errors: list[str] = []
    if len(plan.steps) > max_steps:
        errors.append(f"use at most {max_steps} step(s); got {len(plan.steps)}")
    ids = [s.id for s in plan.steps]
    if len(set(ids)) != len(ids):
        errors.append(f"step ids must be unique: {ids}")
    for step in plan.steps:
        errors += [f"{step.id}: {e}" for e in _step_errors(step, registry, allowed, set(ids))]
    return errors


def _step_errors(step: Step, registry: Registry, allowed: set[str], ids: set[str]) -> list[str]:
    if step.op not in allowed:
        return [f"op {step.op!r} is not in the shortlist; choose from {sorted(allowed)}"]
    errors = [f"after references unknown step {a!r}" for a in step.after if a not in ids]
    _, op = registry.resolve(step.op)
    try:
        build_argv(op.argv, validate_params(op, step.params))
    except LeadingDashError as exc:  # the same checks the executor runs, worded for the planner
        errors.append(
            f"param {exc.name!r} cannot start with '-'; rephrase {exc.value!r} or omit it"
        )
    except (ParamError, ArgvError) as exc:
        errors.append(str(exc))
    return errors
