"""The plan the planner emits, and its validation against the registry and the shortlist."""

from typing import Any

from pydantic import BaseModel, Field

from oracle.exec.bindings import BindingError, binding_refs
from oracle.exec.runners.argv import ArgvError, LeadingDashError, build_argv
from oracle.registry.loader import Registry
from oracle.registry.params import ParamError, validate_shape


class Step(BaseModel):
    id: str = Field(description="short unique id, e.g. s1", pattern=r"^[A-Za-z0-9_/-]+$")
    op: str = Field(description="op or recipe exactly as listed, e.g. acli.jira.search")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="params for the op; a value may be a binding such as $s1.ids",
    )
    after: list[str] = Field(default_factory=list, description="ids of steps this one waits for")


class Plan(BaseModel):
    interpretation: str = Field(description="one sentence: what the question is really asking")
    steps: list[Step] = Field(min_length=1)
    breadth: bool = Field(
        default=False, description="true to fan out across every applicable source"
    )
    expect: str = Field(default="", description="what a complete answer contains")


def plan_errors(plan: Plan, registry: Registry, allowed: set[str], max_steps: int) -> list[str]:
    return step_errors(plan.steps, registry, allowed, max_steps)


def step_errors(
    steps: list[Step],
    registry: Registry,
    allowed: set[str],
    max_steps: int,
    earlier: set[str] | None = None,
) -> list[str]:
    """Everything wrong with these steps, phrased so the planner can fix it on retry.
    `earlier` holds step ids from rounds already run, which bindings may refer to."""
    earlier = earlier or set()
    errors: list[str] = []
    if len(steps) > max_steps:
        errors.append(f"use at most {max_steps} step(s); got {len(steps)}")
    ids = [s.id for s in steps]
    if len(set(ids)) != len(ids) or set(ids) & earlier:
        errors.append(f"step ids must be unique and new: {ids} (already used: {sorted(earlier)})")
    visible = set(ids) | earlier
    for step in steps:
        errors += [f"{step.id}: {e}" for e in _step_errors(step, registry, allowed, visible)]
    return errors


def _step_errors(step: Step, registry: Registry, allowed: set[str], visible: set[str]) -> list[str]:
    if "/" in step.id:
        return ["step ids may not contain '/'"]
    if step.op not in allowed:
        return [f"op {step.op!r} is not in the shortlist; choose from {sorted(allowed)}"]
    errors = [f"after references unknown step {a!r}" for a in step.after if a not in visible]
    try:
        unknown = sorted(r for r in binding_refs(step.params) if r not in visible)
    except BindingError as exc:
        return [*errors, str(exc)]
    if unknown:
        errors.append(f"binds to unknown steps {unknown}")
    errors += _param_errors(step, registry)
    return errors


def _param_errors(step: Step, registry: Registry) -> list[str]:
    recipe = registry.recipes.get(step.op)
    try:
        if recipe is not None:
            validate_shape(recipe, step.params)
            return []
        op = registry.resolve(step.op)[1]
        build_argv(op.argv, validate_shape(op, step.params))
    except LeadingDashError as exc:  # the same checks the executor runs, worded for the planner
        return [f"param {exc.name!r} cannot start with '-'; rephrase {exc.value!r} or omit it"]
    except (ParamError, ArgvError) as exc:
        return [str(exc)]
    return []
