"""Run a plan: steps in dependency waves, independent steps in parallel, recipes expanded into
their stages, bindings resolved from earlier results, and recipe fallbacks when nothing came back.

Every step ends as a StepRun with an OpResult, including steps that were skipped or whose tool is
not active here, so Coverage and the log can account for all of them.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from oracle.exec.bindings import BindingError, binding_refs, fill_template, is_binding, resolve
from oracle.exec.executor import Executor, OpResult, Status
from oracle.plan.models import Step
from oracle.registry.loader import Registry
from oracle.registry.params import ParamError, validate_params
from oracle.registry.schema import Recipe, RecipeStep


@dataclass
class StepRun:
    step_id: str
    result: OpResult
    recipe: str | None = None
    round: int = 0  # 0 for the first plan, n for replan n
    same_as: str | None = None
    """Set when an identical (op, params) already ran in this call; the result is reused."""

    @property
    def op(self) -> str:
        return self.result.op

    @property
    def ran(self) -> bool:
        """The op was actually invoked, whatever it returned. A tool that is not configured
        (auth env unset, binary missing) never ran, so it cannot satisfy an eval clause."""
        return self.result.status in ("ok", "error")


@dataclass
class PlanRunner:
    registry: Registry
    executor: Executor
    builtins: dict[str, Any] = field(default_factory=dict)
    runs: dict[str, StepRun] = field(default_factory=dict)
    aliases: dict[str, list[str]] = field(default_factory=dict)
    recipes_used: list[str] = field(default_factory=list)
    _by_call: dict[str, tuple[str, asyncio.Task[OpResult]]] = field(default_factory=dict)

    async def run(self, steps: list[Step], round_: int = 0) -> list[StepRun]:
        return await self._waves(steps, lambda s: self._node(s, round_))

    def results_of(self, step_id: str) -> list[OpResult]:
        ids = self.aliases.get(step_id, [step_id])
        return [self.runs[i].result for i in ids if i in self.runs]

    async def _waves(
        self, steps: list[Step], run_one: Callable[[Step], Awaitable[list[StepRun]]]
    ) -> list[StepRun]:
        pending = {s.id: s for s in steps}
        done: list[StepRun] = []
        while pending:
            ready = [s for s in pending.values() if not (self._deps(s) & set(pending))]
            if not ready:
                done += [self._settle(s, "invalid", "dependency cycle") for s in pending.values()]
                break
            for batch in await asyncio.gather(*(run_one(s) for s in ready)):
                done += batch
            for s in ready:
                del pending[s.id]
        return done

    def _deps(self, step: Step) -> set[str]:
        try:
            return set(step.after) | binding_refs(step.params)
        except BindingError:
            return set(step.after)

    async def _node(self, step: Step, round_: int) -> list[StepRun]:
        recipe = self.registry.recipes.get(step.op)
        if recipe is not None:
            return await self._recipe(step, recipe, round_)
        return [await self._op(step, None, round_)]

    async def _recipe(self, step: Step, recipe: Recipe, round_: int) -> list[StepRun]:
        try:
            params, missing = self._bind(step.params)
            if missing:
                reason = f"needs results from {', '.join(sorted(missing))}, which returned none"
                return [self._settle(step, "skipped", reason, round_=round_)]
            values = self.builtins | validate_params(recipe, params)
        except (BindingError, ParamError) as exc:
            return [self._settle(step, "invalid", str(exc), round_=round_)]
        self.recipes_used.append(recipe.name)
        runs = await self._stages(step, recipe, recipe.steps, values, round_)
        self.aliases[step.id] = [f"{step.id}/{o}" for o in recipe.outputs]
        if recipe.on_empty and not any(r.records for r in self.results_of(step.id)):
            runs += await self._stages(step, recipe, recipe.on_empty, values, round_)
            self.aliases[step.id] = [f"{step.id}/{s.id}" for s in recipe.on_empty]
        return runs

    async def _stages(
        self,
        step: Step,
        recipe: Recipe,
        stages: list[RecipeStep],
        values: dict[str, Any],
        round_: int,
    ) -> list[StepRun]:
        names = {s.id for s in recipe.steps + recipe.on_empty}
        expanded = [
            Step(
                id=f"{step.id}/{s.id}",
                op=s.op,
                params={k: fill_template(v, values, step.id, names) for k, v in s.params.items()},
                after=[f"{step.id}/{a}" for a in s.after if a in names],
            )
            for s in stages
        ]
        return await self._waves(expanded, lambda s: self._one(s, recipe.name, round_))

    async def _one(self, step: Step, recipe: str, round_: int) -> list[StepRun]:
        return [await self._op(step, recipe, round_)]

    async def _op(self, step: Step, recipe: str | None, round_: int) -> StepRun:
        if step.op.partition(".")[0] not in self.registry.tools:
            return self._settle(step, "unavailable", "not registered here", recipe, round_)
        try:
            params, missing = self._bind(step.params)
        except BindingError as exc:
            return self._settle(step, "invalid", str(exc), recipe, round_)
        if missing:
            reason = f"needs results from {', '.join(sorted(missing))}, which returned none"
            return self._settle(step, "skipped", reason, recipe, round_)
        key = step.op + json.dumps(params, sort_keys=True, default=str)
        if key in self._by_call:  # the same lookup again, even in the same wave: run it once
            first, task = self._by_call[key]
            run = StepRun(step.id, await task, recipe, round_, same_as=first)
        else:
            task = asyncio.ensure_future(self.executor.run(step.op, params))
            self._by_call[key] = (step.id, task)
            run = StepRun(step.id, await task, recipe, round_)
        self.runs[step.id] = run
        return run

    def _bind(self, params: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
        out, missing = {}, set()
        for key, value in params.items():
            if not is_binding(value):
                out[key] = value
                continue
            bound = resolve(value, self.results_of)
            if bound in (None, []):
                missing |= binding_refs({key: value})
            out[key] = bound
        return out, missing

    def _settle(
        self,
        step: Step,
        status: Status,
        reason: str,
        recipe: str | None = None,
        round_: int = 0,
    ) -> StepRun:
        result = OpResult(op=step.op, params=dict(step.params), status=status, error=reason)
        run = StepRun(step.id, result, recipe, round_)
        self.runs[step.id] = run
        return run
