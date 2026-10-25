"""Summary for a crawl: map then reduce. Results are batched (MAP_BATCH records per Haiku
call) rather than one call per result, so a 200-result crawl costs about ten map calls, not
two hundred; one reduce call (Sonnet) merges the notes into the usual output contract. Small
result sets skip the map and go straight to one distill call.
"""

import asyncio
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

from oracle.budget import Budget, Interrupted
from oracle.exec.dag import StepRun
from oracle.plan.digest import (
    DATA_CLOSE,
    DATA_OPEN,
    DATA_RULE,
    defang,
    slice_of,
    summary_of,
)
from oracle.plan.distill import (
    PER_STEP,
    DistillContext,
    Distilled,
    Item,
    build_distiller,
    run_distill,
)
from oracle.plan.planner import PLAN_RETRIES, prompt
from oracle.registry.loader import Registry

MAP_BATCH = 20
MAP_CONCURRENCY = 4


class Notes(BaseModel):
    items: list[Item] = Field(default_factory=list, max_length=10)


@dataclass
class MapContext:
    pointers: set[str]


def build_mapper(model: Model | str) -> Agent[MapContext, Notes]:
    agent = Agent(
        model,
        deps_type=MapContext,
        output_type=Notes,
        instructions=prompt("summarize_map") + "\n" + DATA_RULE,
        retries=PLAN_RETRIES,
    )

    @agent.output_validator
    def _valid(ctx: RunContext[MapContext], notes: Notes) -> Notes:
        bad = [i.pointer for i in notes.items if i.pointer not in ctx.deps.pointers]
        if bad:
            raise ModelRetry(f"pointers not in the data: {bad}; copy pointers exactly")
        return notes

    return agent


@dataclass
class Line:
    pointer: str
    op: str
    text: str


def lines_of(runs: list[StepRun], registry: Registry) -> list[Line]:
    out: list[Line] = []
    for run in runs:
        if not run.result.ok or run.same_as:
            continue
        fields = registry.summary_fields(run.op)
        via = f"[{run.followed()}] " if run.parent else ""
        for record, pointer in zip(run.result.records, run.result.pointers, strict=True):
            text = summary_of(record, fields, 200)
            extra = slice_of(record, fields, 200)
            out.append(Line(pointer, run.op, via + text + (f" | {extra}" if extra else "")))
    return out


async def summarize(
    *,
    question: str,
    interpretation: str,
    when: str,
    breadth: bool,
    gaps: list[str],
    runs: list[StepRun],
    registry: Registry,
    mapper: Model | str,
    reducer: Model | str,
    budget: Budget,
) -> tuple[Distilled, dict[str, str]]:
    """(distilled answer, pointer -> op that produced it)."""
    found = [r for r in runs if r.found]
    source = {p: r.op for r in found for p in r.result.pointers}
    # small enough for one distill call, which shows each step's first PER_STEP records
    if len(source) <= MAP_BATCH * 2 and all(len(r.result.records) <= PER_STEP for r in found):
        distilled = await run_distill(
            build_distiller(reducer),
            question=question,
            interpretation=interpretation,
            when=when,
            breadth=breadth,
            check_gaps=gaps,
            runs=found,
            registry=registry,
            budget=budget,
        )
        return distilled, source
    notes = await _map(question, lines_of(found, registry), mapper, budget)
    return await _reduce(
        question, interpretation, when, breadth, gaps, notes, source, registry, reducer, budget
    ), source


async def _map(question: str, lines: list[Line], model: Model | str, budget: Budget) -> list[Item]:
    agent = build_mapper(model)

    slots = asyncio.Semaphore(MAP_CONCURRENCY)  # whatever the provider; Claude Code caps again

    async def one(batch: list[Line]) -> list[Item]:
        data = "\n".join(f"  {defang(b.pointer)} | {defang(b.text)}" for b in batch)
        text = f"Question: {question}\n\n{DATA_OPEN}\n{data}\n{DATA_CLOSE}"
        async with slots:
            deps = MapContext({b.pointer for b in batch})
            result = await budget.model(agent.run(text, deps=deps))
        return result.output.items

    batches = [lines[i : i + MAP_BATCH] for i in range(0, len(lines), MAP_BATCH)]
    notes = await asyncio.gather(*(one(b) for b in batches), return_exceptions=True)
    if (failure := _all_failed(notes)) is not None:
        raise failure
    out: list[Item] = []
    for batch, group in zip(batches, notes, strict=True):
        # a batch whose map call failed goes to the merge as its lines, unshortened
        out += _unmapped(batch) if isinstance(group, BaseException) else group
    return out


def _all_failed(notes: list[list[Item] | BaseException]) -> BaseException | None:
    """When every batch failed: a Ctrl-C among them if any (so the deep summary reruns), else
    the first failure."""
    failed = [n for n in notes if isinstance(n, BaseException)]
    if not notes or len(failed) < len(notes):
        return None
    return next((n for n in failed if isinstance(n, Interrupted)), failed[0])


def _unmapped(batch: list[Line]) -> list[Item]:
    return [Item(text=b.text, pointer=b.pointer) for b in batch]


async def _reduce(
    question: str,
    interpretation: str,
    when: str,
    breadth: bool,
    gaps: list[str],
    notes: list[Item],
    source: dict[str, str],
    registry: Registry,
    model: Model | str,
    budget: Budget,
) -> Distilled:
    agent = build_distiller(model)
    header = [f"Question: {question}", f"Interpretation: {interpretation}", f"Now: {when}"]
    if breadth:
        header.append("Breadth: cover every source that returned something, most important first.")
    if gaps:
        header.append("Not covered: " + "; ".join(gaps))
    data = "\n".join(
        f"  {defang(n.pointer)} | {source.get(n.pointer, '')}: {defang(n.text)}" for n in notes
    )
    text = "\n".join(
        [*header, "", "Notes from every batch of results:", DATA_OPEN, data, DATA_CLOSE]
    )
    deps = DistillContext({n.pointer for n in notes}, set(registry.tools))
    result = await budget.model(agent.run(text, deps=deps))
    return result.output
