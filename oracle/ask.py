"""One oracle invocation: enter the guard -> rank -> strategy -> summary -> log -> Answer.

The strategy (`oracle/strategy.py`) is how the call gathers: quick plans and runs; standard
adds a completeness check with at most two replans; deep crawls what the results mention, or
(`script`) runs one model-written script against the tools. The summary is per tier: quick
reads summary fields, standard distills, deep map-reduces what was found.
"""

import asyncio
from dataclasses import dataclass, field, replace
from typing import TypedDict

from pydantic_ai.models import Model

from oracle.budget import Budget, Interrupted, parse_budget
from oracle.config import Config, ConfigError
from oracle.exec.dag import StepRun
from oracle.llm.models import resolve_model
from oracle.log.store import CallRecord, open_store
from oracle.plan.distill import Distilled, build_distiller, run_distill
from oracle.registry.loader import Registry
from oracle.render import Answer, Finding, findings_from, tool_of, trace_line
from oracle.sdk.crawl import BACKSTOP_NOTE, backstop
from oracle.sdk.summarize import summarize
from oracle.strategy import Context, Outcome, strategy_for
from oracle.workflows.protocol import AskEntry, Trace, enter_ask
from oracle.workflows.runner import call_scope

DEEP_BUDGET = 200_000  # model tokens per deep call unless --budget or ORACLE_DEEP_BUDGET says


@dataclass
class AskRequest:
    question: str
    intent: str | None = None
    level: str = "standard"
    scope: list[str] | None = None
    prefer: list[str] | None = None
    session: str | None = None
    breadth: bool = False
    """Fan out across every applicable tool, one step each, and group findings by source."""
    depth: int = 3
    """Deep crawl: hops of searching, the first plan included."""
    budget: str | None = None
    """Deep crawl: model tokens, or `unlimited`; default ORACLE_DEEP_BUDGET or DEEP_BUDGET."""
    script: bool = False
    """Deep: write one script against the tools instead of crawling."""


@dataclass
class Models:
    """Injected models (tests); unset roles come from config."""

    planner: Model | str | None = None
    checker: Model | str | None = None
    distiller: Model | str | None = None
    extractor: Model | str | None = None
    """Deep crawl: finds the next terms (Haiku role)."""
    reducer: Model | str | None = None
    """Deep crawl: merges the batched notes into the answer (Sonnet role)."""
    scripter: Model | str | None = None
    """Deep script: writes the script (the `deep` role, Opus by default)."""


class _Coverage(TypedDict):
    searched: list[str]
    not_searched: list[str]
    unavailable: list[str]
    no_match: list[str]
    skipped: list[str]
    step_results: list[dict[str, object]]


class _Meta(TypedDict):
    time_window: str | None
    replans: int
    shortlist: list[str]
    scored: list[str]
    tokens_in: int
    tokens_out: int
    model_calls: int
    wall_ms: int


@dataclass
class _Call:
    ctx: Context
    trace: Trace
    """This call's link in the guard's chain."""
    asked: str
    """The level the caller asked for, before --script raised it."""
    notes: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)

    @property
    def session(self) -> str:
        return self.ctx.req.session or f"adhoc-{self.ctx.call_id}"

    @property
    def level(self) -> str:
        return self.ctx.level


LEVELS = ("quick", "standard", "deep")


async def ask(
    req: AskRequest,
    cfg: Config,
    registry: Registry,
    models: Models | Model | str | None = None,
    stop: asyncio.Event | None = None,
    entry: AskEntry | None = None,
) -> Answer:
    """`stop` (set by Ctrl-C in the CLI) ends a deep crawl at the next hop with what it has.
    `entry` is how a nested call entered the guard (`enter_ask`), and its level and --script
    win over the request's; without one this is a root call, the chain's first link, so a tool
    that asks the same question again is a cycle."""
    if req.level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {req.level!r}")
    if not isinstance(models, Models):
        models = Models(planner=models)  # a bare model is the planner
    entry = entry or enter_ask(None, req.question, req.level, req.script)
    req, notes = _reconcile(replace(req, level=entry.level, script=entry.script))
    budget = _budget(req, cfg, stop)
    with call_scope(cfg, registry, entry.trace, models) as executor:
        ctx = Context.build(req, cfg, registry, models, budget, executor)
        call = _Call(ctx, entry.trace, entry.asked, notes=[*entry.notes, *notes])
        return await _run(call)


async def _run(call: _Call) -> Answer:
    ctx = call.ctx
    if not ctx.shortlist.allowed:
        call.gaps.append("no active tool matches the question or --scope")
        return await _finish(call, Outcome())
    call.notes += _resolve_models(ctx)
    return await _finish(call, await strategy_for(ctx.level, ctx.req.script).run(ctx))


def _resolve_models(ctx: Context) -> list[str]:
    """Fill every role this level needs now, so a missing model setting fails before any tool
    runs (ConfigError, exit 2) instead of half way through."""
    m, cfg = ctx.models, ctx.cfg
    m.planner = m.planner or resolve_model(cfg.model("planner"))
    if ctx.level == "standard":
        m.checker = m.checker or resolve_model(cfg.model("planner"))
    if ctx.level in ("standard", "deep"):
        m.distiller = m.distiller or resolve_model(cfg.model("extract"))
    return _resolve_deep(ctx) if ctx.level == "deep" else []


def _resolve_deep(ctx: Context) -> list[str]:
    m, cfg = ctx.models, ctx.cfg
    m.extractor = m.extractor or m.distiller
    m.reducer = m.reducer or m.planner
    if not ctx.req.script or m.scripter is not None:
        return []
    deep = cfg.models.get("deep")
    m.scripter = resolve_model(deep) if deep else m.planner
    if "deep" in cfg.models:
        return []
    return ["note: no deep model is set; the planner model wrote the script"]


def _budget(req: AskRequest, cfg: Config, stop: asyncio.Event | None) -> Budget:
    """The tier's bounds; at deep, tokens from --budget, else ORACLE_DEEP_BUDGET, else
    DEEP_BUDGET (`unlimited` only when the caller or the config says so)."""
    if req.level != "deep":
        return Budget.for_tier(req.level, stop=stop)
    try:
        tokens = parse_budget(req.budget or cfg.deep_budget, DEEP_BUDGET)
    except ValueError as exc:  # --budget is checked by the CLI; this is the config's value
        raise ConfigError(f"deep budget (ORACLE_DEEP_BUDGET or deep_budget): {exc}") from None
    return Budget.for_tier("deep", tokens, stop)


def _reconcile(req: AskRequest) -> tuple[AskRequest, list[str]]:
    """A request the tier cannot honour as asked, adjusted with a note (the tier and --script
    are settled by `enter_ask`)."""
    if req.breadth and req.level == "quick":
        note = "note: --breadth needs more than one step; ignored at quick"
        return replace(req, breadth=False), [note]
    return req, []


async def _finish(call: _Call, out: Outcome) -> Answer:
    call.notes += out.notes
    call.failures += out.failures
    call.gaps += out.gaps
    if call.level == "deep":
        findings, confidence, refine = await _deep_summary(call, out)
    elif call.level == "standard":
        findings, confidence, refine = await _distill(call, out)
    else:
        findings = findings_from(out.runs, call.ctx.registry)
        confidence, refine = ("medium" if findings else "low"), []
        call.gaps.append("quick tier: no completeness check was run")
    answer = _answer(call, out, findings, confidence, refine)
    await _log(call, answer, out)
    return answer


async def _distill(call: _Call, out: Outcome) -> tuple[list[Finding], str, list[str]]:
    """(findings, confidence, refine). Falls back to summary fields if distill fails."""
    found = out.summary_runs
    if not found:
        return [], "low", []
    distilled = await _try_distill(call, out, found)
    if distilled is None:
        return findings_from(found, call.ctx.registry), "low", []
    source = {p: run.op for run in found for p in run.result.pointers}
    call.gaps = list(dict.fromkeys(call.gaps + distilled.gaps))
    findings = [Finding(f.text, f.pointer, source[f.pointer]) for f in distilled.findings]
    return findings, distilled.confidence, distilled.refine


async def _try_distill(call: _Call, out: Outcome, found: list[StepRun]) -> Distilled | None:
    """None means use raw findings: past the deadline, or distill failed."""
    budget = call.ctx.budget
    if budget.late_for_summary():
        late = budget.bounds.summarise_by_s
        call.notes.append(f"note: past {late:g} s, so findings are raw (no distill)")
        return None
    try:
        return await _run_distill(call, out, found)
    except Exception as exc:
        call.notes.append(f"note: distill failed ({type(exc).__name__}); findings are raw")
        return None


async def _run_distill(call: _Call, out: Outcome, found: list[StepRun]) -> Distilled:
    ctx = call.ctx
    assert ctx.models.distiller is not None
    return await run_distill(
        build_distiller(ctx.models.distiller),
        question=ctx.req.question,
        interpretation=out.interpretation,
        when=ctx.when(),
        breadth=_breadth(ctx, out),
        check_gaps=call.gaps,
        runs=found,
        registry=ctx.registry,
        budget=ctx.budget,
    )


def _breadth(ctx: Context, out: Outcome) -> bool:
    """The plan asked for breadth, or a breadth recipe ran (in the plan or any replan)."""
    planned = ctx.req.breadth or bool(out.plan and out.plan.breadth)
    return planned or any(ctx.registry.recipes[r].breadth for r in ctx.runner.recipes_used)


async def _deep_summary(call: _Call, out: Outcome) -> tuple[list[Finding], str, list[str]]:
    """Map-reduce over everything the crawl found, or what the script reported; raw findings if
    that fails."""
    found = out.summary_runs
    if not found:
        return [], "low", []
    registry = call.ctx.registry
    try:
        distilled, source = await _summarize(call, out, found)
    except Interrupted:
        # the first Ctrl-C landed during the summary and reached its model processes; the user
        # asked to stop and see what was found, so summarise once more (calls started after the
        # stop are not killed by it, so this happens at most once)
        return await _deep_summary(call, out)
    except Exception as exc:
        call.notes.append(f"note: summary failed ({type(exc).__name__}); findings are raw")
        return findings_from(found, registry), "low", []
    call.gaps = list(dict.fromkeys(call.gaps + distilled.gaps))
    findings = [Finding(f.text, f.pointer, source.get(f.pointer, "")) for f in distilled.findings]
    extra = backstop(found, findings, registry)
    call.notes += [f"note: {len(extra)} {BACKSTOP_NOTE}"] if extra else []
    return findings + extra, distilled.confidence, distilled.refine


async def _summarize(
    call: _Call, out: Outcome, found: list[StepRun]
) -> tuple[Distilled, dict[str, str]]:
    ctx = call.ctx
    assert ctx.models.distiller and ctx.models.reducer
    async with ctx.budget.summary() as spend:  # on timeout the findings are raw
        return await summarize(
            question=ctx.req.question,
            interpretation=out.interpretation,
            when=ctx.when(),
            breadth=_breadth(ctx, out),
            gaps=call.gaps,
            runs=found,
            registry=ctx.registry,
            mapper=ctx.models.distiller,
            reducer=ctx.models.reducer,
            budget=spend,
        )


def _gaps(call: _Call, runs: list[StepRun], findings: list[Finding]) -> list[str]:
    gaps = list(call.gaps)
    if not findings and "no results" not in gaps:
        gaps.append("no results")
    spilled = [r.result for r in runs if r.result.ok and r.result.spill]
    return gaps + [f"{r.op} output was truncated; full output at `{r.spill}`" for r in spilled]


def _answer(
    call: _Call, out: Outcome, findings: list[Finding], confidence: str, refine: list[str]
) -> Answer:
    ctx, runs = call.ctx, out.runs
    ran = [r for r in runs if r.ran]
    failures = call.failures + [
        f"{r.op} ({r.result.status}): {r.result.error}"
        for r in runs
        if r.result.status in ("error", "not_configured", "invalid")
    ]
    breadth = _breadth(ctx, out)
    findings = _grouped(findings) if breadth else findings
    return Answer(
        call_id=ctx.call_id,
        session=call.session,
        level=call.level,
        findings=findings,
        failures=failures,
        confidence="medium" if failures and confidence == "high" else confidence,
        gaps=_gaps(call, runs, findings),
        trace=_trace(ctx, out),
        notes=_notes(call, runs),
        refine=refine,
        breadth=breadth,
        ops_run=[r.op for r in ran],
        recipes=list(dict.fromkeys(ctx.runner.recipes_used)),
        **_coverage(ctx.registry, runs),
        **_meta(ctx, out),
        hops=out.hops,
        terms_searched=len(out.searched),
        unexpanded=out.unexpanded,
        stopped=out.stopped,
        cost=ctx.budget.line() if call.level == "deep" else None,  # after the summary's spend
        script=out.script,
    )


def _notes(call: _Call, runs: list[StepRun]) -> list[str]:
    """The call's notes, then what workflows noted (`ctx.log`, tier clamps), once per run: a
    reused lookup shares its first run's result and notes."""
    return call.notes + [
        f"note: {r.op}: {n}" for r in runs if not r.same_as for n in r.result.notes
    ]


def _meta(ctx: Context, out: Outcome) -> _Meta:
    usage = ctx.budget.usage
    return {
        "time_window": ctx.window.describe() if ctx.window else None,
        "replans": out.replans,
        "shortlist": [t.tool for t in ctx.shortlist.tools],
        "scored": [t.tool for t in ctx.shortlist.tools if t.score > 0 or t.pinned],
        "tokens_in": usage.tokens_in,
        "tokens_out": usage.tokens_out,
        "model_calls": usage.model_calls,
        "wall_ms": int(ctx.budget.elapsed() * 1000),
    }


def _grouped(findings: list[Finding]) -> list[Finding]:
    """Group by source, keeping distill's importance order inside and across groups."""
    order = list(dict.fromkeys(tool_of(f.source) for f in findings))
    return sorted(findings, key=lambda f: order.index(tool_of(f.source)))


def _coverage(registry: Registry, runs: list[StepRun]) -> _Coverage:
    """Computed from execution, never by a model."""
    ran = [r for r in runs if r.ran]
    ok = [r for r in ran if r.result.ok]
    return {
        "searched": sorted({tool_of(r.op) for r in ok}),
        "not_searched": sorted(set(registry.tools) - {tool_of(r.op) for r in ran}),
        "unavailable": sorted(set(registry.inactive) | _with_status(runs, "unavailable")),
        "no_match": _no_match(ok),
        "skipped": [f"{r.op} ({r.result.error})" for r in runs if r.result.status == "skipped"],
        "step_results": _step_results(runs),
    }


def _with_status(runs: list[StepRun], status: str) -> set[str]:
    return {tool_of(r.op) for r in runs if r.result.status == status}


def _step_results(runs: list[StepRun]) -> list[dict[str, object]]:
    return [
        {"step": r.step_id, "op": r.op, "status": r.result.status, "n": len(r.result.records)}
        for r in runs
        if not r.same_as
    ]


def _no_match(ok: list[StepRun]) -> list[str]:
    """Ops that ran cleanly and found nothing every time they ran."""
    found = {r.op for r in ok if r.result.records}
    return sorted({r.op for r in ok} - found)


def _trace(ctx: Context, out: Outcome) -> list[str]:
    lines = [f"time: {ctx.window.describe()}"] if ctx.window else []
    lines += out.trace
    current = 0
    for run in out.runs:
        if run.round != current:
            current = run.round
            lines.append(f"{out.round_name} {current}:")
        prefix = f"[{run.recipe}] " if run.recipe else ""
        lines.append(prefix + trace_line(run))
    return lines


async def _log(call: _Call, answer: Answer, out: Outcome) -> None:
    ctx = call.ctx
    async with open_store(ctx.cfg.log_path) as store:
        await store.record_call(
            CallRecord(
                call_id=answer.call_id,
                session=answer.session,
                query=ctx.req.question,
                intent=ctx.req.intent,
                level=call.asked,  # as requested; ran_level is what ran
                ranked_tools=[
                    {"tool": t.tool, "score": t.score, "pinned": t.pinned, "matched": t.matched_ops}
                    for t in ctx.shortlist.tools
                ]
                + [{"recipe": r} for r in ctx.shortlist.recipes],
                plan=out.plan.model_dump() if out.plan else None,
                wall_ms=answer.wall_ms,
                tokens_in=answer.tokens_in,
                tokens_out=answer.tokens_out,
                model_calls=answer.model_calls,
                breadth=answer.breadth,
                confidence=answer.confidence,
                gaps=answer.gaps + answer.failures,
                ran_level=answer.level,
                replans=answer.replans,
                script=out.script,
                trace_id=call.trace.trace_id,
                depth=call.trace.depth,
            )
        )
        for run in out.runs + out.logged:
            result = run.result
            if run.same_as:  # logged as a marker so reports do not count the lookup twice
                result = replace(
                    result, ms=0, records=[], pointers=[], ids=[], error=f"reused {run.same_as}"
                )
            await store.record_op(
                answer.call_id, answer.session, run.step_id, result, run.hop, run.parent,
                commit=False,
            )  # fmt: skip
        await store.commit()  # one transaction for every op of the call
