"""One oracle invocation: rank -> plan -> execute -> check and replan -> distill -> log -> Answer.

quick: one step, no check, findings from summary fields. standard: up to 12 steps, a completeness
check with at most two replans, distilled findings. deep runs as standard until M4 and says so.
"""

import asyncio
import time
import uuid
from collections.abc import Awaitable
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import TypedDict
from zoneinfo import ZoneInfo

from pydantic_ai.models import Model

from oracle.config import REPO_ROOT, Config
from oracle.exec.dag import PlanRunner, StepRun
from oracle.exec.executor import Executor
from oracle.llm.models import resolve_model
from oracle.log.store import CallRecord, open_store
from oracle.plan.check import Check, build_checker, run_check
from oracle.plan.digest import summary_of
from oracle.plan.distill import Distilled, build_distiller, run_distill
from oracle.plan.models import Plan
from oracle.plan.planner import Shortlist, Usage, build_planner, make_plan
from oracle.rank.index import Ranker
from oracle.registry.loader import Registry
from oracle.render import Answer, Finding, findings_from, tool_of, trace_line
from oracle.skills.loader import load_skills, render, select
from oracle.timewin import TimeWindow, default_builtins, resolve_window

SKILLS = load_skills(REPO_ROOT / "skills")
MAX_STEPS = {"quick": 1, "standard": 12}
MAX_REPLANS = 2
WALL_S = 60.0  # no new check after this; DESIGN's bound for standard
MODEL_CALL_S = 90.0  # any single model call, whatever the provider
DISTILL_BY_S = 90.0  # past this, findings are raw summary fields rather than a late distill


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


@dataclass
class Models:
    """Injected models (tests); unset roles come from config."""

    planner: Model | str | None = None
    checker: Model | str | None = None
    distiller: Model | str | None = None


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


def now_in(tz: str | None) -> datetime:
    return datetime.now(ZoneInfo(tz)) if tz else datetime.now().astimezone()


@dataclass
class _Call:
    req: AskRequest
    cfg: Config
    registry: Registry
    models: Models
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started: float = field(default_factory=time.monotonic)
    usage: Usage = field(default_factory=Usage)
    notes: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    plan: Plan | None = None
    replans: int = 0
    window: TimeWindow | None = None
    shortlist: Shortlist = field(default_factory=lambda: Shortlist([]))
    step_ids: set[str] = field(default_factory=set)
    """Plan-level step ids used so far (plan and replans), which later steps may bind to."""
    runner: PlanRunner | None = None

    @property
    def session(self) -> str:
        return self.req.session or f"adhoc-{self.call_id}"

    @property
    def level(self) -> str:
        return "quick" if self.req.level == "quick" else "standard"

    def resolve_models(self) -> None:
        """Fill every role this level needs now, so a missing model setting fails before any
        tool runs (ConfigError, exit 2) instead of half way through."""
        m = self.models
        m.planner = m.planner or resolve_model(self.cfg.model("planner"))
        if self.level == "standard":
            m.checker = m.checker or resolve_model(self.cfg.model("planner"))
            m.distiller = m.distiller or resolve_model(self.cfg.model("extract"))


LEVELS = ("quick", "standard", "deep")


async def ask(
    req: AskRequest, cfg: Config, registry: Registry, models: Models | Model | str | None = None
) -> Answer:
    if req.level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {req.level!r}")
    if not isinstance(models, Models):
        models = Models(planner=models)  # a bare model is the planner
    call = _Call(req, cfg, registry, models)
    req = _reconcile(call)
    now = now_in(cfg.timezone)
    call.window = resolve_window(req.question, now)
    ranker = Ranker(registry)
    call.shortlist = _shortlist(ranker, req.question, req)
    if not call.shortlist.allowed:
        call.gaps.append("no active tool matches the question or --scope")
        return await _finish(call, [])
    call.resolve_models()
    assert models.planner is not None
    planner = build_planner(models.planner)
    runner = _runner(call, now)
    try:
        outcome = await _bounded(make_plan(
            planner,
            question=req.question,
            intent=req.intent,
            shortlist=call.shortlist,
            registry=registry,
            max_steps=MAX_STEPS[call.level],
            now=now,
            window=call.window,
            breadth=req.breadth,
            guidance=_guidance(call, "planner", call.shortlist),
        ))  # fmt: skip
    except Exception as exc:  # any model or transport failure is reported, never a traceback
        call.failures.append(f"planner: {type(exc).__name__}: {exc}")
        return await _finish(call, [])
    call.plan = outcome.plan
    call.runner = runner
    call.step_ids = {s.id for s in outcome.plan.steps}
    call.usage.merge(outcome.usage)
    runs = await runner.run(outcome.plan.steps)
    if call.level == "standard":
        runs = await _check_loop(call, ranker, runner, runs)
    return await _finish(call, runs, runner)


def _reconcile(call: _Call) -> AskRequest:
    """Requests the levels cannot honour as asked, adjusted with a note."""
    if call.req.level == "deep":
        call.notes.append("note: deep is not built yet (M4); ran standard")
    if call.req.breadth and call.req.level == "quick":
        call.req = replace(call.req, breadth=False)
        call.notes.append("note: --breadth needs more than one step; ignored at quick")
    return call.req


def _runner(call: _Call, now: datetime) -> PlanRunner:
    executor = Executor(call.registry, call.cfg.tool_env(), call.cfg.spill_dir)
    builtins = call.window.builtins() if call.window else default_builtins(now)
    return PlanRunner(call.registry, executor, builtins=builtins)


async def _bounded[T](work: Awaitable[T]) -> T:
    async with asyncio.timeout(MODEL_CALL_S):
        return await work


def _shortlist(ranker: Ranker, query: str, req: AskRequest) -> Shortlist:
    tools = ranker.rank(query, scope=req.scope, prefer=req.prefer, breadth=req.breadth)
    recipes = ranker.rank_recipes(query)
    if req.scope:
        active = set(ranker.registry.tools)
        recipes = [r for r in recipes if _recipe_tools(ranker, r) & active <= set(req.scope)]
    return Shortlist(tools, recipes)


def _recipe_tools(ranker: Ranker, name: str) -> set[str]:
    return {tool_of(s.op) for s in ranker.registry.recipes[name].steps}


async def _check_loop(
    call: _Call, ranker: Ranker, runner: PlanRunner, runs: list[StepRun]
) -> list[StepRun]:
    assert call.plan is not None
    assert call.models.checker is not None
    checker = build_checker(call.models.checker)
    planned = len(call.plan.steps)
    while time.monotonic() - call.started <= WALL_S:
        last = call.replans >= MAX_REPLANS or planned >= MAX_STEPS["standard"]
        conditioned = _conditioned(call, ranker, runs)
        try:
            check = await _bounded(run_check(
                checker,
                question=call.req.question,
                intent=call.req.intent,
                plan=call.plan,
                window=call.window,
                runs=runs,
                shortlist=conditioned,
                registry=call.registry,
                max_steps=0 if last else MAX_STEPS["standard"] - planned,
                usage=call.usage,
                earlier=call.step_ids,
                guidance=_guidance(call, "check", conditioned),
            ))  # fmt: skip
        except Exception as exc:  # a failed check leaves the results as they are
            call.notes.append(f"note: completeness check failed ({type(exc).__name__}: {exc})")
            return runs
        call.gaps = list(check.gaps)
        if not _replan(call, check, planned):
            return runs
        call.replans += 1
        planned += len(check.next_steps)
        call.step_ids |= {s.id for s in check.next_steps}
        runs = runs + await runner.run(check.next_steps, round_=call.replans)
    call.notes.append(f"note: stopped checking at the {WALL_S:g} s bound")
    return runs


def _guidance(call: _Call, role: str, shortlist: Shortlist) -> str:
    tools = [t.tool for t in shortlist.tools]
    chosen = select(SKILLS, role, tools, call.registry, call.req.question)
    if chosen and role == "planner":
        call.notes.append(f"skills: {', '.join(s.name for s in chosen)}")
    return render(chosen)


def _replan(call: _Call, check: Check, planned: int) -> bool:
    if check.sufficient or check.complete or not check.next_steps:
        return False
    if call.replans >= MAX_REPLANS or planned >= MAX_STEPS["standard"]:
        call.notes.append("note: the check wanted more steps; replan limit reached")
        return False
    return True


def _conditioned(call: _Call, ranker: Ranker, runs: list[StepRun]) -> Shortlist:
    """Re-rank on the question plus what was found, so a result mentioning a ticket brings
    acli into reach even if the question never mentioned tickets."""
    seen = [summary_of(rec, []) for run in runs if run.result.ok for rec in run.result.records[:5]]
    query = " ".join([call.req.question, *call.gaps, *seen])[:4000]
    return _shortlist(ranker, query, call.req)


async def _distill(call: _Call, runs: list[StepRun]) -> tuple[list[Finding], str, list[str]]:
    """(findings, confidence, refine). Falls back to summary fields if distill fails."""
    found = [r for r in runs if r.result.ok and r.result.records and not r.same_as]
    if not found:
        return [], "low", []
    distilled = await _try_distill(call, found)
    if distilled is None:
        return findings_from(found, call.registry), "low", []
    source = {p: run.op for run in found for p in run.result.pointers}
    call.gaps = list(dict.fromkeys(call.gaps + distilled.gaps))
    findings = [Finding(f.text, f.pointer, source[f.pointer]) for f in distilled.findings]
    return findings, distilled.confidence, distilled.refine


async def _try_distill(call: _Call, found: list[StepRun]) -> Distilled | None:
    """None means use raw findings: past the deadline, or distill failed."""
    if time.monotonic() - call.started > DISTILL_BY_S:
        call.notes.append(f"note: past {DISTILL_BY_S:g} s, so findings are raw (no distill)")
        return None
    try:
        return await _bounded(_run_distill(call, found))
    except Exception as exc:
        call.notes.append(f"note: distill failed ({type(exc).__name__}); findings are raw")
        return None


async def _run_distill(call: _Call, found: list[StepRun]) -> Distilled:
    assert call.models.distiller is not None
    return await run_distill(
        build_distiller(call.models.distiller),
        question=call.req.question,
        interpretation=call.plan.interpretation if call.plan else "",
        when=call.window.describe()
        if call.window
        else now_in(call.cfg.timezone).isoformat(timespec="minutes"),
        breadth=_breadth(call),
        check_gaps=call.gaps,
        runs=found,
        registry=call.registry,
        usage=call.usage,
    )


def _breadth(call: _Call) -> bool:
    """The plan asked for breadth, or a breadth recipe ran (in the plan or any replan)."""
    used = call.runner.recipes_used if call.runner else []
    planned = call.req.breadth or bool(call.plan and call.plan.breadth)
    return planned or any(call.registry.recipes[r].breadth for r in used)


async def _finish(call: _Call, runs: list[StepRun], runner: PlanRunner | None = None) -> Answer:
    if call.level == "standard":
        findings, confidence, refine = await _distill(call, runs)
    else:
        findings = findings_from(runs, call.registry)
        confidence, refine = ("medium" if findings else "low"), []
        call.gaps.append("quick tier: no completeness check was run")
    answer = _answer(call, runs, findings, confidence, refine, runner)
    await _log(call, answer, runs)
    return answer


def _gaps(call: _Call, runs: list[StepRun], findings: list[Finding]) -> list[str]:
    gaps = list(call.gaps)
    if not findings and "no results" not in gaps:
        gaps.append("no results")
    spilled = [r.result for r in runs if r.result.ok and r.result.spill]
    return gaps + [f"{r.op} output was truncated; full output at `{r.spill}`" for r in spilled]


def _answer(
    call: _Call,
    runs: list[StepRun],
    findings: list[Finding],
    confidence: str,
    refine: list[str],
    runner: PlanRunner | None,
) -> Answer:
    ran = [r for r in runs if r.ran]
    failures = call.failures + [
        f"{r.op} ({r.result.status}): {r.result.error}"
        for r in runs
        if r.result.status in ("error", "not_configured", "invalid")
    ]
    recipes = list(dict.fromkeys(runner.recipes_used if runner else []))
    breadth = _breadth(call)
    findings = _grouped(findings) if breadth else findings
    return Answer(
        call_id=call.call_id,
        session=call.session,
        level=call.level,
        findings=findings,
        failures=failures,
        confidence="medium" if failures and confidence == "high" else confidence,
        gaps=_gaps(call, runs, findings),
        trace=_trace(call, runs),
        notes=call.notes,
        refine=refine,
        breadth=breadth,
        ops_run=[r.op for r in ran],
        recipes=recipes,
        **_coverage(call, runs),
        **_meta(call),
    )


def _meta(call: _Call) -> _Meta:
    return {
        "time_window": call.window.describe() if call.window else None,
        "replans": call.replans,
        "shortlist": [t.tool for t in call.shortlist.tools],
        "scored": [t.tool for t in call.shortlist.tools if t.score > 0 or t.pinned],
        "tokens_in": call.usage.tokens_in,
        "tokens_out": call.usage.tokens_out,
        "model_calls": call.usage.model_calls,
        "wall_ms": int((time.monotonic() - call.started) * 1000),
    }


def _grouped(findings: list[Finding]) -> list[Finding]:
    """Group by source, keeping distill's importance order inside and across groups."""
    order = list(dict.fromkeys(tool_of(f.source) for f in findings))
    return sorted(findings, key=lambda f: order.index(tool_of(f.source)))


def _coverage(call: _Call, runs: list[StepRun]) -> _Coverage:
    """Computed from execution, never by a model."""
    ran = [r for r in runs if r.ran]
    ok = [r for r in ran if r.result.ok]
    return {
        "searched": sorted({tool_of(r.op) for r in ok}),
        "not_searched": sorted(set(call.registry.tools) - {tool_of(r.op) for r in ran}),
        "unavailable": sorted(set(call.registry.inactive) | _with_status(runs, "unavailable")),
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


def _trace(call: _Call, runs: list[StepRun]) -> list[str]:
    lines = [f"time: {call.window.describe()}"] if call.window else []
    current = 0
    for run in runs:
        if run.round != current:
            current = run.round
            lines.append(f"replan {current}:")
        prefix = f"[{run.recipe}] " if run.recipe else ""
        lines.append(prefix + trace_line(run))
    return lines


async def _log(call: _Call, answer: Answer, runs: list[StepRun]) -> None:
    async with open_store(call.cfg.log_path) as store:
        await store.record_call(
            CallRecord(
                call_id=answer.call_id,
                session=answer.session,
                query=call.req.question,
                intent=call.req.intent,
                level=call.req.level,  # as requested; ran_level is what ran
                ranked_tools=[
                    {"tool": t.tool, "score": t.score, "pinned": t.pinned, "matched": t.matched_ops}
                    for t in call.shortlist.tools
                ]
                + [{"recipe": r} for r in call.shortlist.recipes],
                plan=call.plan.model_dump() if call.plan else None,
                wall_ms=answer.wall_ms,
                tokens_in=answer.tokens_in,
                tokens_out=answer.tokens_out,
                model_calls=answer.model_calls,
                breadth=answer.breadth,
                confidence=answer.confidence,
                gaps=answer.gaps + answer.failures,
                ran_level=answer.level,
                replans=answer.replans,
            )
        )
        for run in runs:
            result = run.result
            if run.same_as:  # logged as a marker so reports do not count the lookup twice
                result = replace(
                    result, ms=0, records=[], pointers=[], ids=[], error=f"reused {run.same_as}"
                )
            await store.record_op(answer.call_id, answer.session, run.step_id, result)
