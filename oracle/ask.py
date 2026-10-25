"""One oracle invocation: rank -> plan -> execute -> check and replan -> distill -> log -> Answer.

quick: one step, no check, findings from summary fields. standard: up to 12 steps, a completeness
check with at most two replans, distilled findings. deep: the plan, then a crawl that follows what
the results mention, or (`script`) one model-written script against the tools; a map-reduce
summary either way.
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

from oracle.config import REPO_ROOT, Config, ConfigError
from oracle.deep.scripted import ScriptPath, ScriptRun
from oracle.exec.dag import PlanRunner, StepRun
from oracle.exec.executor import TOOL_SLOTS, Executor
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
from oracle.sdk.budget import Budget, parse_budget
from oracle.sdk.crawl import BACKSTOP_NOTE, Crawler, CrawlOutcome, backstop
from oracle.sdk.extract import build_extractor
from oracle.sdk.hop import build_hop_planner
from oracle.sdk.summarize import summarize
from oracle.skills.loader import load_skills, render, select
from oracle.timewin import TimeWindow, default_builtins, resolve_window
from oracle.workflows.protocol import CURRENT_TRACE, Trace
from oracle.workflows.runner import WorkflowHost

SKILLS = load_skills(REPO_ROOT / "skills")
MAX_STEPS = {"quick": 1, "standard": 12, "deep": 12}
DEEP_BUDGET = 200_000  # model tokens per deep call unless --budget or ORACLE_DEEP_BUDGET says
SUMMARY_RESERVE = 20_000
DEEP_WALL_S = 300.0
MAX_REPLANS = 2
WALL_S = 60.0  # no new check after this; DESIGN's bound for standard
MODEL_CALL_S = 90.0  # any single model call, whatever the provider
DISTILL_BY_S = 90.0  # past this, findings are raw summary fields rather than a late distill
DEEP_SUMMARY_S = 240.0  # the deep summary, map batches and merge together


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
    budget: Budget | None = None
    crawl: CrawlOutcome | None = None
    stop: asyncio.Event | None = None
    script: ScriptRun | None = None
    trace: Trace | None = None
    asked: str = "standard"
    """The level the caller asked for, before any adjustment."""

    @property
    def session(self) -> str:
        return self.req.session or f"adhoc-{self.call_id}"

    @property
    def level(self) -> str:
        return self.req.level

    def resolve_models(self) -> None:
        """Fill every role this level needs now, so a missing model setting fails before any
        tool runs (ConfigError, exit 2) instead of half way through."""
        m = self.models
        m.planner = m.planner or resolve_model(self.cfg.model("planner"))
        if self.level == "standard":
            m.checker = m.checker or resolve_model(self.cfg.model("planner"))
        if self.level in ("standard", "deep"):
            m.distiller = m.distiller or resolve_model(self.cfg.model("extract"))
        if self.level == "deep":
            self._resolve_deep()

    def _resolve_deep(self) -> None:
        m = self.models
        m.extractor = m.extractor or m.distiller
        m.reducer = m.reducer or m.planner
        if self.req.script and m.scripter is None:
            if "deep" not in self.cfg.models:
                self.notes.append("note: no deep model is set; the planner model wrote the script")
            deep = self.cfg.models.get("deep")
            m.scripter = resolve_model(deep) if deep else m.planner


LEVELS = ("quick", "standard", "deep")


async def ask(
    req: AskRequest,
    cfg: Config,
    registry: Registry,
    models: Models | Model | str | None = None,
    stop: asyncio.Event | None = None,
    trace: Trace | None = None,
) -> Answer:
    """`stop` (set by Ctrl-C in the CLI) ends a deep crawl at the next hop with what it has.
    `trace` is the guard's chain when this call is nested in a workflow or another oracle."""
    if req.level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {req.level!r}")
    if not isinstance(models, Models):
        models = Models(planner=models)  # a bare model is the planner
    call = _Call(req, cfg, registry, models, stop=stop, asked=req.level)
    req = _reconcile(call, trace)
    # the root call is the chain's first link, so a tool that asks the same question again is
    # a cycle; a nested call arrives with its link already entered
    call.trace = (
        replace(trace, level=req.level)  # --script may have raised it, within the ceiling
        if trace
        else Trace.root(req.level).enter("oracle", "ask", req.question)
    )
    token = CURRENT_TRACE.set(call.trace)  # nested calls below this one see this call's trace
    shared = TOOL_SLOTS.get()
    slots = TOOL_SLOTS.set({} if shared is None else shared)  # nested calls share tool limits
    try:
        return await _run(call)
    finally:
        TOOL_SLOTS.reset(slots)
        CURRENT_TRACE.reset(token)


async def _run(call: _Call) -> Answer:
    req, cfg, registry = call.req, call.cfg, call.registry
    if req.level == "deep":
        call.budget = _deep_budget(call)
    now = now_in(cfg.timezone)
    call.window = resolve_window(req.question, now)
    ranker = Ranker(registry)
    call.shortlist = _shortlist(ranker, req.question, req)
    if not call.shortlist.allowed:
        call.gaps.append("no active tool matches the question or --scope")
        return await _finish(call, [])
    call.resolve_models()
    runner = call.runner = _runner(call, now)
    if req.script:
        return await _finish(call, await _script(call, runner), runner)
    return await _planned(call, ranker, runner, now)


async def _planned(call: _Call, ranker: Ranker, runner: PlanRunner, now: datetime) -> Answer:
    """Plan, run, then check and replan (standard) or crawl (deep)."""
    req = call.req
    assert call.models.planner is not None
    planner = build_planner(call.models.planner)
    try:
        outcome = await _bounded(make_plan(
            planner,
            question=req.question,
            intent=req.intent,
            shortlist=call.shortlist,
            registry=call.registry,
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
    call.step_ids = {s.id for s in outcome.plan.steps}
    call.usage.merge(outcome.usage)
    runs = await runner.run(outcome.plan.steps)
    if call.level == "standard":
        runs = await _check_loop(call, ranker, runner, runs)
    if call.level == "deep":
        runs = await _crawl(call, ranker, runner, runs)
    return await _finish(call, runs, runner)


def _deep_budget(call: _Call) -> Budget:
    """--budget, else ORACLE_DEEP_BUDGET, else DEEP_BUDGET. `unlimited` only when the caller or
    the config says so; it drops the wall-clock bound too, leaving hop depth as the limit."""
    try:
        tokens = parse_budget(call.req.budget or call.cfg.deep_budget, DEEP_BUDGET)
    except ValueError as exc:  # --budget is checked by the CLI; this is the config's value
        raise ConfigError(f"deep budget (ORACLE_DEEP_BUDGET or deep_budget): {exc}") from None
    wall = None if tokens is None else DEEP_WALL_S
    # a small budget keeps a smaller share back: its crawl is small, and so is its summary
    reserve = SUMMARY_RESERVE if tokens is None else min(SUMMARY_RESERVE, tokens // 4)
    return Budget(tokens=tokens, reserve=reserve, wall_s=wall, usage=call.usage)


async def _crawl(
    call: _Call, ranker: Ranker, runner: PlanRunner, runs: list[StepRun]
) -> list[StepRun]:
    assert call.budget is not None and call.models.extractor and call.models.planner
    crawler = Crawler(
        question=call.req.question,
        registry=call.registry,
        runner=runner,
        extractor=build_extractor(call.models.extractor),
        hop_planner=build_hop_planner(call.models.planner),
        budget=call.budget,
        shortlist_for=lambda query: _shortlist(ranker, query, call.req),
        step_ids=call.step_ids,
        usage=call.usage,
        depth=call.req.depth,
        stop=call.stop,
        model_s=MODEL_CALL_S,
    )
    try:
        outcome = await crawler.run(runs)
    except Exception as exc:  # a failed hop keeps what the crawl already found
        outcome = crawler.outcome
        if _stopping(call):  # the Ctrl-C also reached the model process the hop waited on
            outcome.stopped = "interrupted"
        else:
            outcome.stopped = f"hop failed ({type(exc).__name__})"
            call.notes.append(f"note: crawl stopped: {type(exc).__name__}: {exc}")
    call.crawl = outcome
    return outcome.runs


async def _script(call: _Call, runner: PlanRunner) -> list[StepRun]:
    assert call.models.scripter is not None
    path = ScriptPath(
        question=call.req.question,
        when=_when(call),
        shortlist=call.shortlist,
        registry=call.registry,
        home=call.cfg.home,
        runner=runner,
        step_ids=call.step_ids,
        budget=call.budget,
        usage=call.usage,
        call_id=call.call_id,
        bound=_bounded,
    )
    runs = await path.go(call.models.scripter)
    call.script = path.run
    call.failures += path.failures
    call.gaps += path.gaps
    return runs


def _code(call: _Call) -> str | None:
    return call.script.script.code if call.script else None


def _when(call: _Call) -> str:
    now = now_in(call.cfg.timezone).isoformat(timespec="minutes")
    return call.window.describe() if call.window else now


def _reconcile(call: _Call, trace: Trace | None) -> AskRequest:
    """Requests the levels cannot honour as asked, adjusted with a note. A nested call never
    rises above the tier its guard allows, so there --script below deep is dropped."""
    if call.req.script and call.req.level != "deep":
        if trace is not None and trace.ceiling not in (None, "deep"):
            call.req = replace(call.req, script=False)
            call.notes.append(
                f"note: --script needs deep; this nested call runs at {call.req.level}"
            )
        else:
            call.req = replace(call.req, level="deep")
            call.notes.append("note: --script runs at deep; the level was raised")
    if call.req.breadth and call.req.level == "quick":
        call.req = replace(call.req, breadth=False)
        call.notes.append("note: --breadth needs more than one step; ignored at quick")
    return call.req


def _runner(call: _Call, now: datetime) -> PlanRunner:
    assert call.trace is not None
    host = WorkflowHost(call.cfg, call.registry, call.trace, call.models)
    executor = Executor(call.registry, call.cfg.tool_env(), call.cfg.spill_dir, host)
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
    found = [r for r in runs if r.found]
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
        when=_when(call),
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
    if call.level == "deep":
        findings, confidence, refine = await _deep_summary(call, runs)
    elif call.level == "standard":
        findings, confidence, refine = await _distill(call, runs)
    else:
        findings = findings_from(runs, call.registry)
        confidence, refine = ("medium" if findings else "low"), []
        call.gaps.append("quick tier: no completeness check was run")
    answer = _answer(call, runs, findings, confidence, refine, runner)
    await _log(call, answer, runs)
    return answer


async def _deep_summary(call: _Call, runs: list[StepRun]) -> tuple[list[Finding], str, list[str]]:
    """Map-reduce over everything the crawl found, or what the script reported; raw findings if
    that fails."""
    found = _summary_input(call, runs)
    if not found:
        return [], "low", []
    stopping = _stopping(call)
    try:
        distilled, source = await _summarize(call, found)
    except Exception as exc:
        if stopping or not _stopping(call):
            call.notes.append(f"note: summary failed ({type(exc).__name__}); findings are raw")
            return findings_from(found, call.registry), "low", []
        # the first Ctrl-C landed during the summary and reached its model processes; the
        # user asked to stop and see what was found, so summarise once more
        return await _deep_summary(call, runs)
    call.gaps = list(dict.fromkeys(call.gaps + distilled.gaps))
    findings = [Finding(f.text, f.pointer, source.get(f.pointer, "")) for f in distilled.findings]
    extra = backstop(found, findings, call.registry)
    call.notes += [f"note: {len(extra)} {BACKSTOP_NOTE}"] if extra else []
    return findings + extra, distilled.confidence, distilled.refine


def _stopping(call: _Call) -> bool:
    """The first Ctrl-C asked a deep crawl to stop. Model processes are Claude Code's own
    children in the terminal's process group, so that Ctrl-C reaches them too."""
    return call.stop is not None and call.stop.is_set()


def _summary_input(call: _Call, runs: list[StepRun]) -> list[StepRun]:
    """What the script reported, if it reported anything; else every result found."""
    runs = call.script.summary_runs if call.script and call.script.summary_runs else runs
    return [r for r in runs if r.found]


async def _summarize(call: _Call, found: list[StepRun]) -> tuple[Distilled, dict[str, str]]:
    assert call.models.distiller and call.models.reducer
    interpretation = call.plan.interpretation if call.plan else ""
    if call.script:
        interpretation = call.script.script.interpretation
    async with asyncio.timeout(DEEP_SUMMARY_S):  # on timeout the findings are raw
        return await summarize(
            question=call.req.question,
            interpretation=interpretation,
            when=_when(call),
            breadth=_breadth(call),
            gaps=call.gaps,
            runs=found,
            registry=call.registry,
            mapper=call.models.distiller,
            reducer=call.models.reducer,
            usage=call.usage,
        )


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
        notes=_notes(call, runs),
        refine=refine,
        breadth=breadth,
        ops_run=[r.op for r in ran],
        recipes=recipes,
        **_coverage(call, runs),
        **_meta(call),
        **_crawl_fields(call),
        script=_code(call),
    )


def _notes(call: _Call, runs: list[StepRun]) -> list[str]:
    """The call's notes, then what workflows noted (`ctx.log`, tier clamps), once per run: a
    reused lookup shares its first run's result and notes."""
    return call.notes + [
        f"note: {r.op}: {n}" for r in runs if not r.same_as for n in r.result.notes
    ]


class _CrawlFields(TypedDict):
    hops: int
    terms_searched: int
    unexpanded: list[str]
    stopped: str | None
    cost: str | None


def _crawl_fields(call: _Call) -> _CrawlFields:
    crawl = call.crawl
    return {
        "hops": crawl.hops if crawl else 0,
        "terms_searched": len(crawl.searched) if crawl else 0,
        "unexpanded": crawl.unexpanded if crawl else [],
        "stopped": crawl.stopped if crawl else None,
        "cost": call.budget.line() if call.budget else None,
    }


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
    if call.script:
        lines.append(call.script.trace_line())
    current = 0
    for run in runs:
        if run.round != current:
            current = run.round
            lines.append(f"{'hop' if call.level == 'deep' else 'replan'} {current}:")
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
                level=call.asked,  # as requested; ran_level is what ran
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
                script=_code(call),
                trace_id=call.trace.trace_id if call.trace else None,
                depth=call.trace.depth if call.trace else 0,
            )
        )
        report = [call.script.report_run] if call.script and call.script.report_run else []
        for run in runs + report:
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
