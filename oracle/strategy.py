"""How one call gathers results, between ranking and the summary: one strategy per tier.

quick plans and runs. standard plans, runs, then checks for gaps and replans at most twice. deep
plans, runs, then crawls what the results mention, or (`--script`) writes one script against
the tools instead of planning. Each returns an Outcome: every run, and what the summary, the
answer and the log need to know about how they were found. A strategy reads its Context and
changes nothing else; everything it has to say is on the Outcome.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Protocol
from zoneinfo import ZoneInfo

from oracle.budget import Budget, Interrupted
from oracle.config import REPO_ROOT, Config
from oracle.deep.scripted import ScriptPath
from oracle.exec.dag import PlanRunner, StepRun
from oracle.exec.executor import Executor
from oracle.plan.check import Check, build_checker, run_check
from oracle.plan.digest import summary_of
from oracle.plan.models import Plan
from oracle.plan.planner import Shortlist, build_planner, make_plan
from oracle.rank.index import Ranker
from oracle.registry.loader import Registry
from oracle.render import tool_of
from oracle.sdk.crawl import Crawler
from oracle.sdk.extract import build_extractor
from oracle.sdk.hop import build_hop_planner
from oracle.skills.loader import load_skills, render, select
from oracle.timewin import TimeWindow, default_builtins, resolve_window

if TYPE_CHECKING:
    from oracle.ask import AskRequest, Models

SKILLS = load_skills(REPO_ROOT / "skills")
MAX_STEPS = {"quick": 1, "standard": 12, "deep": 12}
MAX_REPLANS = 2


def now_in(tz: str | None) -> datetime:
    return datetime.now(ZoneInfo(tz)) if tz else datetime.now().astimezone()


@dataclass
class Context:
    """What a strategy reads: one call's request, tools, models and budget."""

    req: "AskRequest"
    cfg: Config
    registry: Registry
    models: "Models"
    budget: Budget
    ranker: Ranker
    runner: PlanRunner
    """One runner for the whole call, so an identical lookup is never run twice."""
    shortlist: Shortlist
    window: TimeWindow | None
    now: datetime
    """When the call started, in the user's zone: the window and the plan are read from it."""
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @classmethod
    def build(
        cls,
        req: "AskRequest",
        cfg: Config,
        registry: Registry,
        models: "Models",
        budget: Budget,
        executor: Executor,
    ) -> "Context":
        now = now_in(cfg.timezone)
        window = resolve_window(req.question, now)
        ranker = Ranker(registry)
        builtins = window.builtins() if window else default_builtins(now)
        runner = PlanRunner(registry, executor, builtins=builtins)
        shortlist = _shortlist(ranker, req.question, req)
        return cls(req, cfg, registry, models, budget, ranker, runner, shortlist, window, now)

    @property
    def level(self) -> str:
        return self.req.level

    def when(self) -> str:
        """The question's time window, else now (taken fresh, so a long call says when)."""
        now = now_in(self.cfg.timezone).isoformat(timespec="minutes")
        return self.window.describe() if self.window else now

    def shortlist_for(self, query: str) -> Shortlist:
        return _shortlist(self.ranker, query, self.req)


@dataclass
class Outcome:
    """What a strategy hands back: every run, and what the summary, the answer and the log need
    to know about how they were found."""

    runs: list[StepRun] = field(default_factory=list)
    plan: Plan | None = None
    interpretation: str = ""
    replans: int = 0
    gaps: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    reported: list[StepRun] = field(default_factory=list)
    """What a script reported; the summary reads it in place of every result found."""
    logged: list[StepRun] = field(default_factory=list)
    """Runs the log keeps that are no tool's result (a script's report, so its pointers expand)."""
    trace: list[str] = field(default_factory=list)
    """Lines the trace starts with, before the runs."""
    round_name: str = "replan"
    """What the trace calls each later round of runs."""
    hops: int = 0
    searched: list[str] = field(default_factory=list)
    unexpanded: list[str] = field(default_factory=list)
    stopped: str | None = None
    """Why a crawl stopped."""
    script: str | None = None
    """The code of the script that ran."""

    @property
    def summary_runs(self) -> list[StepRun]:
        """What the summary reads: what a script reported, if anything; else every result."""
        return [r for r in (self.reported or self.runs) if r.found]


class Strategy(Protocol):
    async def run(self, ctx: Context) -> Outcome: ...


def strategy_for(level: str, script: bool) -> Strategy:
    """How a call at this tier gathers; --script reaches here only at deep."""
    if script:
        return Scripted()
    return {"quick": Planned(), "standard": Checked(), "deep": Crawled()}[level]


class Planned:
    """quick: plan, then run the plan."""

    round_name = "replan"

    async def run(self, ctx: Context) -> Outcome:
        out = Outcome(round_name=self.round_name)
        assert ctx.models.planner is not None
        try:
            plan = await make_plan(
                build_planner(ctx.models.planner),
                question=ctx.req.question,
                intent=ctx.req.intent,
                shortlist=ctx.shortlist,
                registry=ctx.registry,
                max_steps=MAX_STEPS[ctx.level],
                now=ctx.now,
                window=ctx.window,
                budget=ctx.budget,
                breadth=ctx.req.breadth,
                guidance=_guidance(ctx, "planner", ctx.shortlist, out.notes),
            )
        except Exception as exc:  # any model or transport failure is reported, never a traceback
            out.failures.append(f"planner: {type(exc).__name__}: {exc}")
            return out
        out.plan, out.interpretation = plan, plan.interpretation
        runs = await ctx.runner.run(plan.steps)
        out.runs = await self.follow_up(ctx, out, runs, {s.id for s in plan.steps})
        return out

    async def follow_up(
        self, ctx: Context, out: Outcome, runs: list[StepRun], step_ids: set[str]
    ) -> list[StepRun]:
        """After the plan has run; `step_ids` are the plan-level ids later steps may bind to."""
        return runs


class Checked(Planned):
    """standard: plan, run, then check for gaps and replan at most twice."""

    async def follow_up(
        self, ctx: Context, out: Outcome, runs: list[StepRun], step_ids: set[str]
    ) -> list[StepRun]:
        assert out.plan is not None and ctx.models.checker is not None
        checker = build_checker(ctx.models.checker)
        planned = len(out.plan.steps)
        while not ctx.budget.out_of_time():
            last = out.replans >= MAX_REPLANS or planned >= MAX_STEPS["standard"]
            conditioned = _conditioned(ctx, out.gaps, runs)
            try:
                check = await run_check(
                    checker,
                    question=ctx.req.question,
                    intent=ctx.req.intent,
                    plan=out.plan,
                    window=ctx.window,
                    runs=runs,
                    shortlist=conditioned,
                    registry=ctx.registry,
                    max_steps=0 if last else MAX_STEPS["standard"] - planned,
                    budget=ctx.budget,
                    earlier=step_ids,
                    guidance=_guidance(ctx, "check", conditioned, out.notes),
                )
            except Exception as exc:  # a failed check leaves the results as they are
                out.notes.append(f"note: completeness check failed ({type(exc).__name__}: {exc})")
                return runs
            out.gaps = list(check.gaps)  # the latest check's gaps, which the next re-rank reads
            if not _replan(out, check, planned):
                return runs
            out.replans += 1
            planned += len(check.next_steps)
            step_ids |= {s.id for s in check.next_steps}
            runs = runs + await ctx.runner.run(check.next_steps, round_=out.replans)
        out.notes.append(f"note: stopped checking at the {ctx.budget.bounds.follow_up_s:g} s bound")
        return runs


class Crawled(Planned):
    """deep: plan, run, then crawl what the results mention."""

    round_name = "hop"

    async def follow_up(
        self, ctx: Context, out: Outcome, runs: list[StepRun], step_ids: set[str]
    ) -> list[StepRun]:
        assert ctx.models.extractor and ctx.models.planner
        crawler = Crawler(
            question=ctx.req.question,
            registry=ctx.registry,
            runner=ctx.runner,
            extractor=build_extractor(ctx.models.extractor),
            hop_planner=build_hop_planner(ctx.models.planner),
            budget=ctx.budget,
            shortlist_for=ctx.shortlist_for,
            step_ids=step_ids,
            depth=ctx.req.depth,
        )
        try:
            crawl = await crawler.run(runs)
        except Interrupted:  # the Ctrl-C also reached the model process the hop waited on
            crawl = crawler.outcome
            crawl.stopped = "interrupted"
        except Exception as exc:  # a failed hop keeps what the crawl already found
            crawl = crawler.outcome
            crawl.stopped = f"hop failed ({type(exc).__name__})"
            out.notes.append(f"note: crawl stopped: {type(exc).__name__}: {exc}")
        out.hops, out.searched, out.unexpanded = crawl.hops, crawl.searched, crawl.unexpanded
        out.stopped = crawl.stopped
        return crawl.runs


class Scripted:
    """deep with --script: write one script against the shortlisted tools and run it."""

    async def run(self, ctx: Context) -> Outcome:
        assert ctx.models.scripter is not None
        path = ScriptPath(
            question=ctx.req.question,
            when=ctx.when(),
            shortlist=ctx.shortlist,
            registry=ctx.registry,
            home=ctx.cfg.home,
            runner=ctx.runner,
            step_ids=set(),
            budget=ctx.budget,
            call_id=ctx.call_id,
        )
        runs = await path.go(ctx.models.scripter)
        out = Outcome(runs, gaps=path.gaps, failures=path.failures, round_name="hop")
        if (run := path.run) is not None:  # None when the writer failed on the first attempt
            out.interpretation = run.script.interpretation
            out.reported = run.summary_runs
            out.logged = [run.report_run] if run.report_run else []
            out.trace = [run.trace_line()]
            out.script = run.script.code
        return out


def _shortlist(ranker: Ranker, query: str, req: "AskRequest") -> Shortlist:
    tools = ranker.rank(query, scope=req.scope, prefer=req.prefer, breadth=req.breadth)
    recipes = ranker.rank_recipes(query)
    if req.scope:
        active = set(ranker.registry.tools)
        recipes = [r for r in recipes if _recipe_tools(ranker, r) & active <= set(req.scope)]
    return Shortlist(tools, recipes)


def _recipe_tools(ranker: Ranker, name: str) -> set[str]:
    return {tool_of(s.op) for s in ranker.registry.recipes[name].steps}


def _guidance(ctx: Context, role: str, shortlist: Shortlist, notes: list[str]) -> str:
    tools = [t.tool for t in shortlist.tools]
    chosen = select(SKILLS, role, tools, ctx.registry, ctx.req.question)
    if chosen and role == "planner":
        notes.append(f"skills: {', '.join(s.name for s in chosen)}")
    return render(chosen)


def _replan(out: Outcome, check: Check, planned: int) -> bool:
    if check.sufficient or check.complete or not check.next_steps:
        return False
    if out.replans >= MAX_REPLANS or planned >= MAX_STEPS["standard"]:
        out.notes.append("note: the check wanted more steps; replan limit reached")
        return False
    return True


def _conditioned(ctx: Context, gaps: list[str], runs: list[StepRun]) -> Shortlist:
    """Re-rank on the question plus what was found, so a result mentioning a ticket brings
    acli into reach even if the question never mentioned tickets."""
    seen = [summary_of(rec, []) for run in runs if run.result.ok for rec in run.result.records[:5]]
    query = " ".join([ctx.req.question, *gaps, *seen])[:4000]
    return ctx.shortlist_for(query)
