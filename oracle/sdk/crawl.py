"""The deep crawl: search, read what the results mention, search those, repeat.

Each hop: extract terms from the hop's new results (Haiku) -> admit them by relevance within the
budget -> plan the whole hop in one call (Sonnet) -> validate -> run in parallel. One PlanRunner
serves the whole crawl, so an identical lookup is never run twice across hops. Stops at the hop
depth, when no new terms appear, or when the budget, time or tool-call cap runs out; every
term found but not searched is reported with why.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic_ai import Agent

from oracle.budget import Budget
from oracle.exec.bindings import is_binding
from oracle.exec.dag import PlanRunner, StepRun
from oracle.plan.models import Step
from oracle.plan.planner import Shortlist
from oracle.registry.loader import Registry
from oracle.registry.schema import Recipe
from oracle.render import Finding, findings_from
from oracle.sdk.extract import Term, Terms, extract_terms, key
from oracle.sdk.hop import STEPS_PER_TERM, HopContext, HopPlan, plan_hop

EXTRACT_BASE = 1500  # tokens: instructions plus the question, before the results
PLAN_BASE = 3000  # tokens: instructions plus the shortlist
PER_TERM = 300  # tokens the hop plan spends per admitted term


@dataclass
class CrawlOutcome:
    runs: list[StepRun]
    hops: int = 1
    searched: list[str] = field(default_factory=list)
    unexpanded: list[str] = field(default_factory=list)
    stopped: str = "depth"


@dataclass
class Crawler:
    question: str
    registry: Registry
    runner: PlanRunner
    extractor: Agent[None, Terms]
    hop_planner: Agent[HopContext, HopPlan]
    budget: Budget
    shortlist_for: Callable[[str], Shortlist]
    step_ids: set[str]
    depth: int = 3
    seen: set[str] = field(default_factory=set)
    outcome: CrawlOutcome = field(default_factory=lambda: CrawlOutcome(runs=[]))

    async def run(self, first: list[StepRun]) -> CrawlOutcome:
        out = self.outcome = CrawlOutcome(runs=list(first))
        for run in first:
            run.hop = 1
        self.budget.tool_calls += _invoked(first)  # the first plan's searches count too
        new = _fresh(first)
        while (reason := self._stop_reason(out.hops, new)) is None:
            hop_runs = await self._hop(out, new)
            if hop_runs is None:
                break
            out.runs += hop_runs
            new = _fresh(hop_runs)
        else:
            out.stopped = reason
        return out

    def _stop_reason(self, hops: int, new: list[StepRun]) -> str | None:
        if hops >= self.depth:
            return "depth"
        if not new:
            return "no new results"
        if self.budget.stopping():
            return "interrupted"
        if self.budget.out_of_time():
            return "time"
        if not self.budget.admits(EXTRACT_BASE + _chars(new) // 4):
            return "budget"
        return None

    async def _hop(self, out: CrawlOutcome, new: list[StepRun]) -> list[StepRun] | None:
        terms = await extract_terms(
            self.extractor,
            question=self.question,
            runs=new,
            seen=self.seen,
            registry=self.registry,
            budget=self.budget,
        )
        self.seen |= {key(t.value) for t in terms}
        admitted, rest = self._admit(terms)
        out.unexpanded += [f"{t.value} (budget)" for t in rest]
        if not admitted:
            out.stopped = "budget" if rest else "no new terms"
            return None
        hop = out.hops + 1
        query = " ".join([self.question, *(t.description for t in admitted)])
        plan = await plan_hop(
            self.hop_planner,
            hop=hop,
            question=self.question,
            terms=admitted,
            shortlist=self.shortlist_for(query),
            registry=self.registry,
            earlier=self.step_ids,
            budget=self.budget,
        )
        searched = _searched(out, admitted, plan)
        if not plan.steps:
            out.stopped = "no searchable terms"
            return None
        runs = await self._run_hop(hop, plan, {key(t.value): t for t in admitted})
        out.hops = hop  # only once the hop's runs are in hand, so the count never runs ahead
        out.searched += [t.value for t in searched]
        return runs

    def _admit(self, terms: list[Term]) -> tuple[list[Term], list[Term]]:
        """Every term the budget can afford, best first; with an unlimited budget, all of them."""
        affordable = len(terms)
        if not self.budget.unlimited:
            room = self.budget.available() - PLAN_BASE
            affordable = max(0, int(room // PER_TERM))
        calls = self.budget.tool_calls_left() / STEPS_PER_TERM
        n = int(min(affordable, calls, len(terms)))
        return terms[:n], terms[n:]

    async def _run_hop(self, hop: int, plan: HopPlan, terms: dict[str, Term]) -> list[StepRun]:
        steps = [Step(id=s.id, op=s.op, params=s.params, after=s.after) for s in plan.steps]
        parent = {s.id: s.term for s in plan.steps}
        self.step_ids |= {s.id for s in steps}
        runs = await self.runner.run(steps, round_=hop)
        for run in runs:
            run.hop = hop
            run.parent = parent.get(run.step_id.split("/")[0])
            term = terms.get(key(run.parent)) if run.parent else None
            if term is not None:
                run.why, run.relevance = term.description, term.relevance
        self.budget.tool_calls += _invoked(runs)
        return runs


def _invoked(runs: list[StepRun]) -> int:
    return sum(1 for r in runs if r.ran and not r.same_as)


def _searched(out: CrawlOutcome, admitted: list[Term], plan: HopPlan) -> list[Term]:
    """The admitted terms the hop plan gave steps; the rest go to `unexpanded`."""
    planned = {key(s.term) for s in plan.steps}
    out.unexpanded += [f"{t.value} (no op can look it up)" for t in admitted
                       if key(t.value) not in planned]  # fmt: skip
    return [t for t in admitted if key(t.value) in planned]


def _fresh(runs: list[StepRun]) -> list[StepRun]:
    return [r for r in runs if r.found]


def _chars(runs: list[StepRun]) -> int:
    return sum(len(str(rec)) for r in runs for rec in r.result.records[:12])


BACKSTOP_MIN = 0.5  # below this the extractor called the term incidental
BACKSTOP_MAX = 3


def uncited_followed(
    runs: list[StepRun], cited: set[str], recipes: dict[str, Recipe]
) -> list[StepRun]:
    """The primary lookup of each followed term that no finding cites, most relevant first.
    The crawl chose to follow these; a summary that drops one loses a link of the chain (the
    runbook's author), so the caller lists them as plain findings. The primary lookup is the
    term's first step with results: the hop planner puts the lookup of what a term is first,
    and a recipe's first stage is its main one. A recipe stage bound to an earlier stage's
    result (a learned chain's author lookup) is a link of its own, with its own primary."""
    primary: dict[tuple[str, str], StepRun] = {}
    for run in _fresh(runs):
        if run.parent:
            primary.setdefault((key(run.parent), _link(run, recipes)), run)
    missed = [r for r in primary.values() if _relevance(r) >= BACKSTOP_MIN]
    missed = [r for r in missed if not set(r.result.pointers) & cited]
    return sorted(missed, key=lambda r: -_relevance(r))[:BACKSTOP_MAX]


def _link(run: StepRun, recipes: dict[str, Recipe]) -> str:
    """The stage id when the run is a recipe stage that looks up what an earlier stage
    returned, else "" (the stage shares its term's primary lookup)."""
    recipe = recipes.get(run.recipe) if run.recipe else None
    stage = run.step_id.rpartition("/")[2]
    for s in (recipe.steps + recipe.on_empty) if recipe else []:
        if s.id == stage and any(is_binding(v) for v in s.params.values()):
            return stage
    return ""


def _relevance(run: StepRun) -> float:
    return run.relevance or 0.0


BACKSTOP_NOTE = "result(s) the crawl followed but the summary left out are listed as found"


def backstop(found: list[StepRun], findings: list[Finding], registry: Registry) -> list[Finding]:
    """Followed terms the summary left out, as plain findings from their primary lookup."""
    out = []
    for run in uncited_followed(found, {f.pointer for f in findings}, registry.recipes):
        first = findings_from([run], registry)[0]
        text = f"{first.text} (followed {run.parent!r}" + (f": {run.why})" if run.why else ")")
        out.append(Finding(text, first.pointer, first.source))
    return out
