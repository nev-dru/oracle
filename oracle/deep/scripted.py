"""The deep tier's script path: write a script against the shortlisted tools, run it, and on a
run-time failure send it back once with the traceback (its calls are reused, not re-run). The
summary reads what the script reported; Coverage and the log see every call it made."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_ai.models import Model

from oracle.budget import Budget
from oracle.deep.bridge import ScriptCalls, report_runs
from oracle.deep.runner import ScriptResult, run_script
from oracle.deep.script_agent import Script, build_script_agent, write_script
from oracle.exec.dag import PlanRunner, StepRun
from oracle.plan.planner import Shortlist
from oracle.registry.codegen import build_library
from oracle.registry.loader import Registry

ATTEMPTS = 2
PRINTED_TAIL = 2000  # chars of the script's print output sent back with a traceback


@dataclass
class ScriptRun:
    script: Script
    result: ScriptResult
    attempts: int
    earlier: list[Any] = field(default_factory=list)
    """What the first attempt reported, kept in case the second reports nothing."""
    summary_runs: list[StepRun] = field(default_factory=list)
    report_run: StepRun | None = None
    """The `script.report` result: logged so its pointers expand, but not a tool's result,
    so it stays out of Coverage, the trace and the ops run."""

    def trace_line(self) -> str:
        state = "ok" if self.result.ok else "failed"
        return f"script: attempt {self.attempts}, {self.result.calls} tool calls, {state}"


@dataclass
class ScriptPath:
    question: str
    when: str
    shortlist: Shortlist
    registry: Registry
    home: Path
    runner: PlanRunner
    step_ids: set[str]
    budget: Budget
    call_id: str
    run: ScriptRun | None = None
    failures: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)

    async def go(self, model: Model | str) -> list[StepRun]:
        library = build_library(self.registry, self.home)
        agent = build_script_agent(model)
        tools = {t.tool for t in self.shortlist.tools}
        calls = ScriptCalls(self.runner, tools, self.step_ids, self.budget)
        failed: tuple[str, str] | None = None
        cost = 0
        for attempt in range(1, ATTEMPTS + 1):
            if attempt > 1 and self._spent(cost):
                self.gaps.append("no second attempt: the budget or the time is spent")
                break
            before = self.budget.spent
            try:
                script = await write_script(
                    agent,
                    question=self.question,
                    when=self.when,
                    shortlist=self.shortlist,
                    library=library,
                    budget=self.budget,
                    failed=failed,
                )
            except Exception as exc:  # reported like a planner failure, never a traceback
                self.failures.append(f"script writer: {type(exc).__name__}: {exc}")
                break
            cost = self.budget.spent - before
            calls.attempt, calls.calls = attempt, 0
            result = await _run_safely(script.code, calls, library.path)
            earlier = self.run.result.reports if self.run else []
            self.run = ScriptRun(script, result, attempt, earlier)
            if result.ok:
                break
            failed = (script.code, _failure(result))
        return self._runs(calls)

    def _spent(self, cost: int) -> bool:
        """A correction would cost about what the first script did; `--budget` and the deep
        wall clock bound the script path as they bound a crawl."""
        return not self.budget.admits(cost) or self.budget.out_of_time()

    def _runs(self, calls: ScriptCalls) -> list[StepRun]:
        run = self.run
        if run is None:
            return calls.runs
        if not run.result.ok:
            last = (run.result.error or "").strip().splitlines()[-1:] or ["unknown error"]
            self.gaps.append(f"the script failed after {run.attempts} attempt(s): {last[0]}")
        if run.result.truncated:
            self.gaps.append("the script reported more than the cap; the rest was dropped")
        summary, extra = report_runs(run.result.reports or run.earlier, calls.origin, self.call_id)
        run.summary_runs = summary + ([extra] if extra else [])
        run.report_run = extra
        return calls.runs


async def _run_safely(code: str, calls: ScriptCalls, libdir: Path) -> ScriptResult:
    """run_script, with a failure to start or serve the child reported like a script error."""
    try:
        return await run_script(code, calls, libdir)
    except Exception as exc:
        return ScriptResult(error=f"the script runner failed: {type(exc).__name__}: {exc}")


def _failure(result: ScriptResult) -> str:
    """What the writer sees on a retry: the traceback, and the end of what the script printed."""
    printed = result.printed[-PRINTED_TAIL:].strip()
    return (result.error or "") + (f"\n\nThe script printed:\n{printed}" if printed else "")
