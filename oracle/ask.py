"""One oracle invocation: rank -> plan -> execute -> log -> answer.

M1 ships the quick tier only: one op chosen and filled by the planner model, no completeness
check, findings straight from summary fields. Other levels run as quick and say so in the trace.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic_ai.models import Model

from oracle.config import Config
from oracle.exec.executor import Executor, OpResult
from oracle.llm.models import resolve_model
from oracle.log.store import CallRecord, open_store
from oracle.plan.planner import PlanOutcome, build_planner, make_plan
from oracle.rank.index import RankedTool, Ranker
from oracle.registry.loader import Registry
from oracle.render import Answer, findings_from, trace_line

QUICK_STEPS = 1
BUILT_LEVELS = ("quick",)


@dataclass
class AskRequest:
    question: str
    intent: str | None = None
    level: str = "standard"
    scope: list[str] | None = None
    prefer: list[str] | None = None
    session: str | None = None


def now_in(tz: str | None) -> datetime:
    return datetime.now(ZoneInfo(tz)) if tz else datetime.now().astimezone()


@dataclass
class _Call:
    req: AskRequest
    cfg: Config
    registry: Registry
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started: float = field(default_factory=time.monotonic)

    @property
    def session(self) -> str:
        return self.req.session or f"adhoc-{self.call_id}"


async def ask(
    req: AskRequest, cfg: Config, registry: Registry, planner_model: Model | str | None = None
) -> Answer:
    call = _Call(req, cfg, registry)
    shortlist = Ranker(registry).rank(req.question, scope=req.scope, prefer=req.prefer)
    if not shortlist:
        return await _finish(call, shortlist, gap="no active tool matches the question or --scope")
    model = planner_model or resolve_model(cfg.model("planner"))
    try:
        outcome = await make_plan(
            build_planner(model),
            question=req.question,
            intent=req.intent,
            shortlist=shortlist,
            registry=registry,
            max_steps=QUICK_STEPS,
            now=now_in(cfg.timezone),
        )
    except Exception as exc:  # any model or transport failure is reported, never a traceback
        return await _finish(call, shortlist, failure=f"planner: {type(exc).__name__}: {exc}")
    executor = Executor(registry, cfg.tool_env(), cfg.spill_dir)
    steps = outcome.plan.steps
    results = await asyncio.gather(*(executor.run(s.op, s.params) for s in steps))
    return await _finish(call, shortlist, outcome, list(results))


async def _finish(
    call: _Call,
    shortlist: list[RankedTool],
    outcome: PlanOutcome | None = None,
    results: list[OpResult] | None = None,
    *,
    failure: str | None = None,
    gap: str | None = None,
) -> Answer:
    results = results or []
    steps = [s.id for s in outcome.plan.steps] if outcome else []
    answer = build_answer(
        call.call_id, call.session, call.registry, list(zip(steps, results, strict=True))
    )
    answer.failures += [failure] if failure else []
    answer.gaps += [gap] if gap else []
    if call.req.level not in BUILT_LEVELS:
        answer.notes.append(f"note: level {call.req.level} is not built yet (M2); ran quick")
    wall_ms = int((time.monotonic() - call.started) * 1000)
    await _log(call, answer, shortlist, outcome, results, wall_ms)
    return answer


def build_answer(
    call_id: str, session: str, registry: Registry, steps: list[tuple[str, OpResult]]
) -> Answer:
    results = [r for _, r in steps]
    ok = [r for r in results if r.ok]
    failures = [f"{r.op} ({r.status}): {r.error}" for r in results if not r.ok]
    findings = findings_from(ok, registry)
    gaps = quick_gaps(ok, bool(findings))
    return Answer(
        call_id=call_id,
        session=session,
        level="quick",
        findings=findings,
        searched=sorted(tools_of(ok)),
        not_searched=sorted(set(registry.tools) - tools_of(results)),
        unavailable=sorted(registry.inactive),
        failures=failures,
        confidence="medium" if findings and not failures else "low",
        gaps=gaps,
        trace=[trace_line(step_id, r) for step_id, r in steps],
    )


def quick_gaps(ok: list[OpResult], found: bool) -> list[str]:
    gaps = ["quick tier: no completeness check was run"] + ([] if found else ["no results"])
    return gaps + [
        f"{r.op} output was truncated; full output at `{r.spill}`" for r in ok if r.spill
    ]


def tools_of(results: list[OpResult]) -> set[str]:
    return {r.op.split(".", 1)[0] for r in results}


async def _log(
    call: _Call,
    answer: Answer,
    shortlist: list[RankedTool],
    outcome: PlanOutcome | None,
    results: list[OpResult],
    wall_ms: int,
) -> None:
    async with open_store(call.cfg.log_path) as store:
        await store.record_call(
            CallRecord(
                call_id=answer.call_id,
                session=answer.session,
                query=call.req.question,
                intent=call.req.intent,
                level=answer.level,
                ranked_tools=[{"tool": t.tool, "score": t.score} for t in shortlist],
                plan=outcome.plan.model_dump() if outcome else None,
                wall_ms=wall_ms,
                tokens_in=outcome.tokens_in if outcome else 0,
                tokens_out=outcome.tokens_out if outcome else 0,
                model_calls=outcome.model_calls if outcome else 0,
                confidence=answer.confidence,
                gaps=answer.gaps + answer.failures,
            )
        )
        steps = outcome.plan.steps if outcome else []
        for step, result in zip(steps, results, strict=True):
            await store.record_op(answer.call_id, answer.session, step.id, result)
