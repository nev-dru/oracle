"""Between a running script and the rest of the oracle.

Each call the script makes becomes a plan step on the call's PlanRunner, so the executor's
checks, identical-call reuse (a corrected script re-runs nothing), Coverage and the ops log
treat script calls exactly like plan steps. What the script `report()`s becomes the summary's
input: records that carry a pointer go back under the op that produced them, so the summary
reads them with that op's summary fields and cites the real pointer; anything else (counts,
groupings) becomes one `script.report` result with pointers of its own.
"""

from dataclasses import dataclass, field
from typing import Any

from oracle.budget import Budget
from oracle.deep.runner import CallRefused
from oracle.exec.dag import PlanRunner, StepRun
from oracle.exec.executor import OpResult
from oracle.plan.models import Step

REPORT_OP = "script.report"
POINTER_KEY = "_pointer"


@dataclass
class ScriptCalls:
    runner: PlanRunner
    tools: set[str]
    """The tools this call shortlisted; a script may call only their ops."""
    step_ids: set[str]
    budget: Budget
    attempt: int = 1
    runs: list[StepRun] = field(default_factory=list)
    calls: int = 0
    """Numbered before the call runs: calls under `gather` start together."""
    origin: dict[str, str] = field(default_factory=dict)
    """pointer -> op, for every record handed to the script."""

    async def __call__(self, method: str, args: dict[str, Any]) -> list[dict[str, Any]]:
        if method != "op":
            raise CallRefused(f"unknown call {method!r}")
        ref, params = str(args.get("op", "")), dict(args.get("params") or {})
        if ref.partition(".")[0] not in self.tools:
            raise CallRefused(f"{ref} is not one of this question's tools")
        self.calls += 1
        step = Step(id=f"x{self.attempt}_{self.calls}", op=ref, params=params)
        self.step_ids.add(step.id)
        run = await self.runner.run_literal(step)
        self.runs.append(run)
        if run.ran and not run.same_as:
            self.budget.tool_calls += 1
        return self._records(run.result)

    def _records(self, result: OpResult) -> list[dict[str, Any]]:
        """A call the script can fix (bad params, a query the tool rejects) raises in the
        script, so a corrected script can fix it; a tool not set up here returns [] and shows
        in Coverage."""
        if result.status in ("invalid", "error"):
            raise CallRefused(f"{result.op}: {result.error or result.status}")
        out = []
        for record, pointer in zip(result.records, result.pointers, strict=True):
            self.origin[pointer] = result.op
            out.append({**record, POINTER_KEY: pointer})
        return out


def report_runs(
    reports: list[Any], origin: dict[str, str], call_id: str
) -> tuple[list[StepRun], StepRun | None]:
    """(per-op runs of the reported records, the `script.report` run of everything else).
    Only the second is new to the log; the first repeats records the ops already logged."""
    by_op, loose = _split([r for value in reports for r in flatten(value)], origin)
    runs = [
        StepRun(f"script/report-{i}", _result(op, list(found.values()), list(found)))
        for i, (op, found) in enumerate(by_op.items(), start=1)
    ]
    pointers = [f"script:{call_id}#{i}" for i in range(1, len(loose) + 1)]
    extra = StepRun("script/report", _result(REPORT_OP, loose, pointers)) if loose else None
    return runs, extra


def _split(
    records: list[dict[str, Any]], origin: dict[str, str]
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    """(op -> pointer -> record, records without a pointer this call handed out)."""
    by_op: dict[str, dict[str, dict[str, Any]]] = {}
    loose: list[dict[str, Any]] = []
    for record in records:
        pointer = record.get(POINTER_KEY)
        clean = {k: v for k, v in record.items() if k != POINTER_KEY}
        if isinstance(pointer, str) and pointer in origin:
            by_op.setdefault(origin[pointer], {})[pointer] = clean
        else:
            loose.append(clean)
    return by_op, loose


def _result(op: str, records: list[dict[str, Any]], pointers: list[str]) -> OpResult:
    return OpResult(op=op, params={}, records=records, pointers=pointers, ids=list(pointers))


def flatten(value: Any) -> list[dict[str, Any]]:
    """Reported values as records: a record with a pointer stays whole; lists and dicts that
    hold records are opened; other fields stay together as one record."""
    if isinstance(value, list):
        return [r for item in value for r in flatten(item)]
    if not isinstance(value, dict):
        return [{"value": value}]
    if POINTER_KEY in value:
        return [value]
    rest: dict[str, Any] = {}
    nested: list[dict[str, Any]] = []
    for key, item in value.items():
        if _holds_records(item):
            nested += flatten(item)
        else:
            rest[key] = item
    return ([rest] if rest else []) + nested


def _holds_records(value: Any) -> bool:
    if isinstance(value, list):
        return any(isinstance(v, dict | list) for v in value)
    if isinstance(value, dict):
        return POINTER_KEY in value or any(_holds_records(v) for v in value.values())
    return False
