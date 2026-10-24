"""`oracle report`: implicit signals from log.sqlite, the raw material for tuning.

- a reworded question in the same session within 2 minutes: the first answer missed;
- `expand` on a pointer: that finding was used;
- calls that replanned twice: a ranking or recipe gap;
- a tool used with results that the ranker had not put in its top 3: a ranker miss, written as
  a candidate case for review (never straight into ranker_eval.yaml);
- explicit feedback from `oracle feedback`.
"""

import json
import re
import time
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import aiosqlite
import yaml

REQUERY_S = 120
# rows that did not fail: reused lookups, tools not active here, steps skipped for lack of input
NOT_FAILURES = ("reused ", "not registered here", "needs results from")
WORD = re.compile(r"[a-z0-9]{3,}")


@dataclass
class Report:
    since: float
    calls: int = 0
    levels: dict[str, int] = field(default_factory=dict)
    requeries: list[tuple[str, str]] = field(default_factory=list)
    pointers_shown: int = 0
    pointers_expanded: int = 0
    replanned_twice: list[str] = field(default_factory=list)
    ranker_misses: list[dict[str, Any]] = field(default_factory=list)
    failing_ops: dict[str, int] = field(default_factory=dict)
    feedback: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        used = f"{self.pointers_expanded}/{self.pointers_shown}" if self.pointers_shown else "0/0"
        start = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.since))
        out = [
            f"calls since {start}: {self.calls}  by level {self.levels}",
            f"re-asked within {REQUERY_S} s (first answer likely missed): {len(self.requeries)}",
            *[f"  {a!r} -> {b!r}" for a, b in self.requeries[:10]],
            f"pointers expanded (findings used): {used}",
            f"replanned twice (ranking or recipe gap): {len(self.replanned_twice)}",
            *[f"  {q!r}" for q in self.replanned_twice[:10]],
            f"ranker misses (tool used, not in its top 3): {len(self.ranker_misses)}",
            *[f"  {m['query']!r} needed {m['expect']}" for m in self.ranker_misses[:10]],
            f"failing ops: {self.failing_ops or 'none'}",
            f"feedback: {self.feedback or 'none'}",
            *[f"  note: {n}" for n in self.notes[:10]],
        ]
        return out


async def build_report(db: aiosqlite.Connection, since: float) -> Report:
    report = Report(since=since)
    async with db.execute(
        "SELECT call_id, session, ts, query, level, ranked_tools, replans, breadth FROM calls "
        "WHERE ts >= ? ORDER BY session, ts",
        (since,),
    ) as cur:
        calls = list(await cur.fetchall())
    report.calls = len(calls)
    for row in calls:
        report.levels[row[4]] = report.levels.get(row[4], 0) + 1
        if (row[6] or 0) >= 2:
            report.replanned_twice.append(row[3])
    report.requeries = _requeries(calls)
    ops = await _ops(db, since)
    report.ranker_misses = _ranker_misses(calls, ops)
    await _pointers(db, since, report)
    _failures(ops, report)
    await _feedback(db, since, report)
    return report


def _words(text: str) -> set[str]:
    return set(WORD.findall(text.lower()))


def _requeries(calls: list[Any]) -> list[tuple[str, str]]:
    """Consecutive calls in one session, close in time, worded differently but on the same
    subject (at least one shared word)."""
    out = []
    for prev, cur in pairwise(calls):
        same_session = prev[1] == cur[1] and not str(cur[1]).startswith("adhoc-")
        close = cur[2] - prev[2] <= REQUERY_S
        reworded = prev[3] != cur[3] and _words(prev[3]) & _words(cur[3])
        if same_session and close and reworded:
            out.append((prev[3], cur[3]))
    return out


async def _ops(db: aiosqlite.Connection, since: float) -> dict[str, list[tuple[Any, ...]]]:
    """Every op row of the calls in the window, in one query: call_id -> (step, op, ok, n,
    error)."""
    out: dict[str, list[tuple[Any, ...]]] = {}
    async with db.execute(
        "SELECT o.call_id, o.step_id, o.op, o.ok, o.n_results, o.error FROM ops o "
        "JOIN calls c ON c.call_id = o.call_id WHERE c.ts >= ?",
        (since,),
    ) as cur:
        for call_id, *row in await cur.fetchall():
            out.setdefault(call_id, []).append(tuple(row))
    return out


def _ranker_misses(calls: list[Any], ops: dict[str, list[tuple[Any, ...]]]) -> list[dict[str, Any]]:
    """A tool the planner reached for directly (not via a recipe stage, not in a breadth
    fan-out, where every tool is meant to be used) with results, that the ranker had not put in
    its top 3."""
    misses: dict[tuple[str, str], dict[str, Any]] = {}
    for call_id, _session, _ts, query, _level, ranked_json, _replans, breadth in calls:
        if breadth:
            continue
        ranked = [r["tool"] for r in json.loads(ranked_json or "[]") if "tool" in r][:3]
        for tool in sorted(_planned_tools(ops.get(call_id, [])) - set(ranked)):
            misses[(query, tool)] = {"query": query, "expect": [tool], "call_id": call_id}
    return list(misses.values())


def _planned_tools(rows: list[tuple[Any, ...]]) -> set[str]:
    """Tools a plan step used directly with results; recipe stages (`s1/stage`) do not count."""
    return {op.split(".", 1)[0] for step, op, ok, n, _error in rows if ok and n and "/" not in step}


async def _pointers(db: aiosqlite.Connection, since: float, report: Report) -> None:
    async with db.execute(
        "SELECT COUNT(*), COUNT(expanded_at) FROM pointers WHERE ts >= ?", (since,)
    ) as cur:
        row = await cur.fetchone()
    report.pointers_shown, report.pointers_expanded = (row or (0, 0))[0], (row or (0, 0))[1]


def _failures(ops: dict[str, list[tuple[Any, ...]]], report: Report) -> None:
    for rows in ops.values():
        for _step, op, ok, _n, error in rows:
            if not ok and not str(error or "").startswith(NOT_FAILURES):
                report.failing_ops[op] = report.failing_ops.get(op, 0) + 1


async def _feedback(db: aiosqlite.Connection, since: float, report: Report) -> None:
    async with db.execute("SELECT useful, note FROM feedback WHERE ts >= ?", (since,)) as cur:
        for useful, note in await cur.fetchall():
            key = "useful" if useful else "not useful"
            report.feedback[key] = report.feedback.get(key, 0) + 1
            if note:
                report.notes.append(note)


def write_candidates(report: Report, path: Path) -> int:
    """Ranker misses as candidate cases, for a person to review into ranker_eval.yaml."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cases = [{"query": m["query"], "expect": m["expect"]} for m in report.ranker_misses]
    path.write_text(yaml.safe_dump({"version": 1, "cases": cases}, sort_keys=False))
    return len(cases)
