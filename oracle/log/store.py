"""log.sqlite: one record per invocation plus the per-session pointer cache `oracle expand` reads.

WAL mode, because the main agent may run several oracle calls in parallel.
"""

import json
import sqlite3
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiosqlite

from oracle.exec.executor import OpResult

EXPAND_GRACE_S = 24 * 3600

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
  call_id TEXT PRIMARY KEY, session TEXT, ts REAL, query TEXT, intent TEXT, level TEXT,
  breadth INTEGER, ranked_tools TEXT, plan TEXT, replans INTEGER, escalated INTEGER,
  wall_ms INTEGER, tokens_in INTEGER, tokens_out INTEGER, model_calls INTEGER,
  confidence TEXT, gaps TEXT, ran_level TEXT, script TEXT, trace_id TEXT, depth INTEGER
);
CREATE TABLE IF NOT EXISTS ops (
  call_id TEXT, step_id TEXT, op TEXT, params TEXT, ok INTEGER, error TEXT,
  n_results INTEGER, ms INTEGER, truncated INTEGER, bytes INTEGER, hop INTEGER, parent TEXT,
  fields TEXT
);
CREATE TABLE IF NOT EXISTS pointers (
  session TEXT, pointer TEXT, call_id TEXT, op TEXT, record TEXT, ts REAL, expanded_at REAL,
  PRIMARY KEY (session, pointer)
);
CREATE TABLE IF NOT EXISTS feedback (
  call_id TEXT, source TEXT, useful INTEGER, note TEXT, ts REAL
);
CREATE INDEX IF NOT EXISTS ops_call ON ops (call_id);
CREATE INDEX IF NOT EXISTS pointers_ptr ON pointers (pointer, ts);
"""


@dataclass
class CallRecord:
    call_id: str
    session: str
    query: str
    intent: str | None
    level: str
    """As requested by the caller."""
    ranked_tools: list[dict[str, Any]]
    plan: dict[str, Any] | None
    wall_ms: int
    tokens_in: int = 0
    tokens_out: int = 0
    model_calls: int = 0
    breadth: bool = False
    confidence: str | None = None
    gaps: list[str] = field(default_factory=list)
    ran_level: str | None = None
    """The level that actually ran (deep runs as standard until M4)."""
    replans: int = 0
    script: str | None = None
    """Deep script: the code that ran, so a successful one can become a recipe."""
    trace_id: str | None = None
    """Shared by every call in one tree (main agent -> oracle -> workflow -> oracle)."""
    depth: int = 0


@dataclass
class Expanded:
    pointer: str
    op: str
    call_id: str
    record: dict[str, Any]


FIELD_RECORDS = 50
FIELD_CHARS = 200


def scalar_fields(record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """A record's scalar values by dotted name (`metadata.name`), lists left out, long strings
    cut: what a chain binds on."""
    out: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, dict):
            out |= scalar_fields(value, f"{prefix}{key}.")
        elif not isinstance(value, list):
            out[f"{prefix}{key}"] = value[:FIELD_CHARS] if isinstance(value, str) else value
    return out


class Store:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def record_call(self, c: CallRecord) -> None:
        await self.db.execute(
            "INSERT INTO calls (call_id, session, ts, query, intent, level, breadth, ranked_tools,"
            " plan, replans, escalated, wall_ms, tokens_in, tokens_out, model_calls, confidence,"
            " gaps, ran_level, script, trace_id, depth)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                c.call_id, c.session, time.time(), c.query, c.intent, c.level, int(c.breadth),
                json.dumps(c.ranked_tools), json.dumps(c.plan), c.replans, 0, c.wall_ms,
                c.tokens_in, c.tokens_out, c.model_calls, c.confidence, json.dumps(c.gaps),
                c.ran_level or c.level, c.script, c.trace_id, c.depth,
            ),
        )  # fmt: skip
        await self.db.commit()

    async def record_op(
        self,
        call_id: str,
        session: str,
        step_id: str,
        r: OpResult,
        hop: int = 0,
        parent: str | None = None,
        commit: bool = True,
    ) -> None:
        """`hop` and `parent` record how a crawl reached this step, and `fields` the scalar
        values of its records, so dreaming can see chains step by step (the pointers table
        keeps only the newest copy of a record per session). A caller logging many ops passes
        `commit=False` and calls `commit()` once."""
        fields = [scalar_fields(rec) for rec in r.records[:FIELD_RECORDS]]
        await self.db.execute(
            "INSERT INTO ops (call_id, step_id, op, params, ok, error, n_results, ms, truncated,"
            " bytes, hop, parent, fields) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                call_id, step_id, r.op, json.dumps(r.params, default=str), int(r.ok), r.error,
                len(r.records), r.ms, int(r.truncated), r.bytes, hop, parent,
                json.dumps(fields, default=str) if fields else None,
            ),
        )  # fmt: skip
        rows = [
            (session, p, call_id, r.op, json.dumps(rec, default=str), time.time())
            for p, rec in zip(r.pointers, r.records, strict=True)
        ]
        await self.db.executemany(
            "INSERT INTO pointers (session, pointer, call_id, op, record, ts) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(session, pointer) DO UPDATE SET call_id = excluded.call_id, "
            "op = excluded.op, record = excluded.record, ts = excluded.ts",
            rows,
        )
        if commit:
            await self.db.commit()

    async def commit(self) -> None:
        await self.db.commit()

    async def record_feedback(
        self, call_id: str, useful: bool, note: str | None, source: str
    ) -> bool:
        """False when no such call was logged."""
        async with self.db.execute("SELECT 1 FROM calls WHERE call_id = ?", (call_id,)) as cur:
            if await cur.fetchone() is None:
                return False
        await self.db.execute(
            "INSERT INTO feedback VALUES (?,?,?,?,?)",
            (call_id, source, int(useful), note, time.time()),
        )
        await self.db.commit()
        return True

    async def lookup(self, pointer: str, session: str | None) -> Expanded | None:
        """Newest record for a pointer: in `session` if given, else any session within 24 h."""
        if session:
            sql = "SELECT pointer, op, call_id, record FROM pointers WHERE pointer=? AND session=?"
            args: tuple[Any, ...] = (pointer, session)
        else:
            sql = "SELECT pointer, op, call_id, record FROM pointers WHERE pointer=? AND ts>=?"
            args = (pointer, time.time() - EXPAND_GRACE_S)
        async with self.db.execute(sql + " ORDER BY ts DESC LIMIT 1", args) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        await self.db.execute(
            "UPDATE pointers SET expanded_at=? WHERE pointer=? AND call_id=?",
            (time.time(), pointer, row[2]),
        )
        await self.db.commit()
        return Expanded(pointer=row[0], op=row[1], call_id=row[2], record=json.loads(row[3]))


@asynccontextmanager
async def open_store(path: Path) -> AsyncIterator[Store]:
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(path) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA busy_timeout=5000")
        await db.executescript(SCHEMA)
        await _migrate(db)
        yield Store(db)


# fmt: off
ADDED_COLUMNS = (
    ("calls", "ran_level", "TEXT"), ("calls", "script", "TEXT"), ("calls", "trace_id", "TEXT"),
    ("calls", "depth", "INTEGER"), ("ops", "hop", "INTEGER"), ("ops", "parent", "TEXT"),
    ("ops", "fields", "TEXT"),
)
# fmt: on


async def _migrate(db: aiosqlite.Connection) -> None:
    """Add columns that logs created by an earlier version lack. Two calls started together
    after an upgrade may both find a column missing; the second one's ALTER then fails as a
    duplicate, which is the state it wanted."""
    for table, column, kind in ADDED_COLUMNS:
        async with db.execute(f"PRAGMA table_info({table})") as cur:
            if column in {row[1] for row in await cur.fetchall()}:
                continue
        try:
            await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc):
                raise
    await db.commit()
