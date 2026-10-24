"""log.sqlite: one record per invocation plus the per-session pointer cache `oracle expand` reads.

WAL mode, because the main agent may run several oracle calls in parallel.
"""

import json
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
  confidence TEXT, gaps TEXT, ran_level TEXT
);
CREATE TABLE IF NOT EXISTS ops (
  call_id TEXT, step_id TEXT, op TEXT, params TEXT, ok INTEGER, error TEXT,
  n_results INTEGER, ms INTEGER, truncated INTEGER, bytes INTEGER
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
    """The level that actually ran (M1 runs every call as quick)."""


@dataclass
class Expanded:
    pointer: str
    op: str
    call_id: str
    record: dict[str, Any]


class Store:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def record_call(self, c: CallRecord) -> None:
        await self.db.execute(
            "INSERT INTO calls (call_id, session, ts, query, intent, level, breadth, ranked_tools,"
            " plan, replans, escalated, wall_ms, tokens_in, tokens_out, model_calls, confidence,"
            " gaps, ran_level) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                c.call_id, c.session, time.time(), c.query, c.intent, c.level, int(c.breadth),
                json.dumps(c.ranked_tools), json.dumps(c.plan), 0, 0, c.wall_ms,
                c.tokens_in, c.tokens_out, c.model_calls, c.confidence, json.dumps(c.gaps),
                c.ran_level or c.level,
            ),
        )  # fmt: skip
        await self.db.commit()

    async def record_op(self, call_id: str, session: str, step_id: str, r: OpResult) -> None:
        await self.db.execute(
            "INSERT INTO ops VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                call_id, step_id, r.op, json.dumps(r.params, default=str), int(r.ok), r.error,
                len(r.records), r.ms, int(r.truncated), r.bytes,
            ),
        )  # fmt: skip
        rows = [
            (session, p, call_id, r.op, json.dumps(rec, default=str), time.time())
            for p, rec in zip(r.pointers, r.records, strict=True)
        ]
        await self.db.executemany(
            "INSERT OR REPLACE INTO pointers (session, pointer, call_id, op, record, ts) "
            "VALUES (?,?,?,?,?,?)",
            rows,
        )
        await self.db.commit()

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


async def _migrate(db: aiosqlite.Connection) -> None:
    """Add columns that logs created by an earlier version lack."""
    async with db.execute("PRAGMA table_info(calls)") as cur:
        columns = {row[1] for row in await cur.fetchall()}
    if "ran_level" not in columns:
        await db.execute("ALTER TABLE calls ADD COLUMN ran_level TEXT")
        await db.commit()
