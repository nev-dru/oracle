"""Shared helpers for the fake acli and workiq CLIs.

The fakes stand in for work-only tools on the personal laptop. Their flags and JSON shapes are
guesses at the real CLIs; on the work laptop the real tools are ingested with `oracle tool add`
and these fakes are not used. Fixture times are offsets from now, so "today" stays today.
Stdlib only and Python 3.9-compatible.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
NOW = datetime.now().astimezone()


def load(name: str) -> list[dict]:
    rows = json.loads((DATA / f"{name}.json").read_text())
    for row in rows:
        for key in [k for k in row if k.endswith("_offset_h")]:
            stamp = offset_stamp(row.pop(key))
            row[key[: -len("_offset_h")]] = stamp.isoformat(timespec="seconds")
        for key in [k for k in row if k.endswith("_today")]:
            hour, minute = (int(x) for x in row.pop(key).split(":"))
            stamp = NOW.replace(hour=hour, minute=minute, second=0, microsecond=0)
            row[key[: -len("_today")]] = stamp.isoformat(timespec="seconds")
    return rows


TODAY_HOURS = 12


def offset_stamp(hours: float) -> datetime:
    """NOW + hours, except that an offset within the last TODAY_HOURS stays today: early in the
    morning all such rows are spread over the time since midnight, in the same order, so a
    "today" fixture does not slide into yesterday before 8am."""
    midnight = NOW.replace(hour=0, minute=0, second=0, microsecond=0)
    early = NOW - midnight < timedelta(hours=TODAY_HOURS)
    if early and -TODAY_HOURS <= hours <= 0:
        return midnight + (NOW - midnight) * ((TODAY_HOURS + hours) / TODAY_HOURS)
    return NOW + timedelta(hours=hours)


def emit(obj: object) -> None:
    json.dump(obj, sys.stdout, indent=None)
    sys.stdout.write("\n")


def fail(msg: str, code: int = 1) -> None:
    sys.stderr.write(msg + "\n")
    raise SystemExit(code)


def terms(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2]


def matches(query: str, *fields: object) -> bool:
    """Every query term prefix-matches some word in the fields, like a forgiving full-text index."""
    words = set(terms(" ".join(str(f) for f in fields if f)))
    return all(any(w.startswith(t[:5]) for w in words) for t in terms(query))


def since(value: str | None) -> datetime | None:
    """`today`, `startOfDay()`, `startOfWeek()`, `-1d`, `24h`, `-2w` or an ISO timestamp.

    Raises ValueError on anything else.
    """
    if not value:
        return None
    v = value.strip().strip("\"'")
    midnight = NOW.replace(hour=0, minute=0, second=0, microsecond=0)
    if v in ("today", "startOfDay()"):
        return midnight
    if v == "startOfWeek()":
        return midnight - timedelta(days=midnight.weekday())
    m = re.fullmatch(r"-?(\d+)([whd])", v)
    if m:
        unit = {"w": "weeks", "d": "days", "h": "hours"}[m.group(2)]
        return NOW - timedelta(**{unit: int(m.group(1))})
    stamp = datetime.fromisoformat(v)
    return stamp if stamp.tzinfo else stamp.astimezone()


def since_arg(value: str | None) -> datetime | None:
    """`since()` for a CLI flag: a bad value is a usage error."""
    try:
        return since(value)
    except ValueError:
        fail(f"cannot parse time {value!r}", code=2)
        raise


def after(stamp: str, start: datetime | None) -> bool:
    return start is None or datetime.fromisoformat(stamp) >= start


CLAUSE = re.compile(
    r"""^\s*(\w+)\s*(?:(!~|~|!=|>=|<=|=|>|<)|\s(not\s+in|in)\b)\s*(.+?)\s*$""", re.IGNORECASE
)
Match = Callable[[dict, str, str, str], "bool | None"]


class Unsupported(Exception):
    pass


def compile_query(query: str, match: Match, supported: str) -> Callable[[dict], bool]:
    """A strict JQL/CQL subset: AND, OR, parentheses, ORDER BY dropped.

    `match(row, field, op, value)` returns None for a clause it cannot evaluate. Any such clause
    makes the fake exit 2, so a plan the fake cannot check is an op error, never a false pass.
    """
    body = _split_top(query, r"order\s+by")[0] if query.strip() else ""
    try:
        keep = _compile(body, match)
    except Unsupported as exc:
        fail(f"unsupported clause {exc} (this fake supports {supported})", code=2)
        raise

    def checked(row: dict) -> bool:
        try:
            return keep(row)
        except Unsupported as exc:
            fail(f"unsupported clause {exc} (this fake supports {supported})", code=2)
            raise

    return checked


def _compile(expr: str, match: Match) -> Callable[[dict], bool]:
    expr = _strip_parens(expr)
    ors = _split_top(expr, "or")
    if len(ors) > 1:
        parts = [_compile(e, match) for e in ors]
        return lambda row: any(p(row) for p in parts)
    ands = _split_top(expr, "and")
    if len(ands) > 1:
        parts = [_compile(e, match) for e in ands]
        return lambda row: all(p(row) for p in parts)
    m = CLAUSE.match(expr)
    if not m:
        raise Unsupported(repr(expr))
    field = m.group(1).lower()
    op = " ".join((m.group(2) or m.group(3)).lower().split())
    value = m.group(4)

    def clause(row: dict) -> bool:
        hit = match(row, field, op, value)
        if hit is None:
            raise Unsupported(repr(expr))
        return hit

    return clause


def _strip_parens(expr: str) -> str:
    while expr.startswith("(") and expr.endswith(")") and _depth_ok(expr[1:-1]):
        expr = expr[1:-1].strip()
    return expr


def _depth_ok(expr: str) -> bool:
    depth = 0
    for c in expr:
        depth += {"(": 1, ")": -1}.get(c, 0)
        if depth < 0:
            return False
    return depth == 0


def _split_top(expr: str, word: str) -> list[str]:
    """Split on a keyword outside parentheses and quotes."""
    parts, depth, quote, start, i = [], 0, "", 0, 0
    pattern = re.compile(rf"\s+{word}\s+", re.I)
    while i < len(expr):
        c = expr[i]
        if quote:
            quote = "" if c == quote else quote
        elif c in "\"'":
            quote = c
        elif c in "()":
            depth += 1 if c == "(" else -1
        elif depth == 0 and (m := pattern.match(expr, i)):
            parts.append(expr[start:i])
            start = i = m.end()
            continue
        i += 1
    parts.append(expr[start:])
    return [p.strip() for p in parts if p.strip()]


def unquote(value: str) -> str:
    return value.strip().strip("\"'")


def listed(value: str) -> list[str]:
    """`("To Do", Done)` -> ["to do", "done"]"""
    return [unquote(v).lower() for v in value.strip().strip("()").split(",") if v.strip()]


def compare(op: str, left: str, right: str) -> bool | None:
    """Equality and IN for strings; anything else is unsupported for this field."""
    left = left.lower()
    if op == "=":
        return left == unquote(right).lower()
    if op == "!=":
        return left != unquote(right).lower()
    if op in ("in", "not in"):
        return (left in listed(right)) == (op == "in")
    return None


def compare_time(op: str, stamp: str, value: str) -> bool | None:
    try:
        start = since(unquote(value))
    except ValueError:
        return None
    if start is None:
        return None
    when = datetime.fromisoformat(stamp)
    ops = {">=": when >= start, ">": when > start, "<=": when <= start, "<": when < start}
    return ops.get(op)


def page(rows: list[dict], start: int, limit: int) -> tuple[list[dict], int | None]:
    chunk = rows[start : start + limit]
    nxt = start + limit if start + limit < len(rows) else None
    return chunk, nxt
