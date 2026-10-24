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
from datetime import datetime, timedelta
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
NOW = datetime.now().astimezone()


def load(name: str) -> list[dict]:
    rows = json.loads((DATA / f"{name}.json").read_text())
    for row in rows:
        for key in [k for k in row if k.endswith("_offset_h")]:
            stamp = NOW + timedelta(hours=row.pop(key))
            row[key[: -len("_offset_h")]] = stamp.isoformat(timespec="seconds")
        for key in [k for k in row if k.endswith("_today")]:
            hour, minute = (int(x) for x in row.pop(key).split(":"))
            stamp = NOW.replace(hour=hour, minute=minute, second=0, microsecond=0)
            row[key[: -len("_today")]] = stamp.isoformat(timespec="seconds")
    return rows


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
    """`today`, `startOfDay()`, `-1d`, `24h`, `-3h` or an ISO timestamp."""
    if not value:
        return None
    v = value.strip().strip("\"'")
    if v in ("today", "startOfDay()"):
        return NOW.replace(hour=0, minute=0, second=0, microsecond=0)
    m = re.fullmatch(r"-?(\d+)([dh])", v)
    if m:
        n = int(m.group(1))
        return NOW - (timedelta(days=n) if m.group(2) == "d" else timedelta(hours=n))
    try:
        stamp = datetime.fromisoformat(v)
    except ValueError:
        fail(f"cannot parse time {value!r}")
        raise
    return stamp if stamp.tzinfo else stamp.astimezone()


def after(stamp: str, start: datetime | None) -> bool:
    return start is None or datetime.fromisoformat(stamp) >= start


CLAUSE = re.compile(r"""^\s*(\w+)\s*(~|=|>=|<=|!=)\s*("[^"]*"|'[^']*'|\S+)\s*$""")


def clauses(query: str) -> list[tuple[str, str, str]]:
    """Split a JQL/CQL string on AND; drop ORDER BY. Unparseable clauses warn and are ignored."""
    body = re.split(r"\border\s+by\b", query, flags=re.I)[0]
    out = []
    for part in re.split(r"\band\b", body, flags=re.I):
        if not part.strip():
            continue
        m = CLAUSE.match(part)
        if not m:
            sys.stderr.write(f"warning: unsupported clause ignored: {part.strip()}\n")
            continue
        out.append((m.group(1).lower(), m.group(2), m.group(3).strip("\"'")))
    return out


def page(rows: list[dict], start: int, limit: int) -> tuple[list[dict], int | None]:
    chunk = rows[start : start + limit]
    nxt = start + limit if start + limit < len(rows) else None
    return chunk, nxt
