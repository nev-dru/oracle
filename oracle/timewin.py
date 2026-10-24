"""Relative time in a question ("today", "this week", "past 3 days") resolved deterministically in
the user's timezone, so the planner and recipes get absolute bounds and the trace can show them.
"""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

PAST = re.compile(r"\b(?:past|last)\s+(\d+)\s+(hour|day|week)s?\b")


@dataclass(frozen=True)
class TimeWindow:
    label: str
    start: datetime
    end: datetime

    def describe(self) -> str:
        zone = self.start.tzname() or "local"
        start, end = (t.isoformat(timespec="seconds") for t in (self.start, self.end))
        return f"{self.label} = {start} .. {end} ({zone})"

    def builtins(self) -> dict[str, str]:
        """Values recipes may use as `{window_start}` and `{window_end}`."""
        start, end = (t.isoformat(timespec="seconds") for t in (self.start, self.end))
        return {"window_start": start, "window_end": end}


def resolve_window(question: str, now: datetime) -> TimeWindow | None:
    q = question.lower()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if m := PAST.search(q):
        n, unit = int(m[1]), m[2]
        return TimeWindow(m[0], now - timedelta(**{f"{unit}s": n}), now)
    if re.search(r"\bsince yesterday\b", q):
        return TimeWindow("since yesterday", midnight - timedelta(days=1), now)
    if "yesterday" in q:
        return TimeWindow("yesterday", midnight - timedelta(days=1), midnight)
    if re.search(r"\bthis week\b", q):
        return TimeWindow("this week", midnight - timedelta(days=midnight.weekday()), now)
    if re.search(r"\blast week\b", q):
        monday = midnight - timedelta(days=midnight.weekday())
        return TimeWindow("last week", monday - timedelta(days=7), monday)
    if re.search(r"\b(today|this morning|tonight|so far)\b", q):
        return TimeWindow("today", midnight, now)
    return None


def default_builtins(now: datetime) -> dict[str, str]:
    """Builtins when the question names no time: today so far."""
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return TimeWindow("today", midnight, now).builtins()
