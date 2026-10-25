"""One budget per call, at every tier: every model call (planner, check, distill, extract, hop
plan, script writer, summary) goes through it, so the per-call timeout, the usage charge and the
reading of a Ctrl-C live here and nowhere else.

A tier's time bounds come from one table. A deep call also has a token budget, with a reserve
kept so a crawl never spends what the final summary needs. Tool calls cost no model tokens and
are not charged; they are counted, paced by per-tool concurrency, and bounded by a separate cap
so a work API is never flooded.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

UNLIMITED = None
SUMMARY_RESERVE = 20_000


@dataclass
class Usage:
    tokens_in: int = 0
    tokens_out: int = 0
    model_calls: int = 0

    def add(self, run_usage: object) -> None:
        """Add a PydanticAI RunUsage."""
        self.tokens_in += getattr(run_usage, "input_tokens", 0) or 0
        self.tokens_out += getattr(run_usage, "output_tokens", 0) or 0
        self.model_calls += getattr(run_usage, "requests", 0) or 0


@dataclass(frozen=True)
class Bounds:
    model_s: float | None = 90.0
    """Any one model call, whatever the provider."""
    follow_up_s: float | None = None
    """No new completeness check (standard) or hop (deep) after this."""
    summarise_by_s: float | None = None
    """Past this, findings are raw summary fields rather than a late distill."""
    summary_s: float | None = None
    """The whole summary, its calls together, in place of the per-call bound."""
    tool_calls: int | None = None
    """Searches a crawl may make."""


BOUNDS = {
    "quick": Bounds(),
    "standard": Bounds(follow_up_s=60.0, summarise_by_s=90.0),  # DESIGN's bound for standard
    "deep": Bounds(follow_up_s=300.0, summary_s=240.0, tool_calls=400),
}


class Interrupted(Exception):
    """A model call failed because the stop was set while it ran: the first Ctrl-C reaches the
    model processes too (Claude Code's children share the terminal's process group)."""


class _Run(Protocol):
    @property
    def usage(self) -> Any: ...


@dataclass
class Budget:
    tokens: int | None = None
    """Model tokens for the whole call; None is `--budget unlimited`, and every tier but deep."""
    reserve: int = 0
    """Kept back for the final summary."""
    bounds: Bounds = Bounds()
    stop: asyncio.Event | None = None
    """Set by the first Ctrl-C; only a crawl has hops to stop between."""
    usage: Usage = field(default_factory=Usage)
    tool_calls: int = 0
    clock: Callable[[], float] = time.monotonic
    """A test passes its own clock, and `started` on that clock."""
    started: float = field(default_factory=time.monotonic)

    @classmethod
    def for_tier(
        cls, level: str, tokens: int | None = None, stop: asyncio.Event | None = None
    ) -> "Budget":
        """`tokens` counts only at deep. An unlimited deep budget drops the wall-clock bound
        too, leaving hop depth as the limit; a small one keeps a smaller share back, since its
        crawl is small and so is its summary."""
        bounds = BOUNDS[level]
        if level != "deep":
            return cls(bounds=bounds, stop=stop)
        if tokens is None:
            unbounded = replace(bounds, follow_up_s=None)
            return cls(tokens=None, reserve=SUMMARY_RESERVE, bounds=unbounded, stop=stop)
        reserve = min(SUMMARY_RESERVE, tokens // 4)
        return cls(tokens=tokens, reserve=reserve, bounds=bounds, stop=stop)

    async def model[R: _Run](self, work: Awaitable[R]) -> R:
        """Run one model call within its bound and charge what it used. A failure while the
        stop was being set raises Interrupted; a call started after the stop is not killed by
        it, so it fails only for its own reasons."""
        stopped = self.stopping()
        try:
            async with asyncio.timeout(self.bounds.model_s):
                result = await work
        except Exception as exc:  # never CancelledError: cancellation passes through untouched
            if not stopped and self.stopping():
                raise Interrupted(f"stopped by Ctrl-C ({type(exc).__name__}: {exc})") from exc
            raise
        self.usage.add(result.usage)
        return result

    @asynccontextmanager
    async def summary(self) -> AsyncIterator["Budget"]:
        """The budget the summary's calls go through: with a summary bound, the whole summary
        has that long and no single call has less; otherwise each call keeps its own bound."""
        if self.bounds.summary_s is None:
            yield self
            return
        view = replace(self, bounds=replace(self.bounds, model_s=None))  # shares usage and stop
        async with asyncio.timeout(self.bounds.summary_s):
            yield view

    def stopping(self) -> bool:
        return self.stop is not None and self.stop.is_set()

    def elapsed(self) -> float:
        return self.clock() - self.started

    def out_of_time(self) -> bool:
        """Past the tier's bound for follow-up work (another check, another hop)."""
        return self.bounds.follow_up_s is not None and self.elapsed() > self.bounds.follow_up_s

    def late_for_summary(self) -> bool:
        return self.bounds.summarise_by_s is not None and (
            self.elapsed() > self.bounds.summarise_by_s
        )

    @property
    def spent(self) -> int:
        return self.usage.tokens_in + self.usage.tokens_out

    @property
    def unlimited(self) -> bool:
        return self.tokens is None

    def available(self) -> float:
        """Tokens left before the reserve."""
        if self.tokens is None:
            return float("inf")
        return self.tokens - self.reserve - self.spent

    def admits(self, estimate: int) -> bool:
        return estimate <= self.available()

    def tool_calls_left(self) -> float:
        cap = self.bounds.tool_calls
        return float("inf") if cap is None else cap - self.tool_calls

    def line(self) -> str:
        cap = "unlimited" if self.tokens is None else f"{self.tokens}"
        return (
            f"cost: {self.spent} tokens of {cap}, {self.usage.model_calls} model calls, "
            f"{self.tool_calls} searches, {self.elapsed():.0f} s"
        )


def parse_budget(value: str | int | None, default: int) -> int | None:
    """`--budget 120000`, `--budget unlimited`, or the default. config.toml may give a bare
    number (`deep_budget = 150000`)."""
    if value is None or value == "":
        return default
    if str(value).strip().lower() in ("unlimited", "none", "inf"):
        return UNLIMITED
    tokens = int(value)
    if tokens <= 0:
        raise ValueError("--budget must be a positive number of tokens or 'unlimited'")
    return tokens
