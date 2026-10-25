"""One budget per call: every model call (planner, check, extract, summary, script agent)
charges it, and a reserve is kept so a crawl never spends what the final summary needs.

Tool calls cost no model tokens and are not charged; they are counted, paced by per-tool
concurrency, and bounded by a separate cap so a work API is never flooded.
"""

import time
from dataclasses import dataclass, field

from oracle.plan.planner import Usage

UNLIMITED = None


@dataclass
class Budget:
    tokens: int | None
    """Model tokens for the whole call; None is `--budget unlimited`."""
    reserve: int = 0
    """Kept back for the final summary."""
    max_tool_calls: int | None = 400
    wall_s: float | None = None
    usage: Usage = field(default_factory=Usage)
    tool_calls: int = 0
    started: float = field(default_factory=time.monotonic)

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

    def out_of_time(self) -> bool:
        return self.wall_s is not None and time.monotonic() - self.started > self.wall_s

    def tool_calls_left(self) -> float:
        return (
            float("inf") if self.max_tool_calls is None else self.max_tool_calls - self.tool_calls
        )

    def line(self) -> str:
        cap = "unlimited" if self.tokens is None else f"{self.tokens}"
        secs = time.monotonic() - self.started
        return (
            f"cost: {self.spent} tokens of {cap}, {self.usage.model_calls} model calls, "
            f"{self.tool_calls} searches, {secs:.0f} s"
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
