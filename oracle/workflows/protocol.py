"""The workflow protocol: what a workflow function receives, and the recursion guard.

A workflow is `async def fn(params: dict, ctx: WorkflowContext) -> list[dict] | dict`, in a
module under the profile's `workflows/` directory, registered by a manifest with
`kind: workflow` and `entry: oracle_workflows.<module>:<fn>`. Its return value goes through the
same parsing, redaction, caps and pointers as any tool's output.

The guard is a `Trace`: the chain of oracle and workflow calls that led here. In-process it is
the `CURRENT_TRACE` context variable (asyncio tasks inherit it); a subprocess gets it in
`ORACLE_TRACE`, so a tool that itself runs `oracle` is still inside the guard. Depth is capped,
a repeated (kind, name, input) is a cycle, and a nested oracle call never runs at a higher tier
than its parent (default: one lower).
"""

import hashlib
import json
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # the context's clients; imported lazily at run time
    from oracle.workflows.runner import OracleClient, ToolsClient

MAX_DEPTH = 3
TIERS = ("quick", "standard", "deep")


class GuardError(Exception):
    """A nested call refused: too deep, a cycle, or the workflow's budget."""


class MissingWorkflow(Exception):
    """The manifest's entry names a module or function that is not there."""


@dataclass(frozen=True)
class Trace:
    trace_id: str
    depth: int = 0
    chain: tuple[tuple[str, str, str], ...] = ()
    level: str = "standard"
    """The tier of the innermost oracle call, which bounds nested calls."""
    max_depth: int = MAX_DEPTH
    ceiling: str | None = None
    """The tier of the link this one entered from: the highest a nested call may run at
    (None at the root)."""

    @classmethod
    def root(cls, level: str) -> "Trace":
        return cls(trace_id=uuid.uuid4().hex[:12], level=level)

    def enter(self, kind: str, name: str, given: Any, level: str | None = None) -> "Trace":
        link = (kind, name, _digest(given))
        if link in self.chain:
            raise GuardError(f"would repeat {name} for the same input ({self.describe()})")
        if self.depth + 1 > self.max_depth:
            raise GuardError(f"nesting deeper than {self.max_depth} ({self.describe()} -> {name})")
        return replace(
            self,
            depth=self.depth + 1,
            chain=(*self.chain, link),
            level=level or self.level,
            ceiling=self.level,
        )

    def describe(self) -> str:
        return " -> ".join(name for _kind, name, _h in self.chain) or "top"

    def to_env(self) -> str:
        return json.dumps(
            {"trace_id": self.trace_id, "depth": self.depth, "chain": self.chain,
             "level": self.level, "max_depth": self.max_depth, "ceiling": self.ceiling}
        )  # fmt: skip

    @classmethod
    def from_env(cls, value: str | None) -> "Trace | None":
        if not value:
            return None
        try:
            data = json.loads(value)
            chain = tuple(tuple(link) for link in data.get("chain", []))
            return cls(
                trace_id=str(data["trace_id"]),
                depth=int(data.get("depth", 0)),
                chain=chain,  # type: ignore[arg-type]
                level=str(data.get("level", "standard")),
                max_depth=int(data.get("max_depth", MAX_DEPTH)),
                ceiling=str(data["ceiling"]) if data.get("ceiling") else None,
            )
        except (ValueError, KeyError, TypeError):
            return None  # a garbled variable is ignored rather than trusted


CURRENT_TRACE: ContextVar[Trace | None] = ContextVar("oracle_trace", default=None)


def clamp(parent: str, asked: str | None) -> tuple[str, str | None]:
    """(the tier a nested call runs at, a note when it differs from what was asked)."""
    top = TIERS.index(parent) if parent in TIERS else 1
    if asked is None:
        return TIERS[max(0, top - 1)], None
    if asked in TIERS and TIERS.index(asked) <= top:
        return asked, None
    return TIERS[
        top
    ], f"note: a nested call cannot run above its parent; {asked} ran as {TIERS[top]}"


def _digest(given: Any) -> str:
    text = given if isinstance(given, str) else json.dumps(given, sort_keys=True, default=str)
    normal = " ".join(str(text).lower().split())
    return hashlib.sha1(normal.encode()).hexdigest()[:12]


ESTIMATE = {"quick": 1, "standard": 4, "deep": 12}
"""Model calls reserved for a nested call at each tier, before it runs."""


@dataclass
class WorkflowBudget:
    """Model calls a workflow's nested oracle calls may make (None: no cap beyond the guard).
    Each call reserves its tier's estimate before it starts, so calls started together under
    `gather` cannot all slip past the cap; the actual count replaces the estimate after."""

    model_calls: int | None = None
    used: int = 0
    reserved: int = 0

    def reserve(self, tier: str) -> int:
        need = ESTIMATE.get(tier, ESTIMATE["standard"])
        if self.model_calls is not None and self.used + self.reserved + need > self.model_calls:
            raise GuardError(
                f"workflow budget of {self.model_calls} model calls: {self.used} used, "
                f"{self.reserved} reserved, a {tier} call needs about {need}"
            )
        self.reserved += need
        return need

    def settle(self, reserved: int, actual: int) -> None:
        self.reserved -= reserved
        self.used += actual


@dataclass
class WorkflowContext:
    oracle: "OracleClient"
    """`await ctx.oracle.ask(question, level=None)`: a nested oracle call under the guard."""
    tools: "ToolsClient"
    """`await ctx.tools.run("kubectl.events", namespace=...)`: one op, through the executor."""
    budget: WorkflowBudget
    trace: Trace
    notes: list[str] = field(default_factory=list)
    """Lines for the workflow's log; anything worth a finding belongs in the returned records."""

    def log(self, line: str) -> None:
        self.notes.append(line)
