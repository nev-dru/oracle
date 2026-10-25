"""Run a workflow op: import its module lazily, enter the guard, bound it in time, and hand it
clients for direct op calls and nested oracle calls.

Workflows are trusted code the user put in the profile's `workflows/` directory (the package
`oracle_workflows`), run in-process; this is not the script sandbox. A module is imported only
when its workflow runs, so the CLI's start-up time does not grow with the number of workflows.
"""

import asyncio
import importlib
import inspect
import sys
import types
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from oracle.workflows.protocol import (
    CURRENT_TRACE,
    MissingWorkflow,
    Trace,
    WorkflowBudget,
    WorkflowContext,
    clamp,
)

if TYPE_CHECKING:
    from oracle.config import Config
    from oracle.exec.executor import Executor, OpResult
    from oracle.registry.loader import Registry
    from oracle.registry.schema import Op, Tool
    from oracle.render import Answer

PACKAGE = "oracle_workflows"
WORKFLOW_WALL_S = 180.0  # without `budget.wall_s`; a tool's 20 s default fits no nested ask
Workflow = Callable[[dict[str, Any], WorkflowContext], Awaitable[Any]]


def load_entry(entry: str, directory: Path) -> Workflow:
    """`oracle_workflows.incident:investigate` -> the function in <directory>/incident.py."""
    module_name, _, fn_name = entry.partition(":")
    package = sys.modules.get(PACKAGE)
    if package is None or list(getattr(package, "__path__", [])) != [str(directory)]:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(directory)]
        sys.modules[PACKAGE] = package
        for name in [m for m in sys.modules if m.startswith(PACKAGE + ".")]:
            del sys.modules[name]  # another registry's workflows
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:  # the module is there; something it imports is not
            raise
        raise MissingWorkflow(f"no workflow module {module_name} in {directory}") from exc
    fn = getattr(module, fn_name, None)
    if not inspect.iscoroutinefunction(fn):
        raise MissingWorkflow(f"{module_name} has no async function {fn_name}")
    return cast(Workflow, fn)


@dataclass
class WorkflowHost:
    cfg: "Config"
    registry: "Registry"
    trace: Trace
    """The trace of the oracle call that owns this executor."""
    models: Any = None
    """The parent call's models, reused by nested oracle calls."""
    executor: "Executor | None" = None

    @property
    def directory(self) -> Path:
        return Path(self.cfg.registry_dir).parent / "workflows"

    def current(self) -> Trace:
        return CURRENT_TRACE.get() or self.trace

    async def run(self, tool: "Tool", op: "Op", params: dict[str, Any]) -> tuple[Any, list[str]]:
        """(the workflow's return value, its notes: `ctx.log` lines and tier clamps)."""
        assert op.entry is not None and self.executor is not None
        child = self.current().enter("workflow", f"{tool.name}.{op.name}", params)
        fn = load_entry(op.entry, self.directory)
        budget = WorkflowBudget(op.budget.model_calls if op.budget else None)
        ctx = WorkflowContext(
            oracle=OracleClient(self, budget),
            tools=ToolsClient(self.executor),
            budget=budget,
            trace=child,
        )
        wall = op.budget.wall_s if op.budget and op.budget.wall_s else WORKFLOW_WALL_S
        token = CURRENT_TRACE.set(child)
        try:
            async with asyncio.timeout(wall):
                value = await fn(dict(params), ctx)
        finally:
            CURRENT_TRACE.reset(token)
        return value, ctx.notes + ctx.oracle.notes


@dataclass
class ToolsClient:
    executor: "Executor"

    async def run(self, ref: str, **params: Any) -> "OpResult":
        """One op with the executor's checks. Never raises: look at `.ok` and `.error`."""
        return await self.executor.run(ref, params)


@dataclass
class OracleClient:
    host: WorkflowHost
    budget: WorkflowBudget
    notes: list[str] = field(default_factory=list)

    async def ask(self, question: str, level: str | None = None, **request: Any) -> "Answer":
        """A nested oracle call: one tier lower than the parent unless asked (never higher).
        Raises GuardError on a cycle, too deep a chain, or the workflow's budget."""
        from oracle.ask import AskRequest, ask  # the full pipeline, only when a workflow asks

        parent = self.host.current()
        tier, note = clamp(parent.level, level)
        child = parent.enter("oracle", "ask", question, level=tier)
        reserved = self.budget.reserve(tier)
        if note:
            self.notes.append(note)
        req = AskRequest(question, level=tier, **request)
        actual = reserved
        try:
            answer = await ask(
                req, self.host.cfg, self.host.registry, self.host.models, trace=child
            )
            actual = answer.model_calls
        finally:
            self.budget.settle(reserved, actual)
        return answer
