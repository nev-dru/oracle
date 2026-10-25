"""run_script(): check a model-written script, run it in a child process, answer its calls.

The child (deep/child.py) is pure compute: `python -I -S -B`, an empty environment, a fresh temp
directory, CPU and file-size limits, restricted builtins and imports. Every op the script calls
comes back here over a pipe and runs through `handle`, which in the oracle is the plan runner,
so the executor's checks, reuse, pointers and logging apply to script calls exactly as to plan
steps. Swapping in a container changes this module only.
"""

import asyncio
import contextlib
import json
import sys
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oracle.deep.sandbox import ALLOWED_IMPORTS, SAFE_BUILTINS, script_problems

CHILD = Path(__file__).with_name("child.py")
LINE_LIMIT = 16 * 1024 * 1024  # one protocol line: a reply or a report


class CallRefused(Exception):
    """The call cannot run as asked (unknown op, bad params); the script sees a ValueError."""


Handler = Callable[[str, dict[str, Any]], Awaitable[Any]]


@dataclass
class ScriptLimits:
    wall_s: float = 180.0
    cpu_s: int = 60
    memory_mb: int = 1024
    max_calls: int = 200
    printed_kb: int = 32
    report_kb: int = 512


@dataclass
class ScriptResult:
    reports: list[Any] = field(default_factory=list)
    printed: str = ""
    error: str | None = None
    calls: int = 0
    refused: list[str] = field(default_factory=list)
    """Problems the static check found; the script did not run."""
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and not self.refused


async def run_script(
    code: str, handle: Handler, libdir: Path, limits: ScriptLimits | None = None
) -> ScriptResult:
    limits = limits or ScriptLimits()
    if problems := script_problems(code):
        return ScriptResult(refused=problems, error="refused: " + "; ".join(problems))
    with tempfile.TemporaryDirectory(prefix="oracle-script-") as cwd:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-I", "-S", "-B", str(CHILD), str(libdir),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={},
            cwd=cwd,
            limit=LINE_LIMIT,
        )  # fmt: skip
        session = _Session(proc, handle, limits)
        return await session.run(code)


@dataclass
class _Session:
    proc: asyncio.subprocess.Process
    handle: Handler
    limits: ScriptLimits
    result: ScriptResult = field(default_factory=ScriptResult)
    tasks: set[asyncio.Task[None]] = field(default_factory=set)
    report_bytes: int = 0

    async def run(self, code: str) -> ScriptResult:
        hello = {
            "code": code,
            "allowed": sorted(ALLOWED_IMPORTS),
            "builtins": sorted(SAFE_BUILTINS),
            "cpu_s": self.limits.cpu_s,
            "memory_mb": self.limits.memory_mb,
        }
        self._send(hello)
        printed = asyncio.ensure_future(self._printed())
        try:
            async with asyncio.timeout(self.limits.wall_s):
                await self._serve()
        except TimeoutError:
            self.result.error = f"the script ran past {self.limits.wall_s:.0f} s and was stopped"
        finally:
            await self._stop()
            self.result.printed = await printed
        return self.result

    async def _serve(self) -> None:
        while True:
            try:
                message = await self._next()
            except ValueError as exc:  # a line past LINE_LIMIT, or not the protocol
                self.result.error = f"the script sent an unreadable message ({exc})"[:300]
                return
            if message is None:
                break
            kind = message.get("t")
            if kind == "call":
                task = asyncio.ensure_future(self._answer(message))
                self.tasks.add(task)
                task.add_done_callback(self.tasks.discard)
            elif kind == "report":
                self._report(message.get("value"))
            elif kind == "error":
                self.result.error = message.get("error") or "the script failed"
                return
            elif kind == "done":
                return
        code = await self.proc.wait()
        self.result.error = f"the script process exited ({_exit_reason(code)})"

    async def _next(self) -> dict[str, Any] | None:
        """The child's next message; None at the end of its output."""
        assert self.proc.stdout is not None
        line = await self.proc.stdout.readline()
        return json.loads(line) if line else None

    async def _answer(self, message: dict[str, Any]) -> None:
        self.result.calls += 1
        reply: dict[str, Any] = {"t": "reply", "id": message.get("id")}
        if self.result.calls > self.limits.max_calls:
            reply |= {"ok": False, "error": f"more than {self.limits.max_calls} tool calls"}
        else:
            try:
                result = await self.handle(message.get("method", ""), message.get("args") or {})
                reply |= {"ok": True, "result": result}
            except CallRefused as exc:
                reply |= {"ok": False, "error": str(exc)}
            except Exception as exc:  # every call gets a reply, or the script waits forever
                reply |= {"ok": False, "error": f"the oracle failed this call: {exc!r}"[:500]}
        self._send(reply)

    def _report(self, value: Any) -> None:
        size = len(json.dumps(value, default=str))
        if self.report_bytes + size > self.limits.report_kb * 1024:
            self.result.truncated = True
            return
        self.report_bytes += size
        self.result.reports.append(value)

    def _send(self, message: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        if not self.proc.stdin.is_closing():
            self.proc.stdin.write(json.dumps(message, default=str).encode() + b"\n")

    async def _printed(self) -> str:
        """The script's print output, capped; read to the end so the child never blocks."""
        assert self.proc.stderr is not None
        cap, kept = self.limits.printed_kb * 1024, bytearray()
        while chunk := await self.proc.stderr.read(65536):
            kept += chunk[: max(0, cap - len(kept))]
        return kept.decode(errors="replace")

    async def _stop(self) -> None:
        for task in list(self.tasks):
            task.cancel()
        if self.proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self.proc.kill()
        await self._drain()  # a paused pipe (an oversized line) would keep wait() from returning
        await self.proc.wait()

    async def _drain(self) -> None:
        assert self.proc.stdout is not None
        with contextlib.suppress(TimeoutError, ConnectionError):
            async with asyncio.timeout(5):
                while await self.proc.stdout.read(65536):
                    pass


def _exit_reason(code: int) -> str:
    if code == -24:  # SIGXCPU
        return "CPU limit"
    if code < 0:
        return f"signal {-code}"
    return f"code {code}"
