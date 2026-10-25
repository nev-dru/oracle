"""The script runner's child process: standard library only, never imports `oracle`.

Launched as `python -I -S -B child.py <libdir>` with an empty environment in a temp directory.
The parent sends one hello line on stdin (the script, the import allowlist, the builtins, the
limits), then replies to calls. Fd 1 carries the protocol, one JSON object per line; the
script's `print` goes to stderr, which the parent captures. Every tool call goes to the parent,
which runs it through the executor, so this process holds no secrets and starts nothing.

Messages to the parent: {"t": "call", "id", "method", "args"}, {"t": "report", "value"},
{"t": "done"}, {"t": "error", "error"}. From the parent: {"t": "reply", "id", "ok", "result"
or "error"}.
"""

import ast
import asyncio
import builtins
import importlib
import inspect
import json
import resource
import sys
import threading
import traceback
import types
from typing import Any

PROTOCOL = sys.stdout.buffer


def send(message: dict[str, Any]) -> None:
    PROTOCOL.write(json.dumps(message, default=str).encode() + b"\n")
    PROTOCOL.flush()


class Channel:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.pending: dict[int, asyncio.Future[Any]] = {}
        self.next_id = 0
        self.closed: str | None = None

    async def call(self, method: str, args: dict[str, Any]) -> Any:
        if self.closed:
            raise RuntimeError(self.closed)
        self.next_id += 1
        future = self.loop.create_future()
        self.pending[self.next_id] = future
        send({"t": "call", "id": self.next_id, "method": method, "args": args})
        return await future

    def read_replies(self) -> None:  # a thread: stdin blocks
        try:
            for line in sys.stdin:
                self.loop.call_soon_threadsafe(self._resolve, json.loads(line))
            reason = "the oracle closed the channel"
        except (ValueError, OSError) as exc:
            reason = f"unreadable reply from the oracle: {exc}"
        self.loop.call_soon_threadsafe(self._fail_all, reason)  # no call waits for ever

    def _fail_all(self, reason: str) -> None:
        self.closed = reason
        for future in self.pending.values():
            if not future.done():
                future.set_exception(RuntimeError(reason))
        self.pending.clear()

    def _resolve(self, reply: dict[str, Any]) -> None:
        future = self.pending.pop(reply["id"], None)
        if future is None or future.done():
            return
        if reply.get("ok"):
            future.set_result(reply.get("result"))
        else:
            future.set_exception(ValueError(reply.get("error", "call failed")))


def limit(cpu_s: int, memory_mb: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))  # no file writes; Python ignores SIGXFSZ
    try:  # not enforced on macOS; the parent's wall clock is the real bound there
        size = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (size, size))
    except (ValueError, OSError):
        pass


def facade(module: types.ModuleType, nested: bool) -> types.SimpleNamespace:
    """The module's public names without its submodules (`json.codecs` would reach files).
    The stubs package keeps its tool modules, as facades too."""
    names: dict[str, Any] = {}
    for name, value in vars(module).items():
        if name.startswith("_"):
            continue
        if isinstance(value, types.ModuleType):
            if nested and value.__name__.startswith(module.__name__ + "."):
                names[name] = facade(value, nested=False)
            continue
        names[name] = value
    return types.SimpleNamespace(**names)


def guarded_import(allowed: set[str]) -> Any:
    def _import(
        name: str, _globals: Any = None, _locals: Any = None, fromlist: Any = (), level: int = 0
    ) -> Any:  # the import statement calls __import__ positionally
        root = name.split(".", 1)[0]
        if level or root not in allowed or name != root:
            raise ImportError(f"import {name} is not allowed")
        if any(str(f).startswith("_") or f == "*" for f in fromlist or ()):
            raise ImportError(f"importing private names from {name} is not allowed")
        return facade(importlib.import_module(name), nested=name == "oracle_tools")

    return _import


async def run(hello: dict[str, Any]) -> None:
    channel = Channel(asyncio.get_running_loop())
    threading.Thread(target=channel.read_replies, daemon=True).start()
    rpc = importlib.import_module("oracle_tools._rpc")
    rpc.call = channel.call  # type: ignore[attr-defined]
    rpc.report = lambda value: send({"t": "report", "value": value})  # type: ignore[attr-defined]
    safe = {n: getattr(builtins, n) for n in hello["builtins"] if hasattr(builtins, n)}
    safe["__import__"] = guarded_import(set(hello["allowed"]))
    scope: dict[str, Any] = {"__builtins__": safe, "__name__": "script"}
    flags = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
    code = compile(hello["code"], "<script>", "exec", flags=flags, dont_inherit=True)
    result = eval(code, scope)  # the point of this process; see the module doc
    if code.co_flags & inspect.CO_COROUTINE:
        await result


def main() -> None:
    sys.path.insert(0, sys.argv[1])
    hello = json.loads(sys.stdin.readline())
    limit(hello["cpu_s"], hello["memory_mb"])
    sys.stdout = sys.stderr  # print() is the script's notes, not the protocol
    try:
        asyncio.run(run(hello))
    except BaseException:  # every failure, SystemExit included, goes to the parent
        send({"t": "error", "error": _trimmed(traceback.format_exc())})
        return
    send({"t": "done"})


def _trimmed(text: str) -> str:
    """The traceback from the script's own frames on, which is what a model can fix."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if '"<script>"' in line), None)
    if start is None:
        return "\n".join(lines[-6:])
    return "\n".join(["Traceback (most recent call last):", *lines[start:]])


if __name__ == "__main__":
    main()
