"""Run one op: validate params, check auth, build argv, run with timeout and per-tool concurrency,
parse, page, cap output and attach pointers.

A failed op returns a structured status instead of raising, so siblings and Coverage can carry on.
"""

import asyncio
import contextlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from oracle.exec.parse import ParseError, parse_output
from oracle.exec.pointers import pointer_for, record_id
from oracle.exec.redact import redact
from oracle.exec.runners.argv import ArgvError, build_argv, is_set
from oracle.exec.runners.mcp import McpError, call_mcp
from oracle.registry import jsonpath
from oracle.registry.loader import Registry
from oracle.registry.params import ParamError, validate_params
from oracle.registry.safety import rendered_problem
from oracle.registry.schema import Op, Tool

Status = Literal["ok", "error", "not_configured", "invalid", "skipped", "unavailable"]
STDERR_TAIL = 400


@dataclass
class OpResult:
    op: str
    params: dict[str, Any]
    status: Status = "ok"
    error: str | None = None
    records: list[dict[str, Any]] = field(default_factory=list)
    pointers: list[str] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    truncated: bool = False
    spill: str | None = None
    pages: int = 0
    bytes: int = 0
    ms: int = 0
    argv: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class OpFailed(Exception):
    def __init__(self, status: Status, message: str) -> None:
        super().__init__(message)
        self.status: Status = status


class Executor:
    def __init__(self, registry: Registry, env: dict[str, str], spill_dir: Path) -> None:
        self.registry = registry
        self.env = env
        self.spill_dir = spill_dir
        self._sems = {
            name: asyncio.Semaphore(t.limits.concurrency) for name, t in registry.tools.items()
        }

    async def run(self, ref: str, raw_params: dict[str, Any]) -> OpResult:
        result = OpResult(op=ref, params=dict(raw_params))
        start = time.monotonic()
        try:
            tool, op = self.registry.resolve(ref)
            result.params = validate_params(op, raw_params)
            self._check_auth(tool)
            await self._run_pages(tool, op, result)
        except (KeyError, ParamError, ArgvError) as exc:
            result.status, result.error = "invalid", str(exc).strip("'\"")
        except OpFailed as exc:
            result.status, result.error = exc.status, str(exc)
        except Exception as exc:  # the contract: an op never raises, siblings carry on
            result.status, result.error = "error", f"{type(exc).__name__}: {exc}"
        result.ms = int((time.monotonic() - start) * 1000)
        return result

    def _check_auth(self, tool: Tool) -> None:
        missing = [v for v in tool.auth.env if not self.env.get(v)]
        if missing:
            raise OpFailed("not_configured", f"missing environment variables {missing}")

    async def _run_pages(self, tool: Tool, op: Op, result: OpResult) -> None:
        cap = tool.limits.max_output_kb * 1024
        cursor: Any = None
        raw_pages: list[bytes] = []
        max_pages = op.paginate.max_pages if op.paginate else 1
        while result.pages < max_pages:
            raw, cursor = await self._page(tool, op, result, cursor)
            raw_pages.append(raw)
            if not is_set(cursor) or sum(len(p) for p in raw_pages) > cap:
                break
        result.bytes = sum(len(p) for p in raw_pages)
        if result.bytes > cap and result.spill is None:
            self._spill(b"\n".join(raw_pages), cap, op, result)

    async def _page(self, tool: Tool, op: Op, result: OpResult, cursor: Any) -> tuple[bytes, Any]:
        """Fetch and parse one page; return its raw bytes and the next cursor, if any."""
        if op.mcp_tool and tool.mcp:
            raw = await self._mcp(tool, op, result)
            result.pages += 1
            self._parse_page(tool, op, raw, result)
            return raw, None  # MCP results are not paged
        params = dict(result.params)
        if op.paginate:
            params[op.paginate.param] = cursor
        argv, stdin = build_argv(op.argv, params)
        if problem := rendered_problem(argv):  # the command exactly as it would run
            raise OpFailed("invalid", problem)
        result.argv = argv
        raw = await self._spawn(tool, argv, stdin)
        result.pages += 1
        doc = self._parse_page(tool, op, raw, result)
        return raw, jsonpath.first(doc, op.paginate.next) if op.paginate else None

    async def _mcp(self, tool: Tool, op: Op, result: OpResult) -> bytes:
        """An MCP tool takes the op's params as its arguments (unset ones left out)."""
        assert tool.mcp is not None and op.mcp_tool is not None
        arguments = {k: v for k, v in result.params.items() if v is not None}
        result.argv = [tool.mcp.command, *tool.mcp.args, f"mcp:{op.mcp_tool}"]
        async with self._sems[tool.name]:
            try:
                text = await call_mcp(
                    tool.mcp, op.mcp_tool, arguments, self.env, tool.limits.timeout_s
                )
            except McpError as exc:
                raise OpFailed(exc.status, str(exc)) from None
            except TimeoutError:
                raise OpFailed("error", f"timed out after {tool.limits.timeout_s:g}s") from None
        return text.encode()

    def _parse_page(self, tool: Tool, op: Op, raw: bytes, result: OpResult) -> Any:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            if result.pages > 1:
                raise OpFailed(
                    "error", f"page {result.pages} was not UTF-8; paging stopped"
                ) from None
            self._binary(raw, result)
            return None
        try:
            doc, records = parse_output(text, op.returns)
        except ParseError as exc:
            raise OpFailed("error", str(exc)) from None
        for raw_record in records:
            # ids and pointers come from the record as returned (a slug can look like a key);
            # everything kept, cached or shown to a model is redacted
            result.records.append(redact(raw_record))
            result.pointers.append(pointer_for(raw_record, op.returns, tool.name))
            rid = record_id(raw_record, op.returns)
            if rid is not None:
                result.ids.append(rid)
        return doc

    async def _spawn(self, tool: Tool, argv: list[str], stdin: str | None) -> bytes:
        timeout = tool.limits.timeout_s
        async with self._sems[tool.name]:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    stdin=asyncio.subprocess.PIPE
                    if stdin is not None
                    else asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=self.env,
                )
            except FileNotFoundError:
                raise OpFailed("not_configured", f"{argv[0]!r} is not on PATH") from None
            except OSError as exc:
                raise OpFailed("error", f"cannot run {argv[0]!r}: {exc.strerror or exc}") from None
            try:
                out, err = await asyncio.wait_for(
                    proc.communicate(stdin.encode() if stdin is not None else None), timeout
                )
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()
                raise OpFailed("error", f"timed out after {timeout:g}s") from None
        code = proc.returncode or 0
        if code != 0:
            tail = err.decode("utf-8", "replace").strip()[-STDERR_TAIL:]
            raise OpFailed("error", f"exit {code}: {tail or '(no stderr)'}")
        return out

    def _spill(self, raw: bytes, cap: int, op: Op, result: OpResult) -> None:
        """Write the full output to a file and keep only the records that fit under the cap."""
        result.spill = self._write_spill(raw)
        kept, size = 0, 0
        for record in result.records:
            size += len(json.dumps(record, default=str))
            if size > cap:
                break
            kept += 1
        result.records, result.pointers = result.records[:kept], result.pointers[:kept]
        result.ids = [i for r in result.records if (i := record_id(r, op.returns)) is not None]
        result.truncated = True

    def _binary(self, raw: bytes, result: OpResult) -> None:
        """Bytes never reach a model: spill them and return a pointer plus a hex head."""
        pointer = self._write_spill(raw)
        result.records.append({"binary": True, "bytes": len(raw), "head_hex": raw[:1024].hex()})
        result.pointers.append(pointer)
        result.spill, result.truncated = pointer, True

    def _write_spill(self, raw: bytes) -> str:
        self.spill_dir.mkdir(parents=True, exist_ok=True)
        path = self.spill_dir / f"{uuid.uuid4().hex}.out"
        path.write_bytes(raw)
        return f"file:{path}"
