"""Emit the `oracle_tools` library that model-written scripts import: one module per tool, one
async function per read-only op, with the manifest's `when` as the docstring.

The stubs do no work. Each function sends its op and params to the parent process, which runs
it through the executor (validation, timeouts, paging, caps, redaction, pointers), so a script
never holds a secret or starts a process. Records come back as plain dicts; each carries its
pointer under `_pointer`, so `report()` can hand pointers to the summary.

The library lives in ~/.oracle/lib/<hash>/, keyed by a hash of its sources: a manifest change
gives a new directory, and nothing generated is ever written into the repo.
"""

import hashlib
import keyword
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from oracle.registry.loader import Registry
from oracle.registry.schema import Op, Param, Tool

PACKAGE = "oracle_tools"
RESERVED = frozenset({"gather", "report", "asyncio", "_rpc"})  # names __init__ defines
LIB_KEEP_S = 7 * 86400  # a library no call has used for this long is removed
TYPES = {
    "str": "str",
    "int": "int",
    "float": "float",
    "bool": "bool",
    "list[str]": "list[str]",
    "json": "dict | list",
}
INIT = '''"""Generated from the registry; do not edit. Import tool modules from here."""

import asyncio

from oracle_tools import _rpc
{imports}

__all__ = ["gather", "report", {names}]


async def gather(*calls):
    """Run tool calls together; returns their results in order, as a list. A script sees only
    coroutines and lists, never the event loop's futures."""
    return list(await asyncio.gather(*calls))


def report(value):
    """The script's result: what the summary reads. Call it once, or once per part."""
    _rpc.report(value)
'''
RPC = '''"""Replaced by the script runner's child process at start."""


async def call(method, args):
    raise RuntimeError("oracle_tools runs only inside the oracle script runner")


def report(value):
    raise RuntimeError("oracle_tools runs only inside the oracle script runner")


async def op(ref, params):
    given = {k: v for k, v in params.items() if v is not None}
    return await call("op", {"op": ref, "params": given})
'''


@dataclass
class Library:
    path: Path
    """The directory to put on sys.path (it holds `oracle_tools/`)."""
    stubs: dict[str, str]
    """Tool name -> its module source, for the script prompt."""
    problems: list[str]


def ident(name: str) -> str:
    """A Python identifier for a tool, op or param name: `code-search` -> `code_search`,
    `jira.search` -> `jira_search`, `for` -> `for_`."""
    out = name.replace("-", "_").replace(".", "_")
    return out + "_" if keyword.iskeyword(out) else out


def module_for(tool: Tool) -> tuple[str, list[str]]:
    lines = [f'"""{tool.name}: {_doc(tool.description)} (generated; do not edit)"""', ""]
    lines += ["from oracle_tools import _rpc", ""]
    seen: dict[str, str] = {}
    problems: list[str] = []
    for op in tool.ops:
        if op.mutating or tool.mutating:
            continue
        fn = ident(op.name)
        if fn in seen:
            problems.append(f"{tool.name}: ops {seen[fn]} and {op.name} both become {fn}()")
            continue
        if clash := _clash([ident(n) for n in op.params]):
            problems.append(f"{tool.name}.{op.name}: params become the same name {clash}")
            continue
        seen[fn] = op.name
        lines += ["", *_function(tool, op, fn)]
    return "\n".join(lines) + "\n", problems


def _clash(names: list[str]) -> str | None:
    return next((n for n in names if names.count(n) > 1), None)


def _function(tool: Tool, op: Op, fn: str) -> list[str]:
    body = ", ".join(f"{n!r}: {ident(n)}" for n in op.params)
    return [
        _signature(op, fn),
        _docstring(op),
        f"    return await _rpc.op({tool.name + '.' + op.name!r}, {{{body}}})",
        "",
    ]


def _signature(op: Op, fn: str) -> str:
    """Keyword-only, required params first; an optional param defaults to its manifest
    default, else None (left out of the call)."""
    ordered = sorted(op.params.items(), key=lambda item: not item[1].required)
    args = [
        f"{ident(n)}: {TYPES[p.type]}" if p.required else
        f"{ident(n)}: {TYPES[p.type]} | None = {_default(p)}"
        for n, p in ordered
    ]  # fmt: skip
    return (
        f"async def {fn}(*, {', '.join(args)}) -> list[dict]:"
        if args
        else (f"async def {fn}() -> list[dict]:")
    )


def _docstring(op: Op) -> str:
    params = [_param_doc(n, p) for n, p in op.params.items() if p.desc]
    if not params:
        return f'    """{_doc(op.when)}"""'
    return f'    """{_doc(op.when)}\n\n' + "\n".join(params) + '\n    """'


def _default(p: Param) -> str:
    return repr(p.default) if isinstance(p.default, str | int | float | bool) else "None"


def _param_doc(name: str, p: Param) -> str:
    choices = f" (one of {', '.join(p.enum)})" if p.enum else ""
    return f"    {ident(name)}: {_doc(p.desc)}{choices}"


def _doc(text: str) -> str:
    """Safe inside a triple-quoted docstring, whatever the manifest says (a `when` that ends
    in a quote would otherwise close the string early)."""
    return " ".join(text.split()).replace("\\", "\\\\").replace('"', '\\"')


def sources(registry: Registry) -> tuple[dict[str, str], list[str]]:
    """Relative path -> source, for the whole package."""
    files = {f"{PACKAGE}/_rpc.py": RPC}
    problems: list[str] = []
    modules: list[str] = []
    for name in sorted(registry.tools):
        module = ident(name)
        source, found = module_for(registry.tools[name])
        problems += found
        if problem := _unusable(name, module, source, modules):
            problems.append(f"{name}: {problem}; left out of {PACKAGE}")
            continue
        modules.append(module)
        files[f"{PACKAGE}/{module}.py"] = source
    imports = "\n".join(f"from oracle_tools import {m}" for m in modules)
    names = ", ".join(repr(m) for m in modules)
    files[f"{PACKAGE}/__init__.py"] = INIT.format(imports=imports, names=names)
    return files, problems


def _unusable(name: str, module: str, source: str, taken: list[str]) -> str | None:
    """Why a tool's module cannot join the package (one bad manifest must not break the
    package's `__init__`, which imports every module)."""
    if module in RESERVED or module in taken:
        return f"its module name {module} is already taken"
    try:
        compile(source, f"{module}.py", "exec", dont_inherit=True)
    except SyntaxError as exc:
        return f"its stub does not compile ({exc.msg}, line {exc.lineno})"
    return None


def build_library(registry: Registry, home: Path) -> Library:
    files, problems = sources(registry)
    digest = hashlib.sha256()
    for rel in sorted(files):
        digest.update(rel.encode() + b"\0" + files[rel].encode() + b"\0")
    target = home / "lib" / digest.hexdigest()[:12]
    if not (target / PACKAGE / "__init__.py").exists():
        _write_atomically(target, files)
    _prune(target)
    stubs = {
        name: files[path]
        for name in registry.tools
        if (path := f"{PACKAGE}/{ident(name)}.py") in files
    }
    return Library(path=target, stubs=stubs, problems=problems)


def _prune(current: Path) -> None:
    """Mark this library as used and remove the ones no call has used for LIB_KEEP_S: each
    manifest edit leaves a directory behind."""
    now = time.time()
    os.utime(current, (now, now))
    for old in current.parent.iterdir():
        if old != current and old.is_dir() and now - old.stat().st_mtime > LIB_KEEP_S:
            shutil.rmtree(old, ignore_errors=True)


def _write_atomically(target: Path, files: dict[str, str]) -> None:
    """Write beside the target and rename, so a parallel call never sees half a library."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    for rel, text in files.items():
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp / rel).write_text(text)
    try:
        tmp.rename(target)
    except OSError:  # another call won the race; its library is identical
        shutil.rmtree(tmp, ignore_errors=True)
