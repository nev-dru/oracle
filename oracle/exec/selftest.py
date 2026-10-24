"""`oracle tool test`: run every manifest example against the real tool and check its expectations.

An op with no example fails: the gate means every active op has run at least once.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path

import yaml

from oracle.exec.executor import Executor, OpResult
from oracle.registry.jsonpath import get_field
from oracle.registry.loader import Registry
from oracle.registry.schema import Example


@dataclass
class ExampleResult:
    op: str
    index: int
    ok: bool
    detail: str
    ms: int = 0


async def run_examples(
    registry: Registry, executor: Executor, only: set[str] | None = None
) -> list[ExampleResult]:
    jobs = []
    for tool in registry.tools.values():
        if only and tool.name not in only:
            continue
        for op in tool.ops:
            ref = f"{tool.name}.{op.name}"
            if not op.examples:
                jobs.append(_missing(ref))
            for i, example in enumerate(op.examples):
                jobs.append(_run_one(executor, ref, i, example))
    return list(await asyncio.gather(*jobs))


async def _missing(ref: str) -> ExampleResult:
    return ExampleResult(ref, 0, False, "no examples; every op needs at least one")


async def _run_one(executor: Executor, ref: str, index: int, example: Example) -> ExampleResult:
    result = await executor.run(ref, example.params)
    ok, detail = check(result, example)
    return ExampleResult(ref, index, ok, detail, result.ms)


def check(result: OpResult, example: Example) -> tuple[bool, str]:
    if not result.ok:
        argv = f" [argv: {' '.join(result.argv)}]" if result.argv else ""
        return False, f"{result.status}: {result.error}{argv}"
    n = len(result.records)
    if n < example.expect.min_results:
        return False, f"{n} results, expected at least {example.expect.min_results}"
    for i, record in enumerate(result.records):
        missing = [f for f in example.expect.has_fields if get_field(record, f) in (None, "")]
        if missing:
            return False, f"result {i} lacks fields {missing}"
    pages = f", {result.pages} pages" if result.pages > 1 else ""
    return True, f"{n} results{pages}"


def golden_ops_missing(registry: Registry, golden: Path) -> list[str]:
    """Ops golden.yaml names that this registry lacks: expected while tools are being added
    one by one, a renamed op otherwise."""
    if not golden.is_file():
        return []
    refs = set(registry.op_refs())
    named = {
        ref
        for case in (yaml.safe_load(golden.read_text()) or {}).get("cases", [])
        for key in ("tools_all", "tools_any")
        for ref in case.get("expect", {}).get(key, [])
    }
    return sorted(named - refs)
