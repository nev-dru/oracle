"""Read <registry>/*.yaml into validated Tool models.

A broken manifest is reported and skipped, never fatal: one bad file must not take down every
other tool. Mutating tools are refused and mutating ops dropped, so read-only holds at every tier.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import ValidationError

from oracle.registry.schema import Op, Tool


@dataclass
class Registry:
    tools: dict[str, Tool] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    inactive: list[str] = field(default_factory=list)

    def resolve(self, ref: str) -> tuple[Tool, Op]:
        """`code-graph.find_symbol` -> (tool, op). Op names may contain dots (`jira.search`)."""
        tool_name, _, op_name = ref.partition(".")
        tool = self.tools.get(tool_name)
        if tool is None:
            raise KeyError(f"unknown tool {tool_name!r}")
        return tool, tool.op(op_name)

    def op_refs(self) -> list[str]:
        return [f"{t.name}.{op.name}" for t in self.tools.values() for op in t.ops]


def load_registry(directory: Path, active: set[str] | None = None) -> Registry:
    reg = Registry()
    for path in sorted(directory.glob("*.yaml")):
        tool = _load_one(path, reg.problems)
        if tool is None:
            continue
        if active is not None and tool.name not in active:
            reg.inactive.append(tool.name)
            continue
        reg.tools[tool.name] = tool
    _check_references(reg)
    return reg


def _load_one(path: Path, problems: list[str]) -> Tool | None:
    try:
        tool = Tool.model_validate(yaml.safe_load(path.read_text()))
    except (yaml.YAMLError, ValidationError) as exc:
        problems.append(f"{path.name}: {exc}")
        return None
    if tool.name != path.stem:
        problems.append(f"{path.name}: name {tool.name!r} does not match the file name")
        return None
    if tool.mutating:
        problems.append(f"{path.name}: refused, tool is marked mutating")
        return None
    dropped = [op.name for op in tool.ops if op.mutating]
    if dropped:
        problems.append(f"{path.name}: dropped mutating ops {dropped}")
        kept = [op for op in tool.ops if not op.mutating]
        if not kept:
            return None
        tool = tool.model_copy(update={"ops": kept})
    return tool


def _check_references(reg: Registry) -> None:
    refs = set(reg.op_refs())
    for tool in reg.tools.values():
        for op in tool.ops:
            missing = [c for c in op.chains_to if c not in refs]
            missing += [
                p.enum_from for p in op.params.values() if p.enum_from and p.enum_from not in refs
            ]
            if missing:
                reg.problems.append(f"{tool.name}.{op.name}: references unknown ops {missing}")
