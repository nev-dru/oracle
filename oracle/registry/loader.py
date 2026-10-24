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
    """Broken manifests or a missing registry: these fail `oracle tool test`."""
    notes: list[str] = field(default_factory=list)
    """Expected states worth showing: refused mutating ops, references to tools not added yet."""
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
    if not directory.is_dir():
        reg.problems.append(f"registry directory {directory} does not exist")
        return reg
    loaded = [t for path in sorted(directory.glob("*.yaml")) if (t := _load_one(path, reg))]
    for tool in loaded:
        if active is not None and tool.name not in active:
            reg.inactive.append(tool.name)
        else:
            reg.tools[tool.name] = tool
    _check_references(reg, loaded)
    return reg


def _load_one(path: Path, reg: Registry) -> Tool | None:
    try:
        tool = Tool.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ValidationError) as exc:
        reg.problems.append(f"{path.name}: {exc}")
        return None
    if tool.name != path.stem:
        reg.problems.append(f"{path.name}: name {tool.name!r} does not match the file name")
        return None
    if tool.mutating:
        reg.notes.append(f"{path.name}: refused, the tool is marked mutating (read-only oracle)")
        return None
    dropped = [op.name for op in tool.ops if op.mutating]
    if dropped:
        reg.notes.append(f"{path.name}: dropped mutating ops {dropped}")
        kept = [op for op in tool.ops if not op.mutating]
        if not kept:
            return None
        tool = tool.model_copy(update={"ops": kept})
    return tool


def _check_references(reg: Registry, loaded: list[Tool]) -> None:
    """chains_to and enum_from may name a tool that is not added yet (a note), or an inactive
    one (fine). A missing op on a tool that is loaded is a typo (a problem)."""
    known = {f"{t.name}.{op.name}" for t in loaded for op in t.ops}
    names = {t.name for t in loaded}
    for tool in reg.tools.values():
        for op in tool.ops:
            refs = op.chains_to + [p.enum_from for p in op.params.values() if p.enum_from]
            missing = [r for r in refs if r not in known]
            _report_missing(reg, f"{tool.name}.{op.name}", missing, names)


def _report_missing(reg: Registry, where: str, missing: list[str], names: set[str]) -> None:
    typos = [r for r in missing if r.partition(".")[0] in names]
    later = [r for r in missing if r not in typos]
    if typos:
        reg.problems.append(f"{where}: references ops that do not exist {typos}")
    if later:
        reg.notes.append(f"{where}: references tools not registered yet {later}")
