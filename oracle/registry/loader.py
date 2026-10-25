"""Read <registry>/*.yaml into validated Tool models.

A broken manifest is reported and skipped, never fatal: one bad file must not take down every
other tool. Mutating tools are refused and mutating ops dropped, so read-only holds at every tier.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import ValidationError

from oracle.registry.params import ParamError, validate_shape
from oracle.registry.safety import op_problem
from oracle.registry.schema import Op, Recipe, Tool


@dataclass
class Registry:
    tools: dict[str, Tool] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    """Broken manifests or a missing registry: these fail `oracle tool test`."""
    notes: list[str] = field(default_factory=list)
    """Expected states worth showing: refused mutating ops, references to tools not added yet."""
    inactive: list[str] = field(default_factory=list)
    recipes: dict[str, Recipe] = field(default_factory=dict)
    known_tools: set[str] = field(default_factory=set)
    """Every tool with a valid manifest, active or not."""

    def resolve(self, ref: str) -> tuple[Tool, Op]:
        """`code-graph.find_symbol` -> (tool, op). Op names may contain dots (`jira.search`)."""
        tool_name, _, op_name = ref.partition(".")
        tool = self.tools.get(tool_name)
        if tool is None:
            raise KeyError(f"unknown tool {tool_name!r}")
        return tool, tool.op(op_name)

    def summary_fields(self, ref: str) -> list[str]:
        """The op's summary fields; none for something that is not a registry op (a script's
        report)."""
        try:
            return self.resolve(ref)[1].returns.summary_fields
        except KeyError:
            return []

    def op_refs(self) -> list[str]:
        return [f"{t.name}.{op.name}" for t in self.tools.values() for op in t.ops]


def load_registry(
    directory: Path, active: set[str] | None = None, recipes: Path | None = None
) -> Registry:
    """Manifests from `directory`; recipes from `recipes`, default the sibling `recipes/`."""
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
    reg.known_tools = {t.name for t in loaded}
    _check_references(reg, loaded)
    _load_recipes(recipes or directory.parent / "recipes", reg)
    return reg


def _load_recipes(directory: Path, reg: Registry) -> None:
    for path in sorted(directory.glob("*.yaml")) if directory.is_dir() else []:
        _load_recipe(path, reg)


def _load_recipe(path: Path, reg: Registry) -> None:
    try:
        recipe = Recipe.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ValidationError) as exc:
        reg.problems.append(f"recipes/{path.name}: {exc}")
        return
    if recipe.name in reg.op_refs():
        reg.problems.append(f"recipes/{path.name}: {recipe.name} collides with an op")
        return
    typos = _template_typos(recipe)
    if typos:
        reg.problems.append(f"recipes/{path.name}: {'; '.join(typos)}")
        return
    missing = []
    for stage in recipe.steps + recipe.on_empty:
        problem = _stage_problem(reg, stage.op, stage.params)
        if problem == "missing":
            missing.append(stage.op)
        elif problem:
            reg.problems.append(f"recipes/{path.name}: stage {stage.id}: {problem}")
            return
    if missing:
        reg.notes.append(f"{recipe.name}: runs without {sorted(set(missing))} (not active here)")
    reg.recipes[recipe.name] = recipe


RECIPE_BUILTINS = {"window_start", "window_end"}
STAGE_REF = re.compile(r"\$([a-z][a-z0-9_]*)\.")
TEMPLATE_REF = re.compile(r"\{([a-z_]+)\}")


def _template_typos(recipe: Recipe) -> list[str]:
    """`$stage` must name a stage of this recipe; `{name}` a declared param or a builtin."""
    stages = {s.id for s in recipe.steps + recipe.on_empty}
    names = set(recipe.params) | RECIPE_BUILTINS
    out = []
    for stage in recipe.steps + recipe.on_empty:
        for key, value in stage.params.items():
            if not isinstance(value, str):
                continue
            out += [
                f"stage {stage.id} param {key}: no stage {ref!r}"
                for ref in STAGE_REF.findall(value)
                if ref not in stages
            ]
            out += [
                f"stage {stage.id} param {key}: no param {ref!r}"
                for ref in TEMPLATE_REF.findall(value)
                if ref not in names
            ]
    return out


def _stage_problem(reg: Registry, ref: str, params: dict) -> str | None:
    """None if fine, "missing" if the tool is not active here, else what is wrong."""
    tool = ref.partition(".")[0]
    if tool not in reg.tools:
        return "missing"
    try:
        validate_shape(reg.resolve(ref)[1], params)
    except KeyError:
        return f"{ref} is not an op of {tool}"
    except ParamError as exc:
        return f"{ref}: {exc}"
    return None


def _load_one(path: Path, reg: Registry) -> Tool | None:
    try:
        tool = Tool.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ValidationError) as exc:
        reg.problems.append(f"{path.name}: {exc}")
        return None
    if tool.name != path.stem:
        reg.problems.append(f"{path.name}: name {tool.name!r} does not match the file name")
        return None
    tool = _read_only_part(tool, path, reg)
    if tool is not None and (problem := refusal(tool)):
        reg.problems.append(f"{path.name}: {problem}")
        return None
    return tool


def _read_only_part(tool: Tool, path: Path, reg: Registry) -> Tool | None:
    """Honest declarations are respected: a mutating tool is refused, mutating ops are dropped,
    each with a note, and the rest of the tool still loads."""
    if tool.mutating:
        reg.notes.append(f"{path.name}: refused, the tool is marked mutating (read-only oracle)")
        return None
    dropped = [op.name for op in tool.ops if op.mutating]
    if not dropped:
        return tool
    reg.notes.append(f"{path.name}: dropped mutating ops {dropped}")
    kept = [op for op in tool.ops if not op.mutating]
    return tool.model_copy(update={"ops": kept}) if kept else None


def refusal(tool: Tool) -> str | None:
    """An op declared read-only whose command says otherwise: never load the tool. Ingestion
    applies the same check before it activates a staged manifest."""
    unsafe = {op.name: p for op in tool.ops if (p := op_problem(op.argv, _named(op), op.params))}
    # declared read-only, but the argv says otherwise: never load it
    return f"refused, {unsafe}" if unsafe else None


def _named(op: Op) -> str | None:
    """The name a non-CLI op runs by: its MCP tool, or its workflow function."""
    return op.mcp_tool or (op.entry.rsplit(":", 1)[-1] if op.entry else None)


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
