"""Pydantic models for tool manifests (<registry>/<name>.yaml).

Everything the planner sees, the executor validates and the codegen emits comes from these models.
Unknown fields are rejected so a typo in a manifest fails loudly instead of being ignored.
"""

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from oracle.registry import jsonpath

NAME = r"^[a-z][a-z0-9-]*$"
OP_NAME = r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$"
PLACEHOLDER = re.compile(r"\{([^{}]+)\}")

ParamType = Literal["str", "int", "float", "bool", "list[str]", "json"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Param(Strict):
    type: ParamType = "str"
    required: bool = False
    default: Any = None
    desc: str = ""
    enum: list[str] | None = None
    enum_from: str | None = None
    max: int | None = None


def _valid_path(path: str) -> str:
    jsonpath.parse(path)  # raises PathError (a ValueError) on anything outside the subset
    return path


class Paginate(Strict):
    param: str
    next: str
    max_pages: int = Field(default=5, ge=1, le=50)

    _next_ok = field_validator("next")(_valid_path)


class Returns(Strict):
    format: Literal["json", "jsonl", "table", "text"] = "json"
    path: str = "$"
    id_field: str | None = None
    summary_fields: list[str] = Field(default_factory=list)
    pointer: str | None = None

    _path_ok = field_validator("path")(_valid_path)


class Expect(Strict):
    min_results: int = 0
    has_fields: list[str] = Field(default_factory=list)


class Example(Strict):
    params: dict[str, Any] = Field(default_factory=dict)
    expect: Expect = Field(default_factory=Expect)


ENTRY = r"^oracle_workflows\.[a-z_][a-z0-9_]*:[a-z_][a-z0-9_]*$"


class OpBudget(Strict):
    """A workflow's own limits: nested oracle calls and wall-clock seconds."""

    model_calls: int | None = Field(default=None, ge=0)
    wall_s: float | None = Field(default=None, gt=0)


class Op(Strict):
    name: str = Field(pattern=OP_NAME)
    when: str
    params: dict[str, Param] = Field(default_factory=dict)
    argv: list[str] = Field(default_factory=list, description="the command, for a CLI tool")
    mcp_tool: str | None = Field(
        default=None, description="the MCP tool to call, for a tool with an `mcp` server"
    )
    entry: str | None = Field(
        default=None,
        pattern=ENTRY,
        description="the function to run, `oracle_workflows.<module>:<fn>`, for a workflow",
    )
    budget: OpBudget | None = None
    returns: Returns = Field(default_factory=Returns)
    paginate: Paginate | None = None
    chains_to: list[str] = Field(default_factory=list)
    examples: list[Example] = Field(default_factory=list)
    mutating: bool = False

    @model_validator(mode="after")
    def _placeholders_are_params(self) -> "Op":
        if [bool(self.argv), bool(self.mcp_tool), bool(self.entry)].count(True) != 1:
            raise ValueError(
                "an op has exactly one of argv (a CLI), mcp_tool (an MCP tool) "
                "or entry (a workflow)"
            )
        if self.paginate and self.entry:
            raise ValueError("a workflow op does not page")
        known = set(self.params) | ({self.paginate.param} if self.paginate else set())
        for item in self.argv:
            for ref in placeholder_params(item):
                if ref not in known:
                    raise ValueError(f"argv references undeclared param {ref!r} in {item!r}")
        return self


class Auth(Strict):
    env: list[str] = Field(default_factory=list)


class McpServer(Strict):
    """A stdio MCP server; its tools become ops with `mcp_tool` instead of argv."""

    command: str
    args: list[str] = Field(default_factory=list)


class Limits(Strict):
    timeout_s: float = Field(default=20, gt=0)
    concurrency: int = Field(default=4, ge=1)
    max_output_kb: int = Field(default=256, ge=1)


class Tool(Strict):
    name: str = Field(pattern=NAME)
    description: str
    tags: list[str] = Field(default_factory=list)
    when: str = ""
    not_when: str = ""
    auth: Auth = Field(default_factory=Auth)
    limits: Limits = Field(default_factory=Limits)
    mutating: bool = False
    kind: Literal["tool", "workflow"] = "tool"
    """A workflow is trusted Python under the profile's workflows/, run in-process."""
    skills: list[str] = Field(default_factory=list, description="tool-scoped skills to attach")
    mcp: McpServer | None = None
    ops: list[Op] = Field(min_length=1)

    @model_validator(mode="after")
    def _transport_matches(self) -> "Tool":
        wrong = [op.name for op in self.ops if bool(op.mcp_tool) != bool(self.mcp)]
        if wrong:
            raise ValueError(f"ops {wrong}: use mcp_tool exactly when the tool has an mcp server")
        workflow = self.kind == "workflow"
        wrong = [op.name for op in self.ops if bool(op.entry) != workflow]
        if wrong:
            raise ValueError(f"ops {wrong}: use entry exactly when the tool is a workflow")
        return self

    @model_validator(mode="after")
    def _unique_ops(self) -> "Tool":
        names = [op.name for op in self.ops]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate op names: {sorted(dupes)}")
        return self

    def op(self, name: str) -> Op:
        for op in self.ops:
            if op.name == name:
                return op
        raise KeyError(f"{self.name}.{name}")


def placeholder_params(item: str) -> list[str]:
    """Param names referenced by one argv item: {p}, {?p:--flag,p} and {stdin:p}."""
    out = []
    for body in PLACEHOLDER.findall(item):
        if body.startswith("?"):
            cond, _, rest = body[1:].partition(":")
            out.append(cond)
            out += [part for part in rest.split(",")[1:] if part]
        elif body.startswith("stdin:"):
            out.append(body.removeprefix("stdin:"))
        else:
            out.append(body)
    return out


RECIPE_NAME = r"^[a-z][a-z0-9-]*(\.[a-z][a-z0-9_]*)+$"


class RecipeStep(Strict):
    """One stage of a recipe. String params may use the recipe's params as `{name}`, the
    resolved time window as `{window_start}` / `{window_end}`, and earlier stages as bindings
    (`$search.pointers`)."""

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    op: str
    params: dict[str, Any] = Field(default_factory=dict)
    after: list[str] = Field(default_factory=list)


class Recipe(Strict):
    """A composite op: a stored plan with free params. The planner sees it as one op."""

    name: str = Field(pattern=RECIPE_NAME)
    when: str
    params: dict[str, Param] = Field(default_factory=dict)
    steps: list[RecipeStep] = Field(min_length=1)
    output: list[str] = Field(default_factory=list, description="stages whose results count")
    on_empty: list[RecipeStep] = Field(
        default_factory=list, description="run instead when every output stage returned nothing"
    )
    breadth: bool = False

    @model_validator(mode="after")
    def _stages_line_up(self) -> "Recipe":
        ids = [s.id for s in self.steps] + [s.id for s in self.on_empty]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate stage ids: {ids}")
        unknown = [a for s in self.steps + self.on_empty for a in s.after if a not in ids]
        unknown += [o for o in self.output if o not in ids]
        if unknown:
            raise ValueError(f"unknown stage ids {unknown}")
        return self

    @property
    def outputs(self) -> list[str]:
        return self.output or [s.id for s in self.steps]
