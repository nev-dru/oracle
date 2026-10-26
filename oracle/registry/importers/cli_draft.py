"""Stages 2-4 of `oracle tool add` for a CLI: draft a manifest from the probe (Sonnet with the
manifest-authoring skill), repair a failing op once, and write the boundary with existing tools.

Every draft is validated before anything runs: the schema, argv[0] being the probed binary, and
the deterministic read-only check.
"""

from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

from oracle.config import REPO_ROOT
from oracle.registry.importers.cli_probe import Probe
from oracle.registry.loader import Registry
from oracle.registry.safety import op_problem
from oracle.registry.schema import Op, Tool, ToolVersion
from oracle.skills.loader import skill_text

SKILLS = REPO_ROOT / "skills"
MAX_OPS = 6
HELP_CHARS = 40_000


def skill(name: str) -> str:
    text = skill_text(SKILLS, name)
    if not text:
        raise FileNotFoundError(f"skill {name!r} is missing from {SKILLS} (run from a checkout)")
    return text


@dataclass
class DraftContext:
    name: str
    binary: str
    mcp: bool = False
    """True for an MCP import: ops must call the server's tools and never carry argv."""
    version: ToolVersion | None = None
    """What the probe read from the binary; the manifest records it, whatever the draft says."""


def op_problems(op: Op, binary: str, mcp: bool = False) -> list[str]:
    """For a CLI op, `binary` is the probed program; for an MCP op it is unused. Each import
    path accepts one transport: a CLI draft cannot name a program as an MCP server, and an MCP
    repair cannot switch to argv (`python -c ...` would pass as the server's launcher)."""
    problems = _transport_problems(op, mcp)
    if op.argv and op.argv[0] != binary:
        problems.append(f"{op.name}: argv[0] must be {binary!r}, got {op.argv[0]!r}")
    if problem := op_problem(op.argv, op.mcp_tool, op.params):
        problems.append(f"{op.name}: {problem}")
    if op.mutating:
        problems.append(f"{op.name}: only read-only ops may be drafted")
    if not op.examples:
        problems.append(f"{op.name}: needs one example")
    return problems


def _transport_problems(op: Op, mcp: bool) -> list[str]:
    if mcp and (op.argv or not op.mcp_tool):
        return [f"{op.name}: an MCP op calls one of the server's tools and has no argv"]
    if not mcp and (op.mcp_tool or not op.argv):
        return [f"{op.name}: a CLI op runs the probed binary via argv"]
    return []


def build_drafter(model: Model | str) -> Agent[DraftContext, Tool]:
    agent = Agent(
        model,
        deps_type=DraftContext,
        output_type=Tool,
        instructions=skill("manifest-authoring"),
        retries=2,
    )

    @agent.output_validator
    def _valid(ctx: RunContext[DraftContext], tool: Tool) -> Tool:
        problems = [] if tool.name == ctx.deps.name else [f"name must be {ctx.deps.name!r}"]
        if len(tool.ops) > MAX_OPS:
            problems.append(f"at most {MAX_OPS} ops")
        problems += [p for op in tool.ops for p in op_problems(op, ctx.deps.binary)]
        if tool.mcp is not None:
            problems.append("a CLI manifest has no mcp server block")
        if tool.mutating:
            problems.append("the tool must be read-only (mutating: false)")
        if problems:
            raise ModelRetry("Fix these problems:\n- " + "\n- ".join(problems))
        return tool

    return agent


def render_probe(probe: Probe, registry: Registry) -> str:
    lines = [f"Tool name: {probe.tool}", f"Binary: {probe.binary}"]
    if probe.version:
        lines.append(f"Version: {probe.version}")
    if probe.refused:
        lines.append(f"Refused subcommands (mutating; never use): {', '.join(probe.refused)}")
    if registry.tools:
        lines += ["", "Existing tools (write not_when against these where they overlap):"]
        lines += [
            f"- {t.name}: {t.description} | when: {t.when} | not_when: {t.not_when}"
            for t in registry.tools.values()
        ]
    lines += ["", "Help output:"]
    used = 0
    for words, text in probe.helps.items():
        command = " ".join(w for w in (probe.binary, words, "--help") if w)
        block = f"\n### {command}\n{text}"
        if used + len(block) > HELP_CHARS:
            lines.append("\n(remaining help omitted for length)")
            break
        lines.append(block)
        used += len(block)
    return "\n".join(lines)


def build_repairer(model: Model | str) -> Agent[DraftContext, Op]:
    agent = Agent(
        model,
        deps_type=DraftContext,
        output_type=Op,
        instructions=skill("manifest-authoring")
        + "\n\nYou are fixing one op whose example failed against the real tool. Change only what"
        " the error and the help text show is wrong. Keep the op name.",  # prompt-ok
        retries=2,
    )

    @agent.output_validator
    def _valid(ctx: RunContext[DraftContext], op: Op) -> Op:
        problems = op_problems(op, ctx.deps.binary, ctx.deps.mcp)
        if problems:
            raise ModelRetry("Fix these problems:\n- " + "\n- ".join(problems))
        return op

    return agent


class Proposal(BaseModel):
    tool: str = Field(description="an existing tool")
    when: str
    not_when: str
    why: str


class Boundary(BaseModel):
    description: str = Field(description="one sentence: what the new tool's listed ops do")
    when: str = Field(description="the new tool's when line")
    not_when: str = Field(description="the new tool's not_when line, naming tools to use instead")
    proposals: list[Proposal] = Field(
        default_factory=list, description="edits to existing tools, only where they now overlap"
    )


BOUNDARY = """You write the boundary between a newly added tool and the existing tools of a search
oracle, as one-sentence `when` and `not_when` lines in the words people use in questions. The
ranker and planner choose tools by these lines. Describe only the ops listed: ops that failed
verification were removed and must not be mentioned. Propose edits to existing tools only where
the new tool now overlaps them; a person reviews every proposal before it is applied."""


def build_boundary(model: Model | str) -> Agent[None, Boundary]:
    return Agent(model, output_type=Boundary, instructions=BOUNDARY, retries=2)


class Candidate(BaseModel):
    query: str = Field(description="a question someone would ask that this op answers")
    op: str


class Candidates(BaseModel):
    cases: list[Candidate] = Field(default_factory=list, max_length=30)


EVALS = """You write test questions for a search oracle's ranker. For each op of the new tool,
write 3 to 5 short questions, in the words a colleague would use, that this op answers. Ground
them in the real sample output you are given (real names, ids, statuses), and vary the phrasing.
Never mention the tool's name."""


def build_evals(model: Model | str) -> Agent[None, Candidates]:
    return Agent(model, output_type=Candidates, instructions=EVALS, retries=2)
