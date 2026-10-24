"""Stages 1-2 of `oracle tool add --mcp`: list a stdio MCP server's tools and turn the read-only
ones into ops. Params come straight from each tool's input schema (no model involved); a model
only picks which tools to expose, writes the `when` lines and one example each, grounded in the
real output of tools that need no arguments.
"""

import json
import os
import re
import shlex
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

from oracle.registry.loader import Registry
from oracle.registry.safety import mcp_problem
from oracle.registry.schema import Example, McpServer, Op, Param, Tool

TYPES = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list[str]",  # items are sent as strings; typed arrays are a later refinement
    "object": "json",
}
MAX_EXPOSED = 8


@dataclass
class McpTool:
    name: str
    description: str
    schema: dict[str, Any]
    read_only: bool | None = None
    destructive: bool | None = None

    @property
    def refused(self) -> str | None:
        """Hints are the server's claims, so they can only add a refusal, never lift one."""
        if self.destructive or self.read_only is False:
            return "the server marks it as changing state"
        return mcp_problem(self.name)


READ_VERBS = ("list", "get", "search", "find", "query", "show", "describe", "read", "count")


def _first_word(name: str) -> str:
    return re.split(r"[_.\-]|(?<=[a-z0-9])(?=[A-Z])", name)[0].lower()


@dataclass
class McpProbe:
    server: McpServer
    tools: list[McpTool] = field(default_factory=list)
    refused: dict[str, str] = field(default_factory=dict)

    @property
    def usable(self) -> list[McpTool]:
        return [t for t in self.tools if t.name not in self.refused]

    @property
    def sampleable(self) -> list[McpTool]:
        """Tools safe to call before any draft exists: named with a read verb and needing no
        arguments. A denylist cannot be complete (`rebuild_cache`, `reload`) and a server's
        read-only hint is only a claim, so this stage uses an allowlist of names."""
        return [
            t
            for t in self.usable
            if not t.schema.get("required") and _first_word(t.name) in READ_VERBS
        ]


def parse_command(command: str) -> McpServer:
    words = shlex.split(command)
    if not words:
        raise ValueError("--mcp needs the command that starts the server")
    return McpServer(command=words[0], args=words[1:])


async def probe_mcp(server: McpServer, env: dict[str, str], timeout: float = 30.0) -> McpProbe:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(command=server.command, args=server.args, env=env)
    with open(os.devnull, "w") as quiet:
        async with (
            stdio_client(params, errlog=quiet) as (read, write),
            ClientSession(read, write, read_timeout_seconds=timeout) as session,
        ):
            await session.initialize()
            listed = (await session.list_tools()).tools
    probe = McpProbe(server=server)
    for t in listed:
        hints = getattr(t, "annotations", None)
        tool = McpTool(
            name=t.name,
            description=(t.description or "").strip(),
            schema=dict(getattr(t, "input_schema", None) or {}),
            read_only=getattr(hints, "read_only_hint", None) if hints else None,
            destructive=getattr(hints, "destructive_hint", None) if hints else None,
        )
        probe.tools.append(tool)
        if reason := tool.refused:
            probe.refused[tool.name] = reason
    return probe


def op_name(tool: str) -> str:
    return tool.lower().replace("-", "_")


def params_of(schema: dict[str, Any]) -> dict[str, Param]:
    required = set(schema.get("required", []))
    out = {}
    for name, prop in (schema.get("properties") or {}).items():
        kind = prop.get("type", "string")
        kind = next((k for k in kind if k != "null"), "string") if isinstance(kind, list) else kind
        enum = (
            [str(e).lower() if isinstance(e, bool) else str(e) for e in prop["enum"]]
            if prop.get("enum")
            else None
        )
        default = prop.get("default")
        out[name] = Param(
            type=TYPES.get(kind, "str"),  # type: ignore[arg-type]
            required=name in required,
            default=default if isinstance(default, str | int | bool) else None,
            desc=str(prop.get("description", ""))[:200],
            enum=enum,
        )
    return out


class Exposed(BaseModel):
    tool: str = Field(description="the MCP tool name, exactly as listed")
    when: str = Field(description="one sentence: the question this op answers")
    example: dict[str, Any] = Field(
        description="arguments likely to return results on this machine"
    )


class McpDraft(BaseModel):
    description: str
    when: str
    not_when: str
    tags: list[str] = Field(default_factory=list, max_length=8)
    expose: list[Exposed] = Field(min_length=1, max_length=MAX_EXPOSED)


@dataclass
class McpContext:
    usable: dict[str, McpTool]


MCP_DRAFT = """You turn a read-only MCP server's tools into ops for a search oracle. Pick the 2 to 8
tools that answer distinct kinds of question; skip near-duplicates and admin or schema tools. For
each, write a one-sentence `when` in the words people use in questions and one example of
arguments that will return results on this machine: use real values from the sample output you
are given (real project names, ids). Required arguments must be present. Then write the tool's
description, when, not_when (naming existing tools to use instead where they overlap) and 4 to 8
routing tags. Never invent a tool name."""


def build_mcp_drafter(model: Model | str) -> Agent[McpContext, McpDraft]:
    agent = Agent(
        model, deps_type=McpContext, output_type=McpDraft, instructions=MCP_DRAFT, retries=2
    )

    @agent.output_validator
    def _valid(ctx: RunContext[McpContext], draft: McpDraft) -> McpDraft:
        problems = []
        for e in draft.expose:
            tool = ctx.deps.usable.get(e.tool)
            if tool is None:
                problems.append(
                    f"{e.tool!r} is not a usable tool; choose from {sorted(ctx.deps.usable)}"
                )
                continue
            missing = sorted(set(tool.schema.get("required", [])) - set(e.example))
            if missing:
                problems.append(f"{e.tool}: example lacks required arguments {missing}")
        if problems:
            raise ModelRetry("Fix these problems:\n- " + "\n- ".join(problems))
        return draft

    return agent


def render_mcp_probe(probe: McpProbe, samples: dict[str, str], registry: Registry) -> str:
    lines = [
        f"Server: {probe.server.command} {' '.join(probe.server.args)}".strip(),
        "",
        "Usable tools:",
    ]
    for t in probe.usable:
        lines.append(f"- {t.name}: {t.description[:300]}")
        lines.append(f"    arguments: {json.dumps(t.schema.get('properties', {}))[:600]}")
        lines.append(f"    required: {t.schema.get('required', [])}")
    if probe.refused:
        lines.append(f"\nRefused (changes state; never expose): {', '.join(probe.refused)}")
    if samples:
        lines += ["", "Real output of tools that need no arguments:"]
        lines += [f"- {name}: {text[:1200]}" for name, text in samples.items()]
    if registry.tools:
        lines += ["", "Existing tools:"]
        lines += [f"- {t.name}: {t.description} | when: {t.when}" for t in registry.tools.values()]
    return "\n".join(lines)


def tool_from_draft(name: str, probe: McpProbe, draft: McpDraft) -> Tool:
    usable = {t.name: t for t in probe.usable}
    ops = [
        Op(
            name=op_name(e.tool),
            when=e.when,
            mcp_tool=e.tool,
            params=params_of(usable[e.tool].schema),
            examples=[Example(params=e.example)],
        )
        for e in draft.expose
        if e.tool in usable
    ]
    return Tool(
        name=name,
        description=draft.description,
        tags=draft.tags,
        when=draft.when,
        not_when=draft.not_when,
        mcp=probe.server,
        ops=ops,
    )
