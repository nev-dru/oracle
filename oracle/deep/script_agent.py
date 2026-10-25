"""The deep tier's script writer (Opus by default, ORACLE_MODEL_DEEP): question + the stubs of
the shortlisted tools -> one Python script. The static check runs as an output validator, so a
refused script goes back to the model with the problems before anything runs; a script that
fails at run time is sent back once with its traceback (`failed`)."""

import ast
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

from oracle.budget import Budget
from oracle.deep.sandbox import script_problems
from oracle.plan.planner import PLAN_RETRIES, Shortlist, prompt
from oracle.registry.codegen import Library, ident

SDK_NAMES = {"gather", "report"}


class Script(BaseModel):
    interpretation: str = Field(description="one sentence: what the question is really asking")
    code: str = Field(description="the Python script, using only the listed tool modules")


@dataclass
class ScriptContext:
    modules: set[str]
    """Tool modules of this call's shortlist, as imported (`code_search`)."""


def build_script_agent(model: Model | str) -> Agent[ScriptContext, Script]:
    agent = Agent(
        model,
        deps_type=ScriptContext,
        output_type=Script,
        instructions=prompt("script"),
        retries=PLAN_RETRIES,
    )

    @agent.output_validator
    def _valid(ctx: RunContext[ScriptContext], script: Script) -> Script:
        problems = script_problems(script.code) or unlisted_modules(script.code, ctx.deps.modules)
        if problems:
            raise ModelRetry("The script was refused:\n- " + "\n- ".join(problems))
        return script

    return agent


def unlisted_modules(code: str, modules: set[str]) -> list[str]:
    """Tool modules a script imports that this call did not list (the runner refuses their
    calls anyway; saying so here saves a failed run)."""
    names: list[str] = []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.ImportFrom) and node.module == "oracle_tools":
            names += [a.name for a in node.names]
    return [f"oracle_tools.{n} is not listed for this question" for n in names
            if n not in modules | SDK_NAMES]  # fmt: skip


async def write_script(
    agent: Agent[ScriptContext, Script],
    *,
    question: str,
    when: str,
    shortlist: Shortlist,
    library: Library,
    budget: Budget,
    failed: tuple[str, str] | None = None,
) -> Script:
    tools = [t.tool for t in shortlist.tools]
    lines = [f"Question: {question}", f"Now: {when}", "", "Tool modules (oracle_tools):"]
    for tool in tools:
        lines += ["", f"# oracle_tools/{ident(tool)}.py", library.stubs[tool].strip()]
    if failed:
        code, error = failed
        lines += ["", "Your previous script failed. Fix it.", "```python", code, "```", error]
    deps = ScriptContext({ident(t) for t in tools})
    result = await budget.model(agent.run("\n".join(lines), deps=deps))
    return result.output
