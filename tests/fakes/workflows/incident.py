"""Example workflows for the tests: an incident timeline from four tools in one call, plus two
that exercise the recursion guard."""

import asyncio
import json
from typing import Any


async def investigate(params: dict[str, Any], ctx: Any) -> list[dict[str, Any]]:
    """Warnings, deploys, error logs and tickets that mention the service, newest first."""
    service = params["service"]
    results = await asyncio.gather(
        ctx.tools.run("kubectl.events", all_namespaces=True, types="Warning"),
        ctx.tools.run("argocd.app.history", name=service),
        ctx.tools.run("es.search", index="logs-*", query=f"service:{service}", since="1d"),
        ctx.tools.run("acli.jira.search", jql=f'text ~ "{service}"', limit=5),
    )
    out = []
    for result in results:
        for pointer, record in zip(result.pointers, result.records, strict=True):
            text = json.dumps(record, default=str)
            if service.split("-")[0] in text:
                out.append({"pointer": pointer, "source": result.op, "text": text[:200]})
    ctx.log(f"{len(out)} items for {service}")
    return out


async def again(params: dict[str, Any], ctx: Any) -> list[dict[str, Any]]:
    """Calls itself with the same input: the guard refuses the second call as a cycle."""
    inner = await ctx.tools.run("incident.again", service=params["service"])
    return [{"pointer": "x:1", "source": "again", "text": inner.error or "ran"}]


async def nested(params: dict[str, Any], ctx: Any) -> list[dict[str, Any]]:
    """Asks the oracle a question of its own; reports the tier it ran at."""
    answer = await ctx.oracle.ask(params["question"], level=params.get("level"))
    note = "; ".join(ctx.oracle.notes)
    return [{"pointer": "x:2", "source": "nested", "text": f"{answer.level} {note}".strip()}]


async def nested_scoped(params: dict[str, Any], ctx: Any) -> list[dict[str, Any]]:
    """Asks the oracle within a scope (the guard tests use it to nest workflows)."""
    answer = await ctx.oracle.ask(params["question"], scope=params.get("scope"))
    ctx.log(f"inner call ran at {answer.level}")
    return [{"pointer": "x:3", "source": "nested", "text": "; ".join(answer.failures) or "ok"}]


async def broken(params: dict[str, Any], ctx: Any) -> list[dict[str, Any]]:
    """A bug in a workflow body: an error, not a setup problem."""
    return [{"pointer": "x:4", "source": "broken", "text": params["no_such_param"]}]
