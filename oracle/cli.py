"""`oracle` entry point: ask (the default command), expand, and tool registry management."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

from oracle.config import REPO_ROOT, Config, ConfigError, load_config
from oracle.exec.executor import Executor
from oracle.exec.pointers import SpillError, expand_spill
from oracle.exec.selftest import ExampleResult, golden_ops_missing, run_examples
from oracle.log.store import open_store
from oracle.registry.loader import Registry, load_registry
from oracle.render import to_json, to_markdown

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
tool_app = typer.Typer(help="Registry management: list and test manifests.")
app.add_typer(tool_app, name="tool")

Format = Annotated[str, typer.Option("--format", help="md or json")]
Session = Annotated[str | None, typer.Option("--session", help="stable id per conversation")]


def _setup() -> tuple[Config, Registry]:
    try:
        cfg = load_config()
    except ConfigError as exc:
        _die(str(exc))
    active = set(cfg.profile.tools) if cfg.profile and cfg.profile.tools is not None else None
    return cfg, load_registry(cfg.registry_dir, active)


def _die(message: str) -> NoReturn:
    typer.echo(f"oracle: {message}", err=True)
    raise typer.Exit(2)


def _csv(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


@app.command("ask")
def ask_cmd(
    question: Annotated[str, typer.Argument(help="the question, in plain language")],
    intent: Annotated[str | None, typer.Option(help="why you are asking")] = None,
    level: Annotated[str, typer.Option(help="quick|standard|deep")] = "standard",
    scope: Annotated[str | None, typer.Option(help="only these tools, comma-separated")] = None,
    prefer: Annotated[str | None, typer.Option(help="favour these tools")] = None,
    fmt: Format = "md",
    session: Session = None,
) -> None:
    """Find things across every registered source (the default command)."""
    # Imported here: pydantic-ai is the slow import, and expand and tool commands never need it.
    from oracle.ask import LEVELS, AskRequest, ask

    if level not in LEVELS:
        _die(f"--level must be one of {', '.join(LEVELS)}")
    cfg, registry = _setup()
    req = AskRequest(question, intent, level, _csv(scope), _csv(prefer), session)
    try:
        answer = asyncio.run(ask(req, cfg, registry))
    except ConfigError as exc:
        _die(str(exc))
    typer.echo(to_json(answer) if fmt == "json" else to_markdown(answer))


@app.command()
def expand(
    pointer: Annotated[str, typer.Argument(help="a pointer from an earlier answer")],
    session: Session = None,
) -> None:
    """Print the full record behind a pointer (or the head of a spilled `file:` output)."""
    cfg, _ = _setup()
    if pointer.startswith("file:"):
        try:
            typer.echo(json.dumps(expand_spill(pointer, cfg.spill_dir), indent=2))
        except SpillError as exc:
            _die(str(exc))
        return

    async def lookup() -> None:
        async with open_store(cfg.log_path) as store:
            hit = await store.lookup(pointer, session)
        if hit is None:
            _die(f"no record for {pointer} (pointers live for the session, plus 24 h)")
        typer.echo(json.dumps({"pointer": hit.pointer, "op": hit.op, "call_id": hit.call_id,
                               "record": hit.record}, indent=2))  # fmt: skip

    asyncio.run(lookup())


@tool_app.command("list")
def tool_list() -> None:
    """Active tools, their ops, and any manifest problems or notes."""
    cfg, registry = _setup()
    profile = cfg.profile.name if cfg.profile else "none"
    typer.echo(f"registry {cfg.registry_dir}  profile {profile}")
    for tool in registry.tools.values():
        typer.echo(f"{tool.name}: {tool.description}")
        for op in tool.ops:
            typer.echo(f"  {tool.name}.{op.name}  {op.when}")
    if registry.inactive:
        typer.echo(f"inactive in this profile: {', '.join(registry.inactive)}")
    for problem in registry.problems:
        typer.echo(f"problem: {problem}")
    for note in registry.notes:
        typer.echo(f"note: {note}")


@tool_app.command("test")
def tool_test(
    names: Annotated[list[str] | None, typer.Argument(help="tools to test; default all")] = None,
) -> None:
    """Run every manifest example against the real tool. Exit 1 if any fails."""
    cfg, registry = _setup()
    unknown = sorted(set(names or []) - set(registry.tools))
    if unknown:
        problems = "".join(f"\n  problem: {p}" for p in registry.problems)
        _die(f"unknown tools {unknown}; active: {sorted(registry.tools)}{problems}")
    executor = Executor(registry, cfg.tool_env(), cfg.spill_dir)
    results = asyncio.run(run_examples(registry, executor, set(names) if names else None))
    raise typer.Exit(1 if _report(results, registry) else 0)


def _report(results: list[ExampleResult], registry: Registry) -> int:
    """Print example results, manifest problems and notes; return the number of failures."""
    for r in sorted(results, key=lambda r: (r.op, r.index)):
        typer.echo(f"{'PASS' if r.ok else 'FAIL'}  {r.op}[{r.index}]  {r.detail}  ({r.ms} ms)")
    for problem in registry.problems:
        typer.echo(f"FAIL  manifest  {problem}")
    notes = list(registry.notes)
    missing = golden_ops_missing(registry, REPO_ROOT / "golden.yaml")
    if missing:
        notes.append(f"golden.yaml names ops this registry lacks: {', '.join(missing)}")
    for note in notes:
        typer.echo(f"note: {note}")
    passed = sum(r.ok for r in results)
    typer.echo(f"{passed}/{len(results)} examples passed")
    if not results:
        typer.echo("FAIL  nothing was tested")
    return len(results) - passed + len(registry.problems) + (0 if results else 1)


@app.command()
def feedback(
    call_id: Annotated[str, typer.Argument(help="the call_id from an answer's trace")],
    useful: Annotated[str, typer.Option(help="yes or no")],
    note: Annotated[str | None, typer.Option(help="what was wrong or missing")] = None,
    source: Annotated[str, typer.Option(help="agent or human")] = "agent",
) -> None:
    """Record whether an answer was useful; cheap enough to call whenever one was wrong."""
    if useful not in ("yes", "no"):
        _die("--useful must be yes or no")
    cfg, _ = _setup()

    async def save() -> bool:
        async with open_store(cfg.log_path) as store:
            return await store.record_feedback(call_id, useful == "yes", note, source)

    if not asyncio.run(save()):
        _die(f"no call {call_id} in {cfg.log_path}")
    typer.echo(f"recorded: {call_id} useful={useful}")


@app.command("eval")
def eval_cmd(
    case: Annotated[
        list[str] | None, typer.Option("--case", help="run only these case ids")
    ] = None,
    no_judge: Annotated[bool, typer.Option("--no-judge", help="skip the Sonnet judge")] = False,
) -> None:
    """Run golden.yaml at each case's level; exit 1 if any case fails. Spends model calls."""
    from oracle.ask import Models
    from oracle.evals import last_run, load_cases, report, run_case, save_run
    from oracle.llm.models import resolve_model

    cfg, registry = _setup()
    cases = load_cases(REPO_ROOT / "golden.yaml", case)
    if not cases:
        _die(f"no golden cases match {case}")
    try:
        judge_model = None if no_judge else resolve_model(cfg.model("planner"))
    except ConfigError as exc:
        _die(str(exc))

    async def run_all() -> list:
        return [await run_case(c, cfg, registry, Models(), judge_model) for c in cases]

    try:
        results = asyncio.run(run_all())
    except ConfigError as exc:
        _die(str(exc))
    path = cfg.home / "eval" / "last.json"
    previous = last_run(path)
    typer.echo(report(results, previous))
    kept = [r for r in _previous_results(path) if r.get("id") not in {x.id for x in results}]
    save_run(path, results, kept)
    raise typer.Exit(0 if all(r.passed for r in results) else 1)


def _previous_results(path: Path) -> list[dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return []


COMMANDS = {"ask", "expand", "tool", "feedback", "eval", "--help", "-h"}


def main() -> None:
    """`oracle "<question>"` is shorthand for `oracle ask "<question>"`."""
    if len(sys.argv) > 1 and sys.argv[1] not in COMMANDS:
        sys.argv.insert(1, "ask")
    app()
