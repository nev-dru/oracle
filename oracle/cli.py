"""`oracle` entry point: ask (the default command), expand, and tool registry management."""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

from oracle.config import REPO_ROOT, Config, ConfigError, load_config
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
    level: Annotated[
        str | None, typer.Option(help="quick|standard|deep (default standard)")
    ] = None,
    scope: Annotated[str | None, typer.Option(help="only these tools, comma-separated")] = None,
    prefer: Annotated[str | None, typer.Option(help="favour these tools")] = None,
    breadth: Annotated[
        bool, typer.Option("--breadth", help="fan out across every applicable tool")
    ] = False,
    depth: Annotated[int, typer.Option(help="deep crawl: hops of searching", min=1)] = 3,
    budget: Annotated[
        str | None, typer.Option(help="deep crawl: model tokens, or 'unlimited'")
    ] = None,
    script: Annotated[
        bool, typer.Option("--script", help="deep: write one script against the tools")
    ] = False,
    fmt: Format = "md",
    session: Session = None,
) -> None:
    """Find things across every registered source (the default command)."""
    # Imported here: pydantic-ai is the slow import, and expand and tool commands never need it.
    from oracle.ask import LEVELS, AskRequest

    if level is not None and level not in LEVELS:
        _die(f"--level must be one of {', '.join(LEVELS)}")
    from oracle.budget import parse_budget

    try:
        parse_budget(budget, 1)
    except ValueError:
        _die("--budget must be a positive number of tokens or 'unlimited'")
    entry = _entry(question, level, script)
    cfg, registry = _setup()
    req = AskRequest(
        question, intent, entry.level, _csv(scope), _csv(prefer), session, breadth, depth,
        budget, entry.script,
    )  # fmt: skip
    if entry.clamped:
        typer.echo(f"oracle: {entry.clamped}", err=True)
    try:
        answer = asyncio.run(_ask_interruptibly(req, cfg, registry, entry))
    except ConfigError as exc:
        _die(str(exc))
    typer.echo(to_json(answer) if fmt == "json" else to_markdown(answer))


def _entry(question: str, level: str | None, script: bool) -> Any:
    """Inside ORACLE_TRACE this call is nested, so it enters the guard (refused on a cycle or
    too deep) and runs no higher than its parent, one lower by default."""
    from oracle.workflows.protocol import GuardError, Trace, enter_ask

    parent = Trace.from_env(os.environ.get("ORACLE_TRACE"))
    try:
        return enter_ask(parent, question, level, script)
    except GuardError as exc:
        _die(f"refused: {exc}")


async def _ask_interruptibly(req: Any, cfg: Config, registry: Registry, entry: Any) -> Any:
    """The first Ctrl-C ends a deep crawl at the next hop and still prints what it found; a
    second one interrupts for real."""
    import signal

    from oracle.ask import ask

    if req.level != "deep" or req.script:  # only a crawl has hops to stop between
        return await ask(req, cfg, registry, entry=entry)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    def first_interrupt() -> None:
        stop.set()
        loop.remove_signal_handler(signal.SIGINT)
        typer.echo("oracle: stopping after this hop (Ctrl-C again to abort)", err=True)

    loop.add_signal_handler(signal.SIGINT, first_interrupt)
    try:
        return await ask(req, cfg, registry, stop=stop, entry=entry)
    finally:
        loop.remove_signal_handler(signal.SIGINT)


@app.command("run")
def run_cmd(
    ref: Annotated[str, typer.Argument(help="tool.op, or a tool that has one op")],
    params: Annotated[
        list[str] | None, typer.Argument(help="key=value ...; non-string values as JSON")
    ] = None,
    session: Session = None,
) -> None:
    """Run one op or workflow directly, through the executor, and print its records as JSON."""
    from oracle.workflows.protocol import GuardError, Trace

    cfg, registry = _setup()
    ref = _op_ref(registry, ref)
    values = _params(registry, ref, params or [])
    parent = Trace.from_env(os.environ.get("ORACLE_TRACE")) or Trace.root("deep")
    try:  # a tool that runs `oracle run` on itself is a cycle
        trace = parent.enter("run", ref, values)
    except GuardError as exc:
        _die(f"refused: {exc}")
    result = asyncio.run(_run_one(cfg, registry, ref, values, trace, session))
    rows = [
        {"pointer": p, "record": r} for p, r in zip(result.pointers, result.records, strict=True)
    ]
    out = {"op": ref, "status": result.status, "error": result.error, "results": rows}
    typer.echo(json.dumps(out, indent=2, default=str))
    if not result.ok:
        raise typer.Exit(1)


def _op_ref(registry: Registry, ref: str) -> str:
    tool = registry.tools.get(ref)
    if tool is not None and len(tool.ops) == 1:
        return f"{ref}.{tool.ops[0].name}"
    if tool is not None:
        names = ", ".join(f"{ref}.{op.name}" for op in tool.ops)
        _die(f"{ref} has several ops; name one: {names}")
    try:
        registry.resolve(ref)
    except KeyError:
        _die(f"no op {ref!r}; see `oracle tool list`")
    return ref


def _params(registry: Registry, ref: str, pairs: list[str]) -> dict[str, Any]:
    """key=value; a value is JSON (number, bool, list) unless the param is a string."""
    op = registry.resolve(ref)[1]
    values: dict[str, Any] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep:
            _die(f"expected key=value, got {pair!r}")
        param = op.params.get(key)
        if param is None or param.type == "str":
            values[key] = raw
            continue
        try:  # `flag=True` as well as `flag=true`
            values[key] = json.loads(raw.lower() if param.type == "bool" else raw)
        except json.JSONDecodeError as exc:
            _die(f'{key} is {param.type}: {raw!r} is not JSON ({exc.msg}); e.g. 10, true, ["a"]')
    return values


async def _run_one(
    cfg: Config,
    registry: Registry,
    ref: str,
    values: dict[str, Any],
    trace: Any,
    session: str | None,
) -> Any:
    """Logged like a one-step call, so its pointers expand."""
    import time
    import uuid

    from oracle.log.store import CallRecord
    from oracle.workflows.runner import call_scope

    start = time.monotonic()
    with call_scope(cfg, registry, trace) as executor:
        result = await executor.run(ref, values)
    call_id = uuid.uuid4().hex[:12]
    session = session or f"adhoc-{call_id}"
    record = CallRecord(
        call_id=call_id, session=session, query=f"run {ref} {json.dumps(values)}", intent=None,
        level="run", ranked_tools=[], plan=None, wall_ms=int((time.monotonic() - start) * 1000),
        confidence=None, trace_id=trace.trace_id, depth=trace.depth,
    )  # fmt: skip
    async with open_store(cfg.log_path) as store:
        await store.record_call(record)
        await store.record_op(call_id, session, "s1", result)
    return result


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
    ranker: Annotated[
        bool, typer.Option("--ranker", help="score the ranker on ranker_eval.yaml instead")
    ] = False,
) -> None:
    """Run every manifest example against the real tool. Exit 1 if any fails."""
    cfg, registry = _setup()
    if ranker:
        from oracle.rank.eval import run_ranker_eval

        report = run_ranker_eval(registry, REPO_ROOT / "ranker_eval.yaml")
        typer.echo("\n".join(report.lines()))
        raise typer.Exit(0 if report.passed else 1)
    unknown = sorted(set(names or []) - set(registry.tools))
    if unknown:
        problems = "".join(f"\n  problem: {p}" for p in registry.problems)
        _die(f"unknown tools {unknown}; active: {sorted(registry.tools)}{problems}")
    from oracle.workflows.protocol import Trace
    from oracle.workflows.runner import call_scope

    with call_scope(cfg, registry, Trace.root("deep")) as executor:  # workflow examples run too
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
    from oracle.evals import last_run, report, run_case, save_run
    from oracle.llm.models import resolve_model

    cfg, registry = _setup()
    cases = _golden_cases(cfg, case)
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


def _golden_cases(cfg: Config, only: list[str] | None) -> list[dict[str, Any]]:
    """golden.yaml's cases as the active profile runs them."""
    from oracle.evals import NO_PROFILE, load_cases

    profile = cfg.profile.name if cfg.profile else NO_PROFILE
    try:
        cases = load_cases(REPO_ROOT / "golden.yaml", only, profile)
    except ValueError as exc:
        _die(str(exc))
    if not cases:
        _die(f"no golden cases match {only}")
    return cases


def _previous_results(path: Path) -> list[dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return []


@tool_app.command("add")
def tool_add(
    binary: Annotated[
        str | None, typer.Argument(help="the CLI to add, as it is run (on PATH)")
    ] = None,
    mcp: Annotated[
        str | None, typer.Option("--mcp", help="instead, the command that starts an MCP server")
    ] = None,
    name: Annotated[str | None, typer.Option(help="manifest name; default the binary")] = None,
    yes: Annotated[
        bool, typer.Option("--yes", help="activate if every op verified and nothing overlaps")
    ] = False,
) -> None:
    """Probe a CLI's real --help, draft a manifest, verify it against the tool, and stage it
    for review. Spends roughly 5-30 model calls."""
    from oracle.workflows.ingest import IngestError, activate

    _check_add_args(binary, mcp, name)
    cfg, registry = _setup()
    source = binary or (mcp or "").split()[0]
    tool = name or source.rsplit("/", 1)[-1]
    result = _ingest(cfg, registry, tool, binary, mcp)
    typer.echo((result.staged / "report.md").read_text())
    if yes and result.ready:
        try:
            typer.echo(f"activated: {activate(cfg, tool)}")
        except IngestError as exc:
            _die(str(exc))
    elif yes:
        typer.echo("not activated: needs review (see report); run `oracle tool activate` after")
    raise typer.Exit(0 if result.manifest else 1)


def _check_add_args(binary: str | None, mcp: str | None, name: str | None) -> None:
    if bool(binary) == bool(mcp):
        _die("give either a CLI binary or --mcp '<server command>'")
    if mcp and not name:
        _die("an MCP server needs --name: its command is usually a launcher such as uvx or python")


def _ingest(cfg: Config, registry: Registry, tool: str, binary: str | None, mcp: str | None) -> Any:
    from oracle.llm.models import resolve_model
    from oracle.workflows.ingest import IngestError, ingest_cli, ingest_mcp

    try:
        model = resolve_model(cfg.model("planner"))
        if mcp:
            return asyncio.run(ingest_mcp(mcp, tool, cfg, registry, model))
        return asyncio.run(ingest_cli(binary or "", tool, cfg, registry, model))
    except (ConfigError, IngestError) as exc:
        _die(str(exc))


@tool_app.command("activate")
def tool_activate(name: Annotated[str, typer.Argument(help="a tool staged by `tool add`")]) -> None:
    """Copy a reviewed, staged manifest into the active registry."""
    from oracle.workflows.ingest import IngestError, activate

    cfg, _ = _setup()
    try:
        typer.echo(f"activated: {activate(cfg, name)}")
    except IngestError as exc:
        _die(str(exc))
    typer.echo(f"next: oracle tool test {name}")


@tool_app.command("build")
def tool_build() -> None:
    """Generate the `oracle_tools` library that deep-tier scripts import (also done on demand)."""
    from oracle.registry.codegen import build_library

    cfg, registry = _setup()
    library = build_library(registry, cfg.home)
    for problem in library.problems:
        typer.echo(f"skipped: {problem}", err=True)
    typer.echo(f"{library.path}  ({len(library.stubs)} tool modules)")


@app.command()
def report(
    days: Annotated[float, typer.Option(help="look back this many days")] = 7.0,
) -> None:
    """Implicit signals from the log: re-asked questions, used findings, replans, ranker misses
    (written as candidates for review, never into ranker_eval.yaml) and feedback."""
    import time

    from oracle.log.report import build_report, write_candidates

    cfg, _ = _setup()

    async def run() -> list[str]:
        async with open_store(cfg.log_path) as store:
            rep = await build_report(store.db, time.time() - days * 86400)
        path = cfg.home / "report" / "ranker_candidates.yaml"
        n = write_candidates(rep, path)
        return [*rep.lines(), f"{n} ranker candidates written to {path} (review before adding)"]

    typer.echo("\n".join(asyncio.run(run())))


@app.command("dream")
def dream_cmd(
    days: Annotated[float, typer.Option(help="look back this many days")] = 14.0,
    min_calls: Annotated[int, typer.Option(help="a chain must repeat this often")] = 3,
    accept_name: Annotated[
        str | None, typer.Option("--accept", help="install a staged recipe by name")
    ] = None,
) -> None:
    """Propose recipes from crawl chains that repeat in the log; `--accept` installs one.
    Nothing is applied or committed without --accept."""
    import yaml

    from oracle.log.dream import accept, dream

    cfg, _ = _setup()
    if accept_name:
        try:
            typer.echo(f"accepted: {accept(cfg.home, cfg.registry_dir, accept_name)}")
        except (OSError, ValueError, yaml.YAMLError) as exc:  # a corrupt staged file too
            _die(str(exc))
        typer.echo("next: oracle tool list, then oracle eval")
        return

    async def run() -> str:
        async with open_store(cfg.log_path) as store:
            result = await dream(store.db, cfg.registry_dir, cfg.home, days, min_calls)
        return (result.staged / "report.md").read_text()

    typer.echo(asyncio.run(run()))


COMMANDS = {"ask", "expand", "run", "tool", "feedback", "eval", "report", "dream", "--help", "-h"}


def main() -> None:
    """`oracle "<question>"` is shorthand for `oracle ask "<question>"`."""
    if len(sys.argv) > 1 and sys.argv[1] not in COMMANDS:
        sys.argv.insert(1, "ask")
    app()
