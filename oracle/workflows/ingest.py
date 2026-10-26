"""`oracle tool add`: turn a CLI into a verified manifest without hand-authoring it.

The governing rule: every fact in the manifest is grounded in a real execution. Stages:
1 probe (real --help) -> 2 draft (Sonnet) -> 3 verify (run every example; repair once; infer
`returns` from the real JSON; drop what still fails) -> 4 boundaries (ranker impact plus `when`
lines; proposals for existing tools are never applied) -> 5 eval candidates (a separate file,
never golden.yaml) -> 6 review (a staged directory and a report; `oracle tool activate` is a
second, deliberate command).
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic_ai.models import Model

from oracle.budget import Usage
from oracle.config import Config
from oracle.exec.executor import Executor, OpResult
from oracle.exec.selftest import check
from oracle.rank.eval import ranker_files, run_ranker_eval
from oracle.registry.importers.cli_draft import (
    DraftContext,
    build_boundary,
    build_drafter,
    build_evals,
    build_repairer,
    op_problems,
    render_probe,
)
from oracle.registry.importers.cli_probe import probe_cli
from oracle.registry.importers.infer import infer_returns
from oracle.registry.jsonpath import get_field
from oracle.registry.loader import Registry, refusal
from oracle.registry.schema import NAME, Example, Expect, Op, Returns, Tool, ToolVersion


class IngestError(RuntimeError):
    pass


@dataclass
class OpOutcome:
    name: str
    kept: bool
    detail: str
    empty: bool = False
    """Kept, but its example returned nothing: the output may go to stderr, or the example
    matches nothing on this machine. Either way a person should look."""


@dataclass
class IngestResult:
    name: str
    staged: Path
    manifest: Tool | None
    outcomes: list[OpOutcome] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    steals: list[str] = field(default_factory=list)
    proposals: list[dict[str, str]] = field(default_factory=list)
    candidates: int = 0
    candidate_hit_rate: float | None = None
    usage: Usage = field(default_factory=Usage)
    records: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    """Real output of each verified op's example, reused by later stages instead of rerunning."""

    @property
    def ready(self) -> bool:
        """Safe to activate without review: every op verified, no ranker steals, no edits
        proposed to other tools."""
        verified = all(o.kept and not o.empty for o in self.outcomes)
        return bool(self.manifest) and verified and not (self.steals or self.proposals)


def staging_dir(cfg: Config, name: str) -> Path:
    return cfg.home / "ingest" / name


async def ingest_cli(
    binary: str, name: str, cfg: Config, registry: Registry, model: Model | str
) -> IngestResult:
    result = _start(name, cfg, registry)
    staged = result.staged
    probe = await probe_cli(name, binary, cfg.tool_env())
    (staged / "probe.json").write_text(json.dumps(probe.to_dict(), indent=2))
    if probe.path is None:
        raise IngestError(f"{binary!r} is not on PATH (profile PATH included)")
    result.refused = probe.refused
    version = ToolVersion(argv=probe.version_argv, seen=probe.version) if probe.version else None
    deps = DraftContext(name=name, binary=binary, version=version)
    drafted = await build_drafter(model).run(render_probe(probe, registry), deps=deps)
    result.usage.add(drafted.usage)
    return await _finish(drafted.output, probe.helps, cfg, registry, model, deps, result)


async def ingest_mcp(
    command: str, name: str, cfg: Config, registry: Registry, model: Model | str
) -> IngestResult:
    """The same pipeline for a stdio MCP server: list its tools, keep the read-only ones."""
    from oracle.registry.importers.mcp_import import (
        McpContext,
        build_mcp_drafter,
        parse_command,
        probe_mcp,
        render_mcp_probe,
        tool_from_draft,
    )

    result = _start(name, cfg, registry)
    try:
        probe = await probe_mcp(parse_command(command), cfg.tool_env())
    except ImportError:
        raise IngestError("MCP import needs the extra: uv sync --extra mcp") from None
    except Exception as exc:  # spawn failures, protocol errors, exception groups from anyio
        raise IngestError(
            f"could not list the server's tools: {type(exc).__name__}: {exc}"
        ) from None
    result.refused = sorted(probe.refused)
    (result.staged / "probe.json").write_text(
        json.dumps(
            {
                "server": probe.server.model_dump(),
                "tools": [vars(t) for t in probe.tools],
                "refused": probe.refused,
            },
            indent=2,
            default=str,
        )
    )
    samples = await _mcp_samples(name, probe, cfg)
    usable = {t.name: t for t in probe.usable}
    drafted = await build_mcp_drafter(model).run(
        render_mcp_probe(probe, samples, registry), deps=McpContext(usable)
    )
    result.usage.add(drafted.usage)
    draft = tool_from_draft(name, probe, drafted.output)
    helps = {t.name: f"{t.description}\narguments: {json.dumps(t.schema)}" for t in probe.usable}
    deps = DraftContext(name=name, binary=probe.server.command, mcp=True)
    return await _finish(draft, helps, cfg, registry, model, deps, result)


async def _mcp_samples(name: str, probe: Any, cfg: Config) -> dict[str, str]:
    """Real output of the read-only tools that need no arguments, to ground the examples."""
    from oracle.registry.importers.mcp_import import op_name

    out = {}
    for t in probe.sampleable:
        op = Op(name=op_name(t.name), when="sample", mcp_tool=t.name)
        tool = Tool(name=name, description="probe", mcp=probe.server, ops=[op])
        raw = await _run(tool, _probing(op), Example(), cfg)
        if raw.ok:
            out[t.name] = json.dumps(raw.records[:3], default=str)
    return out


STAGED = (
    "probe.json",
    "draft.yaml",
    "manifest.yaml",
    "boundary.md",
    "eval_candidates.yaml",
    "report.md",
)


def _start(name: str, cfg: Config, registry: Registry) -> IngestResult:
    """Checks that cost nothing come first: the name, and whether a manifest already exists
    (loaded or not). A previous run's staged files are cleared, so a failed re-run can never
    leave an old manifest for `activate` to install."""
    if not re.fullmatch(NAME, name):
        raise IngestError(f"{name!r} is not a valid tool name (lowercase, digits, -); use --name")
    if name in registry.tools or (cfg.registry_dir / f"{name}.yaml").exists():
        raise IngestError(f"{name} already has a manifest in {cfg.registry_dir}; remove it first")
    staged = staging_dir(cfg, name)
    staged.mkdir(parents=True, exist_ok=True)
    for stale in STAGED:
        (staged / stale).unlink(missing_ok=True)
    return IngestResult(name=name, staged=staged, manifest=None)


async def _finish(
    draft: Tool,
    helps: dict[str, str],
    cfg: Config,
    registry: Registry,
    model: Model | str,
    deps: DraftContext,
    result: IngestResult,
) -> IngestResult:
    """Stages 3-6, shared by every import path."""
    _write_yaml(result.staged / "draft.yaml", draft)
    verified = await _verify(draft, helps, cfg, model, deps, result)
    if verified is None:
        _write_report(result)
        return result
    verified = await _boundaries(verified, registry, cfg, model, result)
    verified = verified.model_copy(update={"version": deps.version})
    result.manifest = verified
    _write_yaml(result.staged / "manifest.yaml", verified)
    await _candidates(verified, registry, cfg, model, result)
    _write_report(result)
    return result


# --- stage 3: verify -------------------------------------------------------------------------


async def _verify(
    draft: Tool,
    helps: dict[str, str],
    cfg: Config,
    model: Model | str,
    deps: DraftContext,
    result: IngestResult,
) -> Tool | None:
    kept: list[Op] = []
    samples: dict[str, list[dict[str, Any]]] = {}
    for op in draft.ops:  # in order, so a later op can be repaired with real ids from earlier ones
        verified, detail = await _verify_op(draft, op, helps, cfg, model, deps, result, samples)
        empty = verified is not None and verified.examples[0].expect.min_results == 0
        result.outcomes.append(OpOutcome(op.name, verified is not None, detail, empty))
        if verified is not None:
            kept.append(verified)
            samples[op.name] = result.records.get(op.name, [])[:3]
    if not kept:
        return None
    drafted = {op.name for op in draft.ops}
    return draft.model_copy(update={"ops": [_chains(draft.name, op, kept, drafted) for op in kept]})


def _chains(tool: str, op: Op, kept: list[Op], drafted: set[str] | None = None) -> Op:
    """References to this tool's own ops (chains_to, enum_from) become `tool.op`, since drafts
    often write just `op`; one to an op that was dropped would break the manifest, so it goes.
    `drafted` holds every op name the draft had, so a dropped `chunk.get` is recognised as
    this tool's rather than read as tool `chunk`, op `get`."""
    names = {o.name for o in kept}
    mine = names | (drafted or set())
    chains = [r for ref in op.chains_to if (r := _ref(tool, ref, names, mine))]
    params = {
        k: p.model_copy(update={"enum_from": _ref(tool, p.enum_from, names, mine)})
        if p.enum_from
        else p
        for k, p in op.params.items()
    }
    return op.model_copy(update={"chains_to": chains, "params": params})


def _ref(tool: str, ref: str, kept: set[str], mine: set[str]) -> str | None:
    local = ref.removeprefix(f"{tool}.")
    if local in kept:
        return f"{tool}.{local}"
    if local in mine or ref.startswith(f"{tool}."):
        return None  # this tool's op, dropped in verification
    return ref if "." in ref else None  # another tool's op is kept as written


async def _verify_op(
    draft: Tool,
    op: Op,
    helps: dict[str, str],
    cfg: Config,
    model: Model | str,
    deps: DraftContext,
    result: IngestResult,
    samples: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[Op | None, str]:
    if problems := op_problems(op, deps.binary, deps.mcp):  # deterministic, before anything runs
        return None, "; ".join(problems)
    fixed, outcome = await _ground(draft, op, cfg, result)
    if fixed is not None:
        return fixed, outcome
    prompt = _repair_prompt(op, outcome, helps, samples or {})
    repaired = await build_repairer(model).run(prompt, deps=deps)
    result.usage.add(repaired.usage)
    retry = repaired.output.model_copy(update={"name": op.name})
    if problems := op_problems(retry, deps.binary, deps.mcp):
        return None, "repair was unsafe: " + "; ".join(problems)
    fixed, second = await _ground(draft, retry, cfg, result)
    return (
        (fixed, f"repaired: {second}")
        if fixed is not None
        else (None, f"{outcome}; after repair: {second}")
    )


async def _ground(draft: Tool, op: Op, cfg: Config, result: IngestResult) -> tuple[Op | None, str]:
    """Run the op's example for real. On success return the op with `returns` and the example's
    expectations taken from the actual output, and keep that output for later stages."""
    example = op.examples[0]
    returns = await _shape(draft, op, example, cfg)
    if isinstance(returns, str):
        return None, returns
    grounded = op.model_copy(update={"returns": returns})
    real = await _run(draft, grounded, example, cfg)
    ok, detail = check(real, Example(params=example.params))
    if not ok:
        return None, detail
    result.records[op.name] = real.records[:5]
    candidates = [f for f in (returns.id_field, *returns.summary_fields) if f]
    filled = [f for f in dict.fromkeys(candidates) if _filled(real.records, f)]
    expect = Expect(min_results=1 if real.records else 0, has_fields=filled[:3])
    final = grounded.model_copy(
        update={"examples": [Example(params=example.params, expect=expect)]}
    )
    return final, f"{len(real.records)} results"


async def _shape(draft: Tool, op: Op, example: Example, cfg: Config) -> Returns | str:
    """`returns` inferred from what the tool printed, or the error if it would not run."""
    raw = await _run(draft, _probing(op), example, cfg)
    if raw.status == "error" and "not JSON" in (raw.error or ""):
        if lines := await _jsonl_shape(draft, op, example, cfg):
            return lines
        raw = await _run(draft, _probing(op, "text"), example, cfg)  # a plain-text command
    if not raw.ok:
        return f"{raw.status}: {raw.error}"
    doc = _document(raw)
    if not raw.records or _empty(doc):
        return op.returns  # nothing to infer from: keep the draft's guess (flagged as empty)
    if set(raw.records[0]) == {"text"}:
        return Returns(format="text")
    return infer_returns(doc, draft.name, prefer=op.returns) or Returns(format="text")


async def _jsonl_shape(draft: Tool, op: Op, example: Example, cfg: Config) -> Returns | None:
    """`returns` for a tool that prints one JSON object per line, if that is what it does."""
    lines = await _run(draft, _probing(op, "jsonl"), example, cfg)
    if not (lines.ok and lines.records):
        return None
    inferred = infer_returns(lines.records, draft.name, prefer=op.returns)
    update = {"format": "jsonl", "path": "$"}
    return inferred.model_copy(update=update) if inferred else Returns(format="jsonl")


def _empty(doc: Any) -> bool:
    """An empty list, or an envelope whose lists are all empty (`{"items": []}`)."""
    if isinstance(doc, list):
        return not doc
    if isinstance(doc, dict):
        lists = [v for v in doc.values() if isinstance(v, list)]
        return bool(lists) and not any(lists)
    return False


def _filled(records: list[dict[str, Any]], name: str) -> bool:
    return bool(records) and all(get_field(r, name) not in (None, "", []) for r in records)


def _probing(op: Op, fmt: str = "json") -> Op:
    """The op with returns that keep the whole document, so its shape can be inferred."""
    returns = Returns(format=fmt, path="$")  # type: ignore[arg-type]
    return op.model_copy(update={"returns": returns, "paginate": None})


async def _run(draft: Tool, op: Op, example: Example, cfg: Config) -> OpResult:
    tool = draft.model_copy(update={"ops": [op]})
    executor = Executor(Registry(tools={tool.name: tool}), cfg.tool_env(), cfg.spill_dir)
    return await executor.run(f"{tool.name}.{op.name}", example.params)


def _document(result: OpResult) -> Any:
    if not result.records:
        return None
    first = result.records[0]
    return first["value"] if set(first) == {"value"} else first


def _repair_prompt(
    op: Op, error: str, helps: dict[str, str], samples: dict[str, list[dict[str, Any]]]
) -> str:
    real = "\n".join(
        f"- {name}: {json.dumps(recs, default=str)[:800]}" for name, recs in samples.items()
    )
    found = f"\n\nReal output from ops that already work (use real ids from here):\n{real}"
    return (
        f"The op:\n{op.model_dump_json(indent=1)}\n\nIts example failed with:\n{error}\n\n"
        f"Help text:\n{_relevant_help(op, helps)[:12000]}{found if real else ''}"
    )


def _relevant_help(op: Op, helps: dict[str, str]) -> str:
    """Help for the op's own subcommand path, or all of it when none matches."""
    words = " ".join(w for w in op.argv[1:] if not w.startswith("-") and "{" not in w)
    relevant = {k: v for k, v in helps.items() if k and words.startswith(k)}
    return "\n".join(f"### {k} --help\n{v}" for k, v in (relevant or helps).items())


# --- stage 4: boundaries ----------------------------------------------------------------------


async def _boundaries(
    tool: Tool, registry: Registry, cfg: Config, model: Model | str, result: IngestResult
) -> Tool:
    if not registry.tools:
        (result.staged / "boundary.md").write_text(
            f"# Boundary for {tool.name}\n\nThe first tool in this registry: nothing to bound.\n"
        )
        return tool
    before = _hits(registry, cfg)
    run = await build_boundary(model).run(_boundary_prompt(tool, registry))
    result.usage.add(run.usage)
    bounded = tool.model_copy(
        update={
            "description": run.output.description,
            "when": run.output.when,
            "not_when": run.output.not_when,
        }
    )
    result.proposals = [p.model_dump() for p in run.output.proposals if p.tool in registry.tools]
    with_tool = Registry(tools={**registry.tools, tool.name: bounded}, recipes=registry.recipes)
    after = _hits(with_tool, cfg)
    result.steals = sorted(q for q, hit in before.items() if hit and not after.get(q, False))
    _write_boundary(bounded, result)
    return bounded


def _boundary_prompt(tool: Tool, registry: Registry) -> str:
    others = "\n".join(
        f"- {t.name}: {t.description} | when: {t.when} | not_when: {t.not_when}"
        for t in registry.tools.values()
    )
    ops = "\n".join(f"- {tool.name}.{op.name}: {op.when}" for op in tool.ops)
    return (
        f"New tool {tool.name}: {tool.description}\nIts ops:\n{ops}\n"
        f"Draft when: {tool.when}\nDraft not_when: {tool.not_when}\n\nExisting tools:\n{others}"
    )


def _write_boundary(bounded: Tool, result: IngestResult) -> None:
    lines = [
        f"# Boundary for {bounded.name}",
        "",
        f"when: {bounded.when}",
        f"not_when: {bounded.not_when}",
        "",
    ]
    lines.append("## Ranker cases this tool now steals")
    lines += [f"- {q}" for q in result.steals] or ["- none"]
    lines += ["", "## Proposed edits to existing tools (not applied)"]
    lines += [
        f"- {p['tool']}: when={p['when']!r} not_when={p['not_when']!r} ({p['why']})"
        for p in result.proposals
    ] or ["- none"]
    (result.staged / "boundary.md").write_text("\n".join(lines) + "\n")


def _hits(registry: Registry, cfg: Config) -> dict[str, bool]:
    report = run_ranker_eval(registry, ranker_files(cfg.home))
    return {c.query: c.top3 for c in report.cases}


# --- stage 5: eval candidates -------------------------------------------------------------------


async def _candidates(
    tool: Tool, registry: Registry, cfg: Config, model: Model | str, result: IngestResult
) -> None:
    samples = []
    for op in tool.ops:
        sample = json.dumps(result.records.get(op.name, [])[:2], default=str)[:600]
        samples.append(f"- {tool.name}.{op.name}: {op.when}\n  sample output: {sample}")
    run = await build_evals(model).run("New tool ops and real output:\n" + "\n".join(samples))
    result.usage.add(run.usage)
    cases = [{"query": c.query, "expect": [tool.name], "op": c.op} for c in run.output.cases]
    result.candidates = len(cases)
    extended = Registry(tools={**registry.tools, tool.name: tool}, recipes=registry.recipes)
    path = result.staged / "eval_candidates.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "cases": cases}, sort_keys=False))
    report = run_ranker_eval(extended, [path])
    result.candidate_hit_rate = report.rate("top3") if report.cases else None


# --- stage 6: review ---------------------------------------------------------------------------


def _write_yaml(path: Path, tool: Tool) -> None:
    data = tool.model_dump(mode="json", exclude_defaults=True)
    data.setdefault("name", tool.name)
    path.write_text(yaml.safe_dump(data, sort_keys=False, width=100))


def _write_report(result: IngestResult) -> None:
    lines = [f"# oracle tool add: {result.name}", ""]
    for o in result.outcomes:
        flag = " (returned nothing: review its example; output may go to stderr)" if o.empty else ""
        lines.append(f"- {'kept' if o.kept else 'DROPPED'} {o.name}: {o.detail}{flag}")
    if result.refused:
        lines.append(f"- never probed or exposed (mutating): {', '.join(result.refused)}")
    lines.append(f"- ranker cases stolen from other tools: {len(result.steals)}")
    lines.append(f"- edits proposed to other tools (not applied): {len(result.proposals)}")
    rate = "n/a" if result.candidate_hit_rate is None else f"{result.candidate_hit_rate:.0%}"
    lines.append(f"- eval candidates: {result.candidates} (ranker top-3 on them: {rate})")
    lines.append(f"- model calls: {result.usage.model_calls}")
    lines += ["", "Files: probe.json, draft.yaml, manifest.yaml, boundary.md, eval_candidates.yaml"]
    verdict = "ready" if result.ready else "needs review"
    lines += ["", f"{verdict}: oracle tool activate {result.name}"]
    (result.staged / "report.md").write_text("\n".join(lines) + "\n")


def activate(cfg: Config, name: str) -> Path:
    """Copy the staged manifest into the active registry. Never overwrites an existing one."""
    source = staging_dir(cfg, name) / "manifest.yaml"
    if not source.is_file():
        raise IngestError(f"nothing staged for {name}; run `oracle tool add` first")
    target = cfg.registry_dir / f"{name}.yaml"
    if target.exists():
        raise IngestError(f"{target} exists; remove it first to replace it")
    try:  # a hand-edited staged manifest gets the same checks the loader will apply
        tool = Tool.model_validate(yaml.safe_load(source.read_text()))
    except (yaml.YAMLError, ValueError) as exc:
        raise IngestError(f"staged manifest is not valid: {exc}") from None
    if tool.name != name:
        raise IngestError(f"staged manifest is named {tool.name!r}, not {name!r}")
    if problem := refusal(tool):
        raise IngestError(f"the staged manifest is not read-only: {problem}")
    cfg.registry_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(source.read_text())
    return target
