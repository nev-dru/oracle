"""`oracle dream`, the first slice: turn crawl chains that repeat in the log into recipes.

A chain is the path a crawl took from hop to hop: step k searched a term that appeared, as the
exact value of one field, in the records of step k-1 (the runbook's `author` became
`people.lookup(name)`). A chain seen in `min_calls` or more successful calls is drafted as a
recipe whose stages bind to that field, validated by the loader against a scratch copy of the
recipes, and staged in ~/.oracle/dreams/<date>/ with a report. Dreaming proposes; a person
accepts (`oracle dream --accept <name>`). Nothing is applied or committed automatically, and
no session transcripts are read.
"""

import json
import re
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiosqlite
import yaml

from oracle.log.store import scalar_fields
from oracle.registry.loader import Registry, load_registry
from oracle.registry.schema import RECIPE_NAME

GOOD = ("medium", "high")


@dataclass(frozen=True)
class Link:
    op: str
    param: str | None = None  # the param the previous stage's field fed
    source: str | None = None  # that field, in the previous stage's records


@dataclass
class Chain:
    links: tuple[Link, ...]
    calls: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    first_params: list[dict[str, Any]] = field(default_factory=list)
    first_terms: list[str | None] = field(default_factory=list)
    """The term each call's first stage looked up (None for a step of the first plan)."""


@dataclass
class _Step:
    step: str
    op: str
    params: dict[str, Any]
    hop: int
    parent: str | None
    records: list[dict[str, Any]]


async def find_chains(db: aiosqlite.Connection, since: float, min_calls: int = 3) -> list[Chain]:
    chains: dict[tuple[Link, ...], Chain] = {}
    for call_id, question in await _good_calls(db, since):
        steps = await _steps(db, call_id)
        for links, first, term in _paths(steps):
            chain = chains.setdefault(links, Chain(links))
            if call_id not in chain.calls:
                chain.calls.append(call_id)
                chain.questions.append(question)
                chain.first_params.append(first)
                chain.first_terms.append(term)
    return [c for c in chains.values() if len(c.calls) >= min_calls]


async def _good_calls(db: aiosqlite.Connection, since: float) -> list[tuple[str, str]]:
    sql = "SELECT call_id, query FROM calls WHERE ts >= ? AND confidence IN (?, ?) ORDER BY ts"
    async with db.execute(sql, (since, *GOOD)) as cur:
        return [(r[0], r[1]) for r in await cur.fetchall()]


async def _steps(db: aiosqlite.Connection, call_id: str) -> list[_Step]:
    """Each step with its own records: the op row's `fields`, or for logs written before that
    column, the pointers table (per op, and only what later calls in the session left)."""
    sql = (
        "SELECT step_id, op, params, hop, parent, fields FROM ops WHERE call_id = ? AND ok = 1 "
        "AND n_results > 0 AND hop IS NOT NULL ORDER BY hop"
    )
    async with db.execute(sql, (call_id,)) as cur:
        rows = await cur.fetchall()
    legacy = await _records_by_op(db, call_id) if any(r[5] is None for r in rows) else {}
    return [
        _Step(r[0], r[1], json.loads(r[2] or "{}"), r[3] or 0, r[4],
              json.loads(r[5]) if r[5] else legacy.get(r[1], []))
        for r in rows
    ]  # fmt: skip


async def _records_by_op(db: aiosqlite.Connection, call_id: str) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    async with db.execute("SELECT op, record FROM pointers WHERE call_id = ?", (call_id,)) as cur:
        for op, record in await cur.fetchall():
            records[op].append(json.loads(record))
    return records


Walk = tuple[tuple[Link, ...], dict[str, Any], str | None]


def _paths(steps: list[_Step]) -> list[Walk]:
    """Every maximal path of two or more bound links, with its first stage's params and the
    term that stage looked up. A path may
    start at any hop: the crawl's first link is often found by reading prose (a ticket key in a
    message), which a recipe cannot do, while later links bind to fields."""
    by_hop: dict[int, list[_Step]] = defaultdict(list)
    for s in steps:
        by_hop[s.hop].append(s)
    fed = {id(n) for s in steps for n in by_hop.get(s.hop + 1, []) if _binding(s.records, n)}
    paths: list[Walk] = []
    for start in (s for s in steps if id(s) not in fed):
        _extend(by_hop, start, (Link(start.op),), start, paths)
    return paths


def _extend(
    by_hop: dict[int, list[_Step]],
    current: _Step,
    links: tuple[Link, ...],
    first: _Step,
    out: list[Walk],
) -> None:
    grew = False
    for nxt in by_hop.get(current.hop + 1, []):
        bound = _binding(current.records, nxt)
        if bound:
            grew = True
            _extend(by_hop, nxt, (*links, Link(nxt.op, *bound)), first, out)
    if not grew and len(links) >= 2:
        out.append((links, first.params, first.parent))


def _binding(records: list[dict[str, Any]], nxt: _Step) -> tuple[str, str] | None:
    """(param of `nxt`, field of `records`) when the term `nxt` searched is exactly a field
    value of the previous stage's first result: a recipe binds `results[0]`, so a term taken
    from a later result is not a link it can replay."""
    if not nxt.parent:
        return None
    term = nxt.parent.strip().lower()
    param = next((k for k, v in nxt.params.items() if str(v).strip().lower() == term), None)
    if param is None:
        return None
    for name, value in scalar_fields(records[0]).items() if records else ():
        if str(value).strip().lower() == term:
            return param, name
    return None


def draft_recipe(chain: Chain, index: int, registry: Registry | None = None) -> dict[str, Any]:
    """Stage 1's params that varied across the calls become the recipe's params; each later
    stage binds to the field that fed it. `when` says what the stages do (from the ops' own
    `when`), since that is the text the ranker and the planners match."""
    varying, stage1 = _first_stage(chain)
    steps = [{"id": "s1", "op": chain.links[0].op, "params": stage1}]
    for i, link in enumerate(chain.links[1:], start=2):
        binding = f"$s{i - 1}.results[0].{link.source}"
        steps.append({"id": f"s{i}", "op": link.op, "params": {link.param: binding}})
    return {
        "name": f"learned.chain_{index}",
        "when": _purpose(chain, registry),
        "params": {k: {"type": "str", "required": True} for k in sorted(varying)},
        "steps": steps,
    }


def _purpose(chain: Chain, registry: Registry | None) -> str:
    """'<what stage 1 does> Then: <what stage 2 does> (its `name` from the previous stage's
    `author`). Learned from 3 crawls, e.g. "<a question>".'"""
    parts = [_op_when(chain.links[0].op, registry)]
    for link in chain.links[1:]:
        parts.append(
            f"Then: {_op_when(link.op, registry)} (its `{link.param}` from `{link.source}`)"
        )
    example = chain.questions[0][:160] if chain.questions else ""
    parts.append(f'Learned from {len(chain.calls)} crawls, e.g. "{example}".')
    return " ".join(parts)


def _op_when(ref: str, registry: Registry | None) -> str:
    try:
        when = registry.resolve(ref)[1].when.strip() if registry else ref
    except KeyError:
        when = ref
    return when if when.endswith(".") else when + "."


def _first_stage(chain: Chain) -> tuple[set[str], dict[str, Any]]:
    """(the recipe's params, stage 1's params: a template for those, the literal for the rest).
    A param is the recipe's when it varied across the calls or carried the term the crawl
    looked up. When none did (the same question every time), every text param is: a recipe
    that always searches one literal answers one question."""
    keys = set().union(*(p.keys() for p in chain.first_params))
    varying = _free(chain, keys)
    first = chain.first_params[0]
    params = {k: f"{{{k}}}" if k in varying else first.get(k) for k in sorted(keys)}
    return varying, {k: v for k, v in params.items() if v is not None}


def _free(chain: Chain, keys: set[str]) -> set[str]:
    seen = {k: {json.dumps(p.get(k)) for p in chain.first_params} for k in keys}
    varying = {k for k in keys if len(seen[k]) > 1} | _carried(chain)
    texts = {k for k in keys if all(isinstance(p.get(k), str) for p in chain.first_params)}
    return varying or texts


def _carried(chain: Chain) -> set[str]:
    """Stage-1 params whose value held the term that stage looked up."""
    out: set[str] = set()
    for params, term in zip(chain.first_params, chain.first_terms, strict=True):
        if term and term.strip():
            out |= {k for k, v in params.items() if term.strip().lower() in str(v).lower()}
    return out


@dataclass
class Dream:
    staged: Path
    proposals: list[tuple[str, str]] = field(default_factory=list)  # (name, verdict)
    known: list[str] = field(default_factory=list)
    """Chains seen again that are already recipes."""


async def dream(
    db: aiosqlite.Connection, registry_dir: Path, home: Path, days: float, min_calls: int
) -> Dream:
    since = time.time() - days * 86400
    out = Dream(staged=home / "dreams" / time.strftime("%Y-%m-%d"))
    found = await find_chains(db, since, min_calls)
    (out.staged / "recipes").mkdir(parents=True, exist_ok=True)
    registry = load_registry(registry_dir)
    known = {_shape(r.model_dump()["steps"]): name for name, r in registry.recipes.items()}
    chains = []
    for chain in found:
        recipe = draft_recipe(chain, _free_index(registry, out.proposals), registry)
        if (same := known.get(_shape(recipe["steps"]))) is not None:
            out.known.append(f"{_path(chain)}: already the recipe {same}")
            continue
        chains.append(chain)
        verdict = _validate(recipe, registry_dir)
        (out.staged / "recipes" / f"{recipe['name']}.yaml").write_text(
            yaml.safe_dump(recipe, sort_keys=False, width=100)
        )
        out.proposals.append((recipe["name"], verdict))
    _report(out, chains)
    return out


def _shape(steps: list[dict[str, Any]]) -> str:
    """A recipe's stages as data, to recognise a chain that is already a recipe."""
    return json.dumps([(s["op"], s.get("params", {})) for s in steps], sort_keys=True)


def _free_index(registry: Registry, proposals: list[tuple[str, str]]) -> int:
    """The next `learned.chain_<n>` no recipe and no proposal of this run uses."""
    taken = set(registry.recipes) | {name for name, _verdict in proposals}
    return next(i for i in range(1, len(taken) + 2) if f"learned.chain_{i}" not in taken)


def _path(chain: Chain) -> str:
    return " -> ".join(_describe(link) for link in chain.links)


def _validate(recipe: dict[str, Any], registry_dir: Path) -> str:
    """Load the registry with its recipes plus this one, in a scratch directory."""
    active = Path(registry_dir).parent / "recipes"
    with tempfile.TemporaryDirectory(prefix="oracle-dream-") as scratch:
        tmp = Path(scratch)
        for existing in active.glob("*.yaml") if active.is_dir() else []:
            (tmp / existing.name).write_text(existing.read_text())
        (tmp / f"{recipe['name']}.yaml").write_text(yaml.safe_dump(recipe))
        reg: Registry = load_registry(registry_dir, recipes=tmp)
    mine = [p for p in reg.problems if recipe["name"] in p]
    if recipe["name"] not in reg.recipes:
        return "rejected: " + ("; ".join(mine) or "did not load")
    return "valid: loads with the registry; run `oracle eval` after accepting"


def _report(out: Dream, chains: list[Chain]) -> None:
    lines = [f"# oracle dream {out.staged.name}", ""]
    if not chains:
        lines.append("No new chain repeated often enough to propose a recipe.")
    lines += [f"- {k}" for k in out.known]
    for (name, verdict), chain in zip(out.proposals, chains, strict=True):
        lines.append(f"- {name}: {_path(chain)}")
        lines.append(f"  seen in {len(chain.calls)} calls; {verdict}")
        lines.append(f"  accept: oracle dream --accept {name}")
    (out.staged / "report.md").write_text("\n".join(lines) + "\n")


def _describe(link: Link) -> str:
    return f"{link.op}({link.param}<-{link.source})" if link.param else link.op


def accept(home: Path, registry_dir: Path, name: str) -> Path:
    """Copy a staged recipe into the active recipes directory; never overwrite."""
    if not re.fullmatch(RECIPE_NAME, name):
        raise ValueError(f"{name!r} is not a recipe name (learned.chain_1, say)")
    staged = sorted((home / "dreams").glob(f"*/recipes/{name}.yaml"))
    if not staged:
        raise FileNotFoundError(f"no staged recipe {name}; run `oracle dream` first")
    target_dir = Path(registry_dir).parent / "recipes"
    target = target_dir / f"{name}.yaml"
    if target.exists():
        raise FileExistsError(f"{target} exists")
    if _validate(yaml.safe_load(staged[-1].read_text()), registry_dir).startswith("rejected"):
        raise ValueError(f"{name} no longer loads with this registry")
    target_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(staged[-1].read_text())
    return target
