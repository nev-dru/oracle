"""The output contract: Findings, Coverage, Refine, Trace, as markdown or JSON.

Coverage always names what was not searched, what returned nothing and what is not available
here, so the main agent never assumes completeness it does not have. With breadth, findings are
grouped by source.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from oracle.exec.dag import StepRun
from oracle.exec.executor import OpResult
from oracle.registry.loader import Registry

MAX_FINDINGS = 15
FIELD_CHARS = 140


@dataclass
class Finding:
    text: str
    pointer: str
    source: str  # the op ref that produced it


@dataclass
class Answer:
    call_id: str
    session: str
    level: str  # the level that ran
    findings: list[Finding]
    searched: list[str]
    not_searched: list[str]
    unavailable: list[str]
    failures: list[str]
    confidence: str
    gaps: list[str]
    trace: list[str]
    notes: list[str] = field(default_factory=list)
    no_match: list[str] = field(default_factory=list)
    refine: list[str] = field(default_factory=list)
    breadth: bool = False
    ops_run: list[str] = field(default_factory=list)
    recipes: list[str] = field(default_factory=list)
    time_window: str | None = None
    replans: int = 0
    shortlist: list[str] = field(default_factory=list)
    scored: list[str] = field(default_factory=list)
    """Shortlisted tools the ranker actually matched (not padding)."""
    skipped: list[str] = field(default_factory=list)
    step_results: list[dict[str, Any]] = field(default_factory=list)
    """One row per step that was attempted: step, op, status, n results. For evals and logs."""
    tokens_in: int = 0
    tokens_out: int = 0
    model_calls: int = 0
    wall_ms: int = 0


def findings_from(runs: list[StepRun], registry: Registry) -> list[Finding]:
    """Deterministic findings: each record's summary fields (quick tier, or distill fallback)."""
    out: list[Finding] = []
    for run in runs:
        r = run.result
        if not r.ok:
            continue
        fields = registry.resolve(r.op)[1].returns.summary_fields
        for record, pointer in zip(r.records, r.pointers, strict=True):
            out.append(Finding(summarise(record, fields, pointer), pointer, r.op))
    return out


def summarise(record: dict[str, Any], fields: list[str], pointer: str = "") -> str:
    """Summary field values, minus any the pointer already spells out (paths, line numbers):
    main-context tokens are the expensive ones."""
    keys = fields or list(record)[:4]
    values = [str(record[k]) for k in keys if record.get(k) not in (None, "", [])]
    parts = [_clip(v) for v in values if not _spelled_by(v, pointer)]
    return " · ".join(parts) if parts else _clip(json.dumps(record, default=str))


def _spelled_by(value: str, pointer: str) -> bool:
    """True when the pointer carries the value as a whole component: the path or line in
    `code:/r/a.py:11`, or the id in `teams:19:msg-5501`. A substring is not enough: hop 1 is
    not spelled by line 12."""
    return value in pointer.split(":") or pointer.endswith(":" + value)


def _clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= FIELD_CHARS else text[: FIELD_CHARS - 1] + "…"


def trace_line(run: StepRun) -> str:
    r = run.result
    args = "".join(f" {k}={_short(v)}" for k, v in r.params.items() if v is not None)
    if run.same_as:
        return f"{run.step_id} {r.op}{args} → same as {run.same_as}, reused"
    outcome = f"{len(r.records)} results" if r.ok else f"{r.status}: {r.error}"
    extra = f", {r.pages} pages" if r.pages > 1 else ""
    extra += ", truncated" if r.truncated else ""
    timing = f", {r.ms} ms" if r.status not in ("skipped", "unavailable", "invalid") else ""
    return f"{run.step_id} {r.op}{args} → {outcome}{timing}{extra}"


def _short(value: Any) -> str:
    text = json.dumps(value, default=str)
    return text if len(text) <= 80 else text[:79] + "…"


def tool_of(result: OpResult | str) -> str:
    ref = result if isinstance(result, str) else result.op
    return ref.split(".", 1)[0]


def to_markdown(a: Answer) -> str:
    lines = ["## Findings", *_findings_md(a), "", "## Coverage", *_coverage_md(a)]
    if a.refine:
        lines += ["", "## Refine", *a.refine]
    lines += ["", "## Trace", *a.trace, *a.notes]
    lines.append(f"call_id {a.call_id}  session {a.session}  level {a.level}")
    return "\n".join(lines)


def _findings_md(a: Answer) -> list[str]:
    shown = a.findings[:MAX_FINDINGS]
    if not shown:
        return ["- (no results)"]
    lines: list[str] = []
    if a.breadth:
        for source in dict.fromkeys(tool_of(f.source) for f in shown):
            lines.append(f"### {source}")
            lines += [f"- {f.text} — `{f.pointer}`" for f in shown if tool_of(f.source) == source]
    else:
        lines += [f"- {f.text} — `{f.pointer}`" for f in shown]
    if len(a.findings) > len(shown):
        lines.append(f"- … {len(a.findings) - len(shown)} more; narrow the question or `--scope`")
    return lines


def _coverage_md(a: Answer) -> list[str]:
    lines = [
        f"Searched: {', '.join(a.searched) or 'nothing'}",
        f"Not searched: {', '.join(a.not_searched) or 'nothing'}",
    ]
    if a.no_match:
        lines.append(f"No match: {', '.join(a.no_match)}")
    if a.unavailable:
        lines.append(f"Not available here: {', '.join(a.unavailable)}")
    if a.skipped:
        lines.append(f"Skipped: {'; '.join(a.skipped)}")
    lines += [f"Failed: {f}" for f in a.failures]
    lines.append(f"Confidence: {a.confidence}   Gaps: {'; '.join(a.gaps) or 'none found'}")
    return lines


def to_json(a: Answer) -> str:
    """The same contract as markdown, for the MCP wrapper; run details sit under `meta`."""
    meta = ("ops_run", "recipes", "replans", "shortlist", "scored", "tokens_in", "tokens_out",
            "model_calls", "wall_ms", "breadth", "time_window", "step_results")  # fmt: skip
    data = {
        "call_id": a.call_id,
        "session": a.session,
        "level": a.level,
        "findings": [asdict(f) for f in a.findings[:MAX_FINDINGS]],
        "more_findings": max(0, len(a.findings) - MAX_FINDINGS),
        "coverage": {
            "searched": a.searched,
            "not_searched": a.not_searched,
            "no_match": a.no_match,
            "unavailable": a.unavailable,
            "skipped": a.skipped,
            "failures": a.failures,
            "confidence": a.confidence,
            "gaps": a.gaps,
        },
        "refine": a.refine,
        "trace": a.trace + a.notes,
        "meta": {k: getattr(a, k) for k in meta},
    }
    return json.dumps(data, indent=2)
