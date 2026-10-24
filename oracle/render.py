"""The output contract: Findings, Coverage, (Refine from M2), Trace, as markdown or JSON.

At the quick tier there is no distill step, so findings are the summary fields of each record,
deterministically. Coverage always names what was not searched, so the main agent never
assumes completeness it does not have.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from oracle.exec.executor import OpResult
from oracle.registry.loader import Registry

MAX_FINDINGS = 15
FIELD_CHARS = 140


@dataclass
class Finding:
    text: str
    pointer: str
    source: str


@dataclass
class Answer:
    call_id: str
    session: str
    level: str
    findings: list[Finding]
    searched: list[str]
    not_searched: list[str]
    unavailable: list[str]
    failures: list[str]
    confidence: str
    gaps: list[str]
    trace: list[str]
    notes: list[str] = field(default_factory=list)


def findings_from(results: list[OpResult], registry: Registry) -> list[Finding]:
    out: list[Finding] = []
    for r in results:
        _, op = registry.resolve(r.op)
        for record, pointer in zip(r.records, r.pointers, strict=True):
            text = summarise(record, op.returns.summary_fields, pointer)
            out.append(Finding(text, pointer, r.op))
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


def trace_line(step_id: str, r: OpResult) -> str:
    args = " ".join(f"{k}={json.dumps(v)}" for k, v in r.params.items() if v is not None)
    outcome = f"{len(r.records)} results" if r.ok else f"{r.status}: {r.error}"
    extra = f", {r.pages} pages" if r.pages > 1 else ""
    extra += ", truncated" if r.truncated else ""
    return f"{step_id} {r.op} {args} → {outcome}, {r.ms} ms{extra}"


def to_markdown(a: Answer) -> str:
    lines = ["## Findings"]
    shown = a.findings[:MAX_FINDINGS]
    lines += [f"- {f.text} — `{f.pointer}`" for f in shown] or ["- (no results)"]
    if len(a.findings) > len(shown):
        lines.append(f"- … {len(a.findings) - len(shown)} more; narrow the question or `--scope`")
    lines += ["", "## Coverage", f"Searched: {', '.join(a.searched) or 'nothing'}"]
    lines.append(f"Not searched: {', '.join(a.not_searched) or 'nothing'}")
    if a.unavailable:
        lines.append(f"Not available here: {', '.join(a.unavailable)}")
    lines += [f"Failed: {f}" for f in a.failures]
    lines.append(f"Confidence: {a.confidence}   Gaps: {'; '.join(a.gaps) or 'none found'}")
    lines += ["", "## Trace", *a.trace, *a.notes, f"call_id {a.call_id}  session {a.session}"]
    return "\n".join(lines)


def to_json(a: Answer) -> str:
    return json.dumps(
        {
            "call_id": a.call_id,
            "session": a.session,
            "level": a.level,
            "findings": [vars(f) for f in a.findings[:MAX_FINDINGS]],
            "more_findings": max(0, len(a.findings) - MAX_FINDINGS),
            "coverage": {
                "searched": a.searched,
                "not_searched": a.not_searched,
                "unavailable": a.unavailable,
                "failures": a.failures,
                "confidence": a.confidence,
                "gaps": a.gaps,
            },
            "trace": a.trace + a.notes,
        },
        indent=2,
    )
