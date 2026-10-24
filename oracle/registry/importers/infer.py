"""Infer a manifest's `returns` block from real JSON output: where the records are, which field
identifies them, which fields summarise them, and the pointer. Ingestion uses this so `returns`
describes what the tool actually printed, not what a model expected it to print.
"""

import re
from typing import Any

from oracle.registry.jsonpath import get_field
from oracle.registry.schema import Returns

ID_NAMES = (
    "id", "key", "uid", "uuid", "_id", "symbol_id", "chunk_id", "metadata.uid", "name",
    "metadata.name",
)  # fmt: skip
SUMMARY_NAMES = (
    "title", "name", "summary", "subject", "display_name", "status", "kind", "message", "text",
    "path", "line", "start_line", "from", "updated", "created", "start", "author", "email",
    "metadata.name", "metadata.namespace", "status.phase", "status.sync.status",
    "status.health.status", "status.operationState.finishedAt", "reason",
    "_source.message", "_source.level", "_source.service", "@timestamp", "_source.@timestamp",
)  # fmt: skip
MAX_SUMMARY = 4


def record_lists(doc: Any, path: str = "$", depth: int = 0) -> list[tuple[str, list[dict]]]:
    """Every list of objects in the document, with its JSONPath, outermost first."""
    found: list[tuple[str, list[dict]]] = []
    if isinstance(doc, list) and doc and all(isinstance(x, dict) for x in doc):
        found.append((f"{path}[*]", doc))
    elif isinstance(doc, dict) and depth < 3:
        for key, value in doc.items():
            if key.isidentifier():
                found += record_lists(value, f"{path}.{key}", depth + 1)
    return found


def infer_returns(doc: Any, tool: str, prefer: Returns | None = None) -> Returns | None:
    """None when the output holds no objects at all."""
    lists = record_lists(doc)
    if lists:
        path, records = max(lists, key=lambda pr: len(pr[1]))
    elif isinstance(doc, dict):
        path, records = "$", [doc]
    else:
        return None
    id_field = _id_field(records)
    summary = _summary(records, id_field)
    pointer = _pointer(records, tool, id_field)
    keep = prefer.pointer if prefer and prefer.pointer and _fills(prefer.pointer, records) else None
    return Returns(
        format="json",
        path=path,
        id_field=id_field,
        summary_fields=summary,
        pointer=keep or pointer,
    )


def flat(record: dict, depth: int = 3, prefix: str = "") -> dict[str, Any]:
    """Scalar fields as dotted names, up to `depth` levels (`status.sync.status` in Argo CD)."""
    out: dict[str, Any] = {}
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and depth > 1:
            out |= flat(value, depth - 1, f"{name}.")
        elif not isinstance(value, dict | list):
            out[name] = value
    return out


def _id_field(records: list[dict]) -> str | None:
    records = [flat(r) for r in records]
    candidates = _id_candidates(records)
    return next((name for name in candidates if _unique([r[name] for r in records])), None)


def _id_candidates(records: list[dict]) -> list[str]:
    """Fields every record has, known id names first, then `*_id` / `*Id`."""
    common = set(records[0]).intersection(*records[1:]) if records else set()
    known = [n for n in ID_NAMES if n in common]
    return known + sorted(k for k in common if k.endswith(("_id", "Id")) and k not in known)


def _unique(values: list[Any]) -> bool:
    """Non-empty scalars, all different: usable as an id."""
    scalars = all(isinstance(v, str | int) and v != "" for v in values)
    return scalars and len(set(map(str, values))) == len(values)


def _short_scalars(record: dict, skip: str | None) -> list[str]:
    return [
        k
        for k, v in record.items()
        if isinstance(v, str | int | float) and k != skip and len(str(v)) <= 200
    ]


def _summary(records: list[dict], id_field: str | None) -> list[str]:
    """Short scalar fields that every record fills (an empty title in one record is not a
    summary field)."""
    flats = [flat(r) for r in records[:20]]
    short = [
        k
        for k in _short_scalars(flats[0], id_field)
        if all(f.get(k) not in (None, "") for f in flats)
    ]
    ranked = sorted(short, key=lambda k: SUMMARY_NAMES.index(k) if k in SUMMARY_NAMES else 99)
    # a human key (PLAT-412) or a name helps a reader; an opaque id (a hash, a uuid) is noise
    readable = id_field is not None and (id_field == "key" or id_field.endswith("name"))
    return ([id_field] if id_field and readable else []) + ranked[:MAX_SUMMARY]


def _pointer(records: list[dict], tool: str, id_field: str | None) -> str | None:
    first = records[0]
    if "path" in first and ("line" in first or "start_line" in first):
        return "code:{path}:{" + ("line" if "line" in first else "start_line") + "}"
    return f"{tool}:{{{id_field}}}" if id_field else None


def _fills(template: str, records: list[dict]) -> bool:
    return all(
        all(get_field(r, f) not in (None, "") for f in re.findall(r"\{([^{}]+)\}", template))
        for r in records
    )
