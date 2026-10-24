"""Turn a tool's stdout into records according to the manifest's `returns` block."""

import json
import re
from typing import Any

from oracle.exec import jsonpath
from oracle.registry.schema import Returns

TEXT_RECORD_CHARS = 8000


class ParseError(ValueError):
    pass


def parse_output(text: str, returns: Returns) -> tuple[Any, list[dict[str, Any]]]:
    """(document, records). The document is what pagination's `next` path reads."""
    return _PARSERS[returns.format](text, returns.path)


def _json(text: str, path: str) -> tuple[Any, list[dict[str, Any]]]:
    doc = _loads(text)
    return doc, [_as_record(r) for r in jsonpath.find(doc, path)]


def _jsonl(text: str, path: str) -> tuple[Any, list[dict[str, Any]]]:
    rows = [_loads(line) for line in text.splitlines() if line.strip()]
    if path != "$":
        rows = [hit for row in rows for hit in jsonpath.find(row, path)]
    return rows, [_as_record(r) for r in rows]


def _table(text: str, _path: str) -> tuple[Any, list[dict[str, Any]]]:
    rows = parse_table(text)
    return rows, rows


def _text(text: str, _path: str) -> tuple[Any, list[dict[str, Any]]]:
    body = text.strip()
    return body, ([{"text": body[:TEXT_RECORD_CHARS]}] if body else [])


_PARSERS = {"json": _json, "jsonl": _jsonl, "table": _table, "text": _text}


def parse_table(text: str) -> list[dict[str, Any]]:
    """Columns separated by two or more spaces (kubectl-style); the first line is the header."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    header = [h.strip().lower().replace(" ", "_") for h in re.split(r"\s{2,}", lines[0].strip())]
    rows = []
    for line in lines[1:]:
        cells = re.split(r"\s{2,}", line.strip(), maxsplit=len(header) - 1)
        rows.append(dict(zip(header, cells + [""] * (len(header) - len(cells)), strict=True)))
    return rows


def _loads(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParseError(
            f"output is not JSON ({exc.msg} at char {exc.pos}): {text[:120]!r}"
        ) from None


def _as_record(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {"value": value}
