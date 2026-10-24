"""Stable pointers: short ids the main agent can hand back to `oracle expand`."""

import hashlib
import json
from pathlib import Path
from typing import Any

from oracle.exec.redact import redact_text
from oracle.registry.schema import PLACEHOLDER, Returns


def pointer_for(record: dict[str, Any], returns: Returns, tool: str) -> str:
    """Fill the manifest's pointer template, e.g. `code:{path}:{line}`.

    A record missing a template field falls back to `<tool>:<id>`, then to a content hash,
    so every record still gets a pointer that `expand` can resolve within the session.
    """
    if returns.pointer:
        fields = PLACEHOLDER.findall(returns.pointer)
        if all(record.get(f) not in (None, "") for f in fields):
            return PLACEHOLDER.sub(lambda m: str(record[m.group(1)]), returns.pointer)
    if returns.id_field and record.get(returns.id_field) not in (None, ""):
        return f"{tool}:{record[returns.id_field]}"
    digest = hashlib.sha1(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()
    return f"{tool}:#{digest[:12]}"


def record_id(record: dict[str, Any], returns: Returns) -> str | None:
    if returns.id_field is None:
        return None
    value = record.get(returns.id_field)
    return None if value in (None, "") else str(value)


SPILL_HEAD_CHARS = 4000


class SpillError(ValueError):
    pass


def expand_spill(pointer: str, spill_dir: Path) -> dict[str, Any]:
    """`file:<path>` pointers name oversized outputs; only files under the spill dir resolve."""
    path = Path(pointer.removeprefix("file:")).resolve()
    if not path.is_relative_to(spill_dir.resolve()) or not path.is_file():
        raise SpillError(f"{pointer} is not a spill file under {spill_dir}")
    raw = path.read_bytes()
    head = raw[: SPILL_HEAD_CHARS * 4].decode("utf-8", "replace")[:SPILL_HEAD_CHARS]
    head = redact_text(head)  # spill files hold raw stdout; records were redacted, this was not
    return {
        "pointer": pointer,
        "path": str(path),
        "bytes": len(raw),
        "head": head,
        "truncated": len(head.encode()) < len(raw),
    }
