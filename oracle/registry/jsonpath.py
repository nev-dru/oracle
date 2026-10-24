"""The JSONPath subset manifests use: `$`, `.key`, `['key']`, `[n]` and `[*]`.

Hand-rolled because this is all the registry needs; a missing key yields no matches, not an error.
"""

import re
from typing import Any

TOKEN = re.compile(r"\.([A-Za-z_][\w-]*)|\[(\d+|\*|'[^']*'|\"[^\"]*\")\]")


class PathError(ValueError):
    pass


def parse(path: str) -> list[str | int]:
    if not path.startswith("$"):
        raise PathError(f"path must start with $: {path!r}")
    steps: list[str | int] = []
    pos = 1
    while pos < len(path):
        m = TOKEN.match(path, pos)
        if not m:
            raise PathError(f"cannot parse {path!r} at {path[pos:]!r}")
        key, index = m.groups()
        if key is not None:
            steps.append(key)
        elif index == "*":
            steps.append("*")
        elif index[0] in "'\"":
            steps.append(index[1:-1])
        else:
            steps.append(int(index))
        pos = m.end()
    return steps


def find(doc: Any, path: str) -> list[Any]:
    nodes = [doc]
    for step in parse(path):
        nodes = [child for node in nodes for child in _children(node, step)]
    return nodes


def first(doc: Any, path: str) -> Any:
    hits = find(doc, path)
    return hits[0] if hits else None


def _children(node: Any, step: str | int) -> list[Any]:
    if step == "*":
        if isinstance(node, list):
            return list(node)
        return list(node.values()) if isinstance(node, dict) else []
    if isinstance(step, int):
        return [node[step]] if isinstance(node, list) and -len(node) <= step < len(node) else []
    return [node[step]] if isinstance(node, dict) and step in node else []


def get_field(record: Any, name: str) -> Any:
    """A record field by name; `metadata.name` reaches into nested objects (kubectl, ES
    `_source`). An exact key wins over the dotted reading."""
    if isinstance(record, dict) and name in record:
        return record[name]
    node = record
    for part in name.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node
