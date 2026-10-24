"""Step bindings: a param whose value is `$s1.ids`, `$s1.pointers` or `$s1.results[0].path`
takes it from an earlier step's results; `+` concatenates lists (`$s1.ids + $s2.ids`).

Bindings are whole values. A value that only mentions a binding inside other text is a literal.
"""

import re
from collections.abc import Callable
from typing import Any

from oracle.exec.executor import OpResult
from oracle.registry import jsonpath

TERM = re.compile(
    r"^\$(?P<step>[A-Za-z0-9_/-]+)\.(?P<kind>ids|pointers|results)"
    r"(?P<path>(?:\.[A-Za-z_][\w-]*|\[(?:\d+|\*|'[^']*')\])*)$"
)
GRAMMAR = "$<step>.ids, $<step>.pointers or $<step>.results[0].<field>, joined with ' + '"


class BindingError(ValueError):
    pass


def is_binding(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("$")


def terms(value: str) -> list[re.Match[str]]:
    out = []
    for raw in value.split(" + "):
        m = TERM.match(raw.strip())
        if not m:
            raise BindingError(f"bad binding {raw.strip()!r}; use {GRAMMAR}")
        out.append(m)
    return out


def binding_refs(params: dict[str, Any]) -> set[str]:
    """Step ids a step's params bind to."""
    return {m["step"] for v in params.values() if is_binding(v) for m in terms(v)}


def resolve(value: str, results_of: Callable[[str], list[OpResult]]) -> Any:
    """The bound value. A single `results[...]` term without `[*]` yields one value (or None);
    everything else yields a list."""
    found = terms(value)
    single = len(found) == 1 and found[0]["kind"] == "results" and found[0]["path"]
    if single and "[*]" not in found[0]["path"]:
        hits = _values(found[0], results_of(found[0]["step"]))
        return hits[0] if hits else None
    return [v for m in found for v in _values(m, results_of(m["step"]))]


def _values(m: re.Match[str], results: list[OpResult]) -> list[Any]:
    if m["kind"] == "ids":
        return [i for r in results for i in r.ids]
    if m["kind"] == "pointers":
        return [p for r in results for p in r.pointers]
    records = [rec for r in results for rec in r.records]
    return jsonpath.find(records, "$" + m["path"])


def fill_template(value: Any, values: dict[str, Any], prefix: str, stages: set[str]) -> Any:
    """A recipe stage's param: `{name}` from the recipe's params and builtins, and bindings to
    sibling stages rewritten to their run ids (`$search.ids` -> `$s1/search.ids`)."""
    if not isinstance(value, str):
        return value
    if is_binding(value):
        return re.sub(
            r"\$([a-z][a-z0-9_]*)\.",
            lambda m: f"${prefix}/{m[1]}." if m[1] in stages else m[0],
            value,
        )
    whole = re.fullmatch(r"\{([a-z_]+)\}", value)
    if whole and whole[1] in values:
        return values[whole[1]]  # keeps the type: a list stays a list
    return re.sub(
        r"\{([a-z_]+)\}", lambda m: str(values.get(m[1], m[0])) if m[1] in values else m[0], value
    )
