"""Build an argv list from a manifest template. The executor never builds a shell string.

Grammar, one placeholder kind per argv item:
  {p}              the value of p, also inside a larger item ("--limit={limit}")
  {?p:--flag,q}    "--flag" then the value of q, only when p is set; q may be omitted for flags
  {?p:--flag=,q}   the same as one item, "--flag=<q>", for values that may start with "-"
  {stdin:p}        the item is removed and the value of p is fed on stdin
Lists render comma-joined, json params as compact JSON, bools as true/false.

A value that would stand alone as an argv item and starts with "-" is refused unless a literal
"--" came earlier: otherwise a param could smuggle in a flag the manifest never declared.
"""

import json
from typing import Any

from oracle.registry.schema import PLACEHOLDER


class ArgvError(ValueError):
    pass


def render(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ",".join(render(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def is_set(value: Any) -> bool:
    return value is not None and value is not False and value != [] and value != ""


def build_argv(template: list[str], params: dict[str, Any]) -> tuple[list[str], str | None]:
    argv: list[str] = []
    stdin: str | None = None
    for item in template:
        whole = PLACEHOLDER.fullmatch(item)
        body = whole.group(1) if whole else ""
        positional_ok = "--" in argv
        if body.startswith("?"):
            argv += _conditional(body[1:], params, positional_ok)
        elif body.startswith("stdin:"):
            stdin = render(_value(body.removeprefix("stdin:"), params))
        elif whole:
            argv.append(_standalone(render(_value(body, params)), body, positional_ok))
        else:
            argv.append(PLACEHOLDER.sub(lambda m: render(_value(m.group(1), params)), item))
    return argv, stdin


def _conditional(body: str, params: dict[str, Any], positional_ok: bool) -> list[str]:
    cond, _, rest = body.partition(":")
    if not is_set(params.get(cond)):
        return []
    flag, *names = rest.split(",")
    values = [render(_value(n, params)) for n in names if n]
    if flag.endswith("=") and len(values) == 1:
        return [flag + values[0]]
    return [flag, *(_standalone(v, cond, positional_ok) for v in values)]


def _standalone(value: str, name: str, positional_ok: bool) -> str:
    if value.startswith("-") and not positional_ok:
        raise ArgvError(
            f"param {name!r} value {value!r} starts with '-' and would be read as a flag; "
            "the manifest should use '--' before positionals or the --flag=value form"
        )
    return value


def _value(name: str, params: dict[str, Any]) -> Any:
    value = params.get(name)
    if value is None:
        raise ArgvError(f"param {name!r} is unset; optional params need the {{?{name}:...}} form")
    return value
