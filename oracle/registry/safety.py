"""Deterministic read-only check on argv: refuse any op or probe whose subcommand is a mutating
verb. It runs at load, before an ingestion draft is ever executed, and before help is probed,
because a model-drafted manifest is untrusted until reviewed and some CLIs run a subcommand even
when `--help` follows it.
"""

import re
from collections.abc import Mapping
from typing import Any

MUTATING_VERBS = frozenset(
    [
        "add",
        "trigger",
        "reload",
        "rebuild",
        "notify",
        "flush",
        "clear",
        "annotate",
        "apply",
        "approve",
        "archive",
        "assign",
        "attach",
        "autoscale",
        "cancel",
        "close",
        "comment",
        "commit",
        "cordon",
        "cp",
        "drop",
        "create",
        "debug",
        "delete",
        "deploy",
        "destroy",
        "detach",
        "disable",
        "drain",
        "edit",
        "enable",
        "exec",
        "expose",
        "import",
        "index",
        "ingest",
        "insert",
        "install",
        "kill",
        "label",
        "login",
        "manage",
        "logout",
        "merge",
        "move",
        "new",
        "patch",
        "port-forward",
        "post",
        "promote",
        "proxy",
        "publish",
        "purge",
        "push",
        "put",
        "refresh",
        "reindex",
        "remove",
        "rename",
        "reopen",
        "replace",
        "reset",
        "restart",
        "restore",
        "resume",
        "revoke",
        "rm",
        "rollback",
        "rollout",
        "run",
        "scale",
        "send",
        "set",
        "start",
        "stop",
        "suspend",
        "sync",
        "taint",
        "terminate",
        "transition",
        "truncate",
        "uncordon",
        "uninstall",
        "unset",
        "update",
        "upgrade",
        "upload",
        "write",
    ]
)


def literal_words(argv: list[str]) -> list[str]:
    """Every literal word after the binary, wherever it sits, up to a `--` (after which come
    positional values such as a search query). Flags placed before a subcommand do not hide it:
    `kubectl -n prod delete pod` still shows `delete`."""
    words = []
    for item in argv[1:]:
        if item == "--":
            break
        if not item.startswith("-") and "{" not in item:
            words.append(item)
    return words


def first_positional(argv: list[str]) -> str | None:
    """The first item after the binary that is neither a flag nor a flag's value: the
    subcommand position (`kubectl {verb} ...` puts a param there)."""
    after_flag = False
    for item in argv[1:]:
        if item == "--":
            return None
        if item.startswith("-"):
            after_flag = "=" not in item
            continue
        if after_flag or item.startswith("{?"):
            after_flag = False
            continue
        return item
    return None


def _norm(word: str) -> str:
    """`deleteProject`, `issues.delete`, `delete_project` -> hyphenated lower-case parts."""
    word = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", word)
    return re.sub(r"[_./:]", "-", word.lower())


def mutating_word(words: list[str]) -> str | None:
    """The first word that is, or contains as a part, a mutating verb (`app sync`,
    `set-context`, `deleteProject`). For literal subcommands and tool names."""
    for word in words:
        norm = _norm(word)
        if norm in MUTATING_VERBS or any(p in MUTATING_VERBS for p in norm.split("-")):
            return word
    return None


def mcp_problem(tool_name: str) -> str | None:
    """MCP tool names: snake_case, camelCase or dotted verbs (`delete_project`, `createIssue`)."""
    word = mutating_word([tool_name])
    return f"MCP tool {tool_name!r} looks mutating; the oracle is read-only" if word else None


def op_problem(
    argv: list[str], mcp_tool: str | None, params: Mapping[str, Any] | None = None
) -> str | None:
    return mcp_problem(mcp_tool) if mcp_tool else argv_problem(argv, params)


def argv_problem(argv: list[str], params: Mapping[str, Any] | None = None) -> str | None:
    """Static check of a template, at load and before an ingestion draft runs. Values that only
    exist at run time are checked on the rendered argv (`rendered_problem`)."""
    if problem := _inline_code(argv):
        return problem
    word = mutating_word(literal_words(argv))
    if word:
        return f"subcommand {word!r} looks mutating; the oracle is read-only"
    for name, spec in (params or {}).items():
        enum = list(getattr(spec, "enum", None) or [])
        if (bad := _bare_verb(enum)) and f"{{{name}}}" in argv:
            return f"param {name!r} may be {bad!r}, a mutating verb"
    return None


def rendered_problem(argv: list[str]) -> str | None:
    """The command exactly as it will run. Catches what a template check cannot see: a param
    in any subcommand slot (`argocd app {action}` with action=sync) and a literal inside a
    conditional (`{?yes:delete}`). Only positional words count: a flag's value is data (code
    tools take symbol names such as `publish` or `update`), as are multi-word items and names
    such as `user-sync-worker`, which are not split.

    Known limit: a boolean flag followed by a verb param (`kubectl -A {verb}`) reads like a
    flag and its value, so the verb is not seen; the static check and review cover that shape.
    """
    if problem := _inline_code(argv):
        return problem
    words, after_flag = [], False
    for item in argv[1:]:
        if item == "--":
            break
        if item.startswith("-"):
            after_flag = "=" not in item
            continue
        if not after_flag and len(item.split()) == 1:
            words.append(item)
        after_flag = False
    if bad := _bare_verb(words):
        return f"{bad!r} would run as a subcommand; the oracle is read-only"
    return None


def _bare_verb(words: list[str]) -> str | None:
    return next((w for w in words if _norm(w) in MUTATING_VERBS), None)


SHELLS = frozenset(
    [
        "sh",
        "bash",
        "zsh",
        "fish",
        "dash",
        "ksh",
        "pwsh",
        "powershell",
        "python",
        "python3",
        "node",
        "deno",
        "bun",
        "perl",
        "ruby",
        "php",
        "mongosh",
    ]
)
SQL_CLIENTS = frozenset(["psql", "mysql", "sqlite3", "duckdb", "clickhouse-client"])
INLINE_FLAGS = frozenset(["-c", "-e", "--eval", "--command", "--execute", "-x"])
SQL_WRITE = re.compile(
    r"(?i)\b(insert|update|delete|drop|alter|create|truncate|replace|grant|revoke|merge)\b"
)


def _inline_code(argv: list[str]) -> str | None:
    """`sh -c ...`, `python -c ...`, `mongosh --eval ...` run arbitrary code; a SQL client may
    only be given statements that write nothing."""
    if not argv:
        return None
    program = argv[0].rsplit("/", 1)[-1]
    if program in SHELLS and INLINE_FLAGS & set(argv[1:]):
        return f"{program} with inline code can do anything; wrap the read in a script instead"
    if program in SQL_CLIENTS and any(
        SQL_WRITE.search(item) for item in argv[1:] if "{" not in item
    ):
        return f"{program} is given a statement that writes"
    return None
