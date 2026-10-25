"""The AST check every model-written script passes before it runs. An allowlist, not a
denylist: a denylist of `subprocess`, `socket` and friends misses `__import__`, `importlib`,
`getattr(os, ...)`, `eval`, `exec`, `open(..., "w")` and `ctypes`.

A script may use only the syntax listed in ALLOWED_NODES, may import only the generated tool
stubs and a short list of pure standard-library modules, may not touch any name or attribute
that starts with `_`, and may call only safe builtins. The child process also runs with
restricted builtins and no secrets, so this check is the first wall, not the only one.
"""

import ast
import re

# Each module was read for what it exposes as plain attributes: `typing` and `statistics` hold
# `sys`, `operator.attrgetter` takes an attribute name as a string, `functools.update_wrapper`
# copies attributes by name, so none of them is here. The child also hands a script a view of
# each module without its submodules.
# fmt: off
ALLOWED_IMPORTS = frozenset({
    "oracle_tools", "json", "re", "itertools", "collections", "math", "datetime",
})
SAFE_BUILTINS = frozenset({
    "abs", "all", "any", "bool", "chr", "dict", "divmod", "enumerate", "filter", "float",
    "format", "frozenset", "hash", "int", "isinstance", "iter", "len", "list", "map", "max",
    "min", "next", "ord", "print", "range", "repr", "reversed", "round", "set", "slice",
    "sorted", "str", "sum", "tuple", "zip", "True", "False", "None",
    "Exception", "ArithmeticError", "AttributeError", "IndexError", "KeyError",
    "LookupError", "RuntimeError", "StopIteration", "TypeError", "ValueError",
    "ZeroDivisionError",
})
FORBIDDEN_NAMES = frozenset({
    "eval", "exec", "compile", "open", "input", "breakpoint", "help", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "memoryview", "object", "type", "super",
    "classmethod", "staticmethod", "property", "issubclass",
})
# Driving a coroutine by hand hands back the event loop's own futures, and through them the
# loop; a script only awaits.
DRIVER_ATTRS = frozenset({
    "send", "throw", "asend", "athrow", "get_loop", "get_coro", "get_stack", "print_stack",
})
# fmt: on
# frame and code objects lead back to the real builtins: gen.gi_frame.f_back.f_globals
FORBIDDEN_ATTRS = re.compile(r"^(gi|cr|ag|tb|f|co)_")
# Every node type a script may use. Anything else is refused: `match` patterns read attributes
# by name with no Attribute node to check, and syntax a later Python adds is unknown here.
ALLOWED_NODES = (
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete, ast.Pass,
    ast.FunctionDef, ast.AsyncFunctionDef, ast.Return, ast.Lambda, ast.arguments, ast.arg,
    ast.If, ast.For, ast.AsyncFor, ast.While, ast.Break, ast.Continue,
    ast.Try, ast.ExceptHandler, ast.Raise, ast.Assert, ast.Import, ast.ImportFrom, ast.alias,
    ast.BoolOp, ast.NamedExpr, ast.BinOp, ast.UnaryOp, ast.IfExp, ast.Compare, ast.Call,
    ast.keyword, ast.Await, ast.Attribute, ast.Subscript, ast.Slice, ast.Starred, ast.Name,
    ast.Constant, ast.JoinedStr, ast.FormattedValue, ast.List, ast.Tuple, ast.Dict, ast.Set,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension,
    ast.expr_context, ast.boolop, ast.operator, ast.unaryop, ast.cmpop,
)  # fmt: skip


def script_problems(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error line {exc.lineno}: {exc.msg}"]
    problems: list[str] = []
    todo: list[ast.AST] = [tree]
    while todo:
        node = todo.pop()
        problems += _node_problems(node)
        if isinstance(node, ALLOWED_NODES):  # what sits inside refused syntax goes unsaid
            todo += reversed(list(ast.iter_child_nodes(node)))
    return list(dict.fromkeys(problems))


def _node_problems(node: ast.AST) -> list[str]:
    line = getattr(node, "lineno", "?")
    if isinstance(node, ast.ClassDef):
        return [f"line {line}: classes are not allowed; use dicts and functions"]
    if not isinstance(node, ALLOWED_NODES):
        return [f"line {line}: {_syntax(node)} is not allowed"]
    if isinstance(node, ast.Import | ast.ImportFrom):
        return _import_problems(node, line)
    if isinstance(node, ast.Attribute) and _forbidden_attr(node.attr):
        return [f"line {line}: attribute {node.attr} is not allowed"]
    if isinstance(node, ast.Name) and (node.id.startswith("_") or node.id in FORBIDDEN_NAMES):
        return [f"line {line}: {node.id} is not allowed"]
    return []


def _forbidden_attr(name: str) -> bool:
    return name.startswith("_") or bool(FORBIDDEN_ATTRS.match(name)) or name in DRIVER_ATTRS


def _syntax(node: ast.AST) -> str:
    """`global`, `match`, `with`: the keyword a model will recognise, else the node's name."""
    words = {"Global": "global", "Nonlocal": "nonlocal", "Match": "match", "With": "with",
             "AsyncWith": "async with", "Yield": "yield", "YieldFrom": "yield from"}  # fmt: skip
    name = type(node).__name__
    return words.get(name, name)


def _import_problems(node: ast.Import | ast.ImportFrom, line: object) -> list[str]:
    if isinstance(node, ast.Import):
        return [
            f"line {line}: import {a.name} is not allowed"
            for a in node.names
            if _root(a.name) not in ALLOWED_IMPORTS
        ]
    module = node.module or ""
    if node.level or _root(module) not in ALLOWED_IMPORTS:
        return [f"line {line}: from {module} import is not allowed"]
    return [
        f"line {line}: importing {a.name} is not allowed"
        for a in node.names
        if a.name.startswith("_") or a.name == "*"
    ]


def _root(module: str) -> str:
    return module.split(".", 1)[0]
