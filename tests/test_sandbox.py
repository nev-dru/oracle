"""Bypass tests for the script check, written before the runner (each must be refused)."""

import pytest

from oracle.deep.sandbox import script_problems

BYPASSES = {
    "subprocess": "import subprocess",
    "os system": "import os\nos.system('rm -rf /')",
    "dunder import": "__import__('os').system('x')",
    "importlib": "import importlib\nimportlib.import_module('os')",
    "getattr builtins": "getattr(print, '__self__')",
    "eval": "eval('1+1')",
    "exec": "exec('import os')",
    "open write": "open('/tmp/x', 'w').write('x')",
    "ctypes": "import ctypes",
    "socket": "import socket",
    "requests": "import requests",
    "relative import": "from . import x",
    "dunder attribute": "().__class__.__bases__[0].__subclasses__()",
    "private stub": "import oracle_tools\noracle_tools._rpc.call('x', {})",
    "private name import": "from oracle_tools import _rpc",
    "star import": "from oracle_tools import *",
    "builtins name": "__builtins__['open']('/etc/passwd')",
    "globals": "globals()['x'] = 1",
    "class trick": "class A: pass",
    "asyncio": "import asyncio\nasyncio.create_subprocess_exec('sh')",
    "pathlib": "import pathlib\npathlib.Path('/etc/passwd').read_text()",
    "compile": "compile('x', 'f', 'exec')",
    "vars": "vars()",
    "type": "type('A', (), {})",
}


@pytest.mark.parametrize("name", sorted(BYPASSES))
def test_bypass_is_refused(name: str) -> None:
    assert script_problems(BYPASSES[name]), name


def test_ordinary_scripts_pass() -> None:
    code = """
import json
from collections import Counter
from oracle_tools import code_search, code_graph, report, gather

hits = await code_search.search(query="ownership transfer", limit=20)
frontier = {h["symbol_id"] for h in hits if h["path"]}
seen = set()
for _hop in range(3):
    pass
callers = await code_graph.expand(ids=sorted(frontier), direction="callers")
counts = Counter(c["path"] for c in callers)
report({"callers": callers, "by_file": dict(counts)})
"""
    # `_hop` is a private name; scripts must use names without a leading underscore
    assert script_problems(code) == ["line 9: _hop is not allowed"]
    assert script_problems(code.replace("_hop", "hop")) == []


def test_syntax_errors_are_reported() -> None:
    assert script_problems("def (:")[0].startswith("syntax error")


MORE_BYPASSES = {
    "typing sys": "import typing\ntyping.sys.modules['os'].system('x')",
    "statistics sys": "import statistics\nstatistics.sys.modules",
    "operator attrgetter": "import operator\noperator.attrgetter('__class__')(1)",
    "functools update_wrapper": "import functools\nfunctools.update_wrapper",
    "generator frame walk": (
        "def g():\n    yield gen.gi_frame.f_back\ngen = g()\nnext(gen).f_back.f_globals"
    ),
    "coroutine frame": "async def f():\n    pass\nf().cr_frame",
    "traceback frame": "try:\n    1 / 0\nexcept Exception as e:\n    e.tb_next.tb_frame",
    "code object": "def f():\n    pass\nf.co_consts",
    "match class attributes": "match 1:\n    case int(real=r):\n        pass",
    "match mapping": "match {}:\n    case {'a': a}:\n        pass",
    "coroutine driven by hand": "async def f():\n    pass\nf().send(None)",
    "coroutine thrown into": "async def f():\n    pass\nf().throw(ValueError)",
    "future's loop": "x.get_loop()",
    "with": "with x:\n    pass",
    "global": "def f():\n    global x",
    "yield": "def g():\n    yield 1",
}


@pytest.mark.parametrize("name", sorted(MORE_BYPASSES))
def test_more_bypasses_are_refused(name: str) -> None:
    assert script_problems(MORE_BYPASSES[name]), name


def test_refusals_name_the_syntax() -> None:
    code = "match 1:\n    case 1:\n        pass"
    assert script_problems(code) == ["line 1: match is not allowed"]
