import pytest

from oracle.exec import jsonpath
from oracle.exec.parse import ParseError, parse_output
from oracle.exec.pointers import pointer_for
from oracle.exec.runners.argv import ArgvError, build_argv
from oracle.registry.schema import Returns


def test_argv_plain_conditional_and_stdin() -> None:
    template = ["t", "--json", "{q}", "{?repo:--repo,repo}", "{?v:--verbose}", "{stdin:body}"]
    argv, stdin = build_argv(template, {"q": "ownership", "repo": None, "v": True, "body": "x"})
    assert argv == ["t", "--json", "ownership", "--verbose"]
    assert stdin == "x"


def test_argv_renders_lists_json_and_embedded() -> None:
    argv, _ = build_argv(
        ["t", "--ids", "{ids}", "--filter", "{f}", "--limit={n}"],
        {"ids": ["a", "b"], "f": {"x": 1}, "n": 5},
    )
    assert argv == ["t", "--ids", "a,b", "--filter", '{"x":1}', "--limit=5"]


def test_argv_refuses_flag_injection_but_allows_safe_forms() -> None:
    with pytest.raises(ArgvError, match="starts with '-'"):
        build_argv(["t", "{q}"], {"q": "--delete-everything"})
    with pytest.raises(ArgvError, match="starts with '-'"):
        build_argv(["t", "{?s:--since,s}"], {"s": "-7d"})
    assert build_argv(["t", "{?s:--since=,s}"], {"s": "-7d"})[0] == ["t", "--since=-7d"]
    assert build_argv(["t", "--", "{q}"], {"q": "-x"})[0] == ["t", "--", "-x"]


def test_argv_unset_plain_param_is_an_error() -> None:
    with pytest.raises(ArgvError, match="unset"):
        build_argv(["t", "{q}"], {"q": None})


def test_jsonpath_subset() -> None:
    doc = {"results": [{"id": 1, "tags": ["a"]}, {"id": 2}], "next": "c2", "x-y": {"k": 3}}
    assert jsonpath.find(doc, "$.results[*].id") == [1, 2]
    assert jsonpath.find(doc, "$.results[0].tags[0]") == ["a"]
    assert jsonpath.find(doc, "$['x-y'].k") == [3]
    assert jsonpath.first(doc, "$.next") == "c2"
    assert jsonpath.find(doc, "$.missing[*]") == []
    with pytest.raises(jsonpath.PathError):
        jsonpath.find(doc, "results")


def test_parse_formats() -> None:
    _, recs = parse_output('{"items": [{"a": 1}, 2]}', Returns(path="$.items[*]"))
    assert recs == [{"a": 1}, {"value": 2}]
    _, recs = parse_output('{"a": 1}\n\n{"a": 2}\n', Returns(format="jsonl"))
    assert recs == [{"a": 1}, {"a": 2}]
    table = "NAME     READY   STATUS\nweb-1    1/1     Running\nweb-2    0/1     Crash Loop\n"
    _, recs = parse_output(table, Returns(format="table"))
    assert recs[1] == {"name": "web-2", "ready": "0/1", "status": "Crash Loop"}
    assert parse_output("  hello \n", Returns(format="text"))[1] == [{"text": "hello"}]
    with pytest.raises(ParseError, match="not JSON"):
        parse_output("oops", Returns())


def test_pointer_template_and_fallbacks() -> None:
    r = Returns(pointer="code:{path}:{line}", id_field="id")
    assert pointer_for({"path": "/a.py", "line": 3, "id": "x"}, r, "cg") == "code:/a.py:3"
    assert pointer_for({"path": "/a.py", "id": "x"}, r, "cg") == "cg:x"
    hashed = pointer_for({"other": 1}, Returns(), "cg")
    assert hashed.startswith("cg:#") and hashed == pointer_for({"other": 1}, Returns(), "cg")


def test_summary_drops_values_the_pointer_already_carries() -> None:
    from oracle.render import summarise

    record = {"title": "transfer_ownership", "path": "/r/a.py", "start_line": 11, "kind": "fn"}
    fields = ["title", "path", "start_line", "kind"]
    assert summarise(record, fields, "code:/r/a.py:11") == "transfer_ownership · fn"
