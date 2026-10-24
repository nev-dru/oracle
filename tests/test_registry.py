from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from oracle.config import REPO_ROOT
from oracle.registry.loader import load_registry
from oracle.registry.params import ParamError, validate_params
from oracle.registry.schema import Tool, placeholder_params
from tests.conftest import REGISTRY


def tool_dict(**over: object) -> dict:
    base = {
        "name": "demo",
        "description": "demo tool",
        "ops": [
            {
                "name": "search",
                "when": "find things",
                "params": {"q": {"type": "str", "required": True}},
                "argv": ["demo", "{q}"],
            }
        ],
    }
    base.update(over)
    return base


def write(directory: Path, data: dict, name: str | None = None) -> None:
    (directory / f"{name or data['name']}.yaml").write_text(yaml.safe_dump(data))


def test_home_registry_loads_clean() -> None:
    reg = load_registry(REGISTRY)
    assert reg.problems == []
    assert {"code-search", "code-graph", "acli", "workiq"} <= set(reg.tools)
    assert "code-graph.find_symbol" in reg.op_refs()


def test_golden_op_names_exist_in_registry() -> None:
    """golden.yaml refers to ops by name; renaming one must fail here, not silently in eval."""
    golden = yaml.safe_load((REPO_ROOT / "golden.yaml").read_text())
    refs = set(load_registry(REGISTRY).op_refs())
    for case in golden["cases"]:
        expect = case["expect"]
        for ref in expect.get("tools_all", []):
            assert ref in refs, f"{case['id']}: {ref}"
        assert any(r in refs for r in expect.get("tools_any", [])) or not expect.get("tools_any")


def test_undeclared_placeholder_is_rejected() -> None:
    data = tool_dict()
    data["ops"][0]["argv"] = ["demo", "{nope}"]
    with pytest.raises(ValidationError, match="undeclared param 'nope'"):
        Tool.model_validate(data)


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Tool.model_validate(tool_dict(colour="blue"))


def test_mutating_tool_refused_and_mutating_op_dropped(tmp_path: Path) -> None:
    write(tmp_path, tool_dict(name="writer", mutating=True))
    mixed = tool_dict(name="mixed")
    mixed["ops"].append({"name": "delete", "when": "x", "argv": ["demo"], "mutating": True})
    write(tmp_path, mixed)
    reg = load_registry(tmp_path)
    assert "writer" not in reg.tools
    assert [op.name for op in reg.tools["mixed"].ops] == ["search"]
    assert reg.problems == []  # refusing writes is the design, not a broken manifest
    assert any("refused" in n for n in reg.notes)
    assert any("dropped mutating" in n for n in reg.notes)


def test_bad_manifest_does_not_block_others(tmp_path: Path) -> None:
    write(tmp_path, tool_dict(name="good"))
    (tmp_path / "broken.yaml").write_text("name: [unclosed")
    write(tmp_path, tool_dict(name="other"), name="mismatch")
    reg = load_registry(tmp_path)
    assert list(reg.tools) == ["good"]
    assert len(reg.problems) == 2


def test_profile_marks_inactive_tools(tmp_path: Path) -> None:
    write(tmp_path, tool_dict(name="a"))
    write(tmp_path, tool_dict(name="b"))
    reg = load_registry(tmp_path, {"a"})
    assert list(reg.tools) == ["a"]
    assert reg.inactive == ["b"]


def test_placeholder_params_grammar() -> None:
    assert placeholder_params("{q}") == ["q"]
    assert placeholder_params("{?repo:--repo,repo}") == ["repo", "repo"]
    assert placeholder_params("{?since:--since=,since}") == ["since", "since"]
    assert placeholder_params("{stdin:body}") == ["body"]
    assert placeholder_params("--limit={limit}") == ["limit"]


def test_validate_params_coerces_and_checks() -> None:
    reg = load_registry(REGISTRY)
    _, op = reg.resolve("code-graph.expand")
    out = validate_params(op, {"ids": "a, b", "limit": "7"})
    assert out == {"ids": ["a", "b"], "repo": None, "direction": "callers", "limit": 7}
    with pytest.raises(ParamError, match="one of"):
        validate_params(op, {"ids": ["a"], "direction": "sideways"})
    with pytest.raises(ParamError, match="at most"):
        validate_params(op, {"ids": ["a"], "limit": 999})
    with pytest.raises(ParamError, match="missing required"):
        validate_params(op, {})
    with pytest.raises(ParamError, match="unknown params"):
        validate_params(op, {"ids": ["a"], "depth": 2})
    with pytest.raises(ParamError, match="must be int"):
        validate_params(op, {"ids": ["a"], "limit": True})


def test_unreadable_manifest_and_missing_dir_are_problems_not_crashes(tmp_path: Path) -> None:
    write(tmp_path, tool_dict(name="good"))
    (tmp_path / "latin.yaml").write_bytes("name: caf\xe9".encode("latin-1"))
    (tmp_path / "dir.yaml").mkdir()
    reg = load_registry(tmp_path)
    assert list(reg.tools) == ["good"] and len(reg.problems) == 2
    missing = load_registry(tmp_path / "nope")
    assert missing.problems and "does not exist" in missing.problems[0]


def test_bad_jsonpath_fails_at_load() -> None:
    data = tool_dict()
    data["ops"][0]["returns"] = {"path": "$..results"}
    with pytest.raises(ValidationError, match="cannot parse"):
        Tool.model_validate(data)


def test_references_inactive_fine_unadded_note_typo_problem(tmp_path: Path) -> None:
    a = tool_dict(name="a")
    a["ops"][0]["chains_to"] = ["b.search", "later.op"]
    write(tmp_path, a)
    write(tmp_path, tool_dict(name="b"))
    reg = load_registry(tmp_path, {"a"})
    assert reg.problems == []
    assert reg.notes == ["a.search: references tools not registered yet ['later.op']"]
    a["ops"][0]["chains_to"] = ["b.nope"]
    write(tmp_path, a)
    typo = load_registry(tmp_path)
    assert typo.problems == ["a.search: references ops that do not exist ['b.nope']"]


def test_explicit_null_takes_the_default() -> None:
    _, op = load_registry(REGISTRY).resolve("acli.jira.search")
    assert validate_params(op, {"jql": "x", "limit": None})["limit"] == 20
