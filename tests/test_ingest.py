"""oracle tool add: safety first (nothing mutating ever runs), then the staged pipeline, with
scripted models and the fake tools."""

import os
import stat
from pathlib import Path

import pytest
import yaml

from oracle.config import Config
from oracle.registry.importers.cli_draft import DraftContext
from oracle.registry.importers.cli_probe import probe_cli, subcommand_names
from oracle.registry.importers.infer import infer_returns
from oracle.registry.loader import load_registry
from oracle.registry.safety import argv_problem, mcp_problem, rendered_problem
from oracle.registry.schema import Tool
from oracle.workflows.ingest import IngestError, _verify_op, activate, ingest_cli
from tests.conftest import REGISTRY, scripted

TRAP = """#!/bin/sh
# Leaves a marker whenever a mutating subcommand runs, even with --help.
case "$1" in
  delete|sync) echo ran > "{marker}"; echo "deleted"; exit 0;;
esac
cat <<'HELP'
trapctl manages things.

Available Commands:
  get         Show things
  delete      Delete things
  sync        Sync things

Flags:
  -h, --help   help for trapctl
HELP
"""


@pytest.fixture
def trap(tmp_path: Path) -> tuple[Path, Path]:
    marker = tmp_path / "MUTATED"
    binary = tmp_path / "bin" / "trapctl"
    binary.parent.mkdir()
    binary.write_text(TRAP.replace("{marker}", str(marker)))
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    return binary, marker


def env_with(binary: Path) -> dict[str, str]:
    return {**os.environ, "PATH": f"{binary.parent}:{os.environ['PATH']}"}


async def test_probe_never_runs_mutating_subcommands(trap: tuple[Path, Path]) -> None:
    binary, marker = trap
    probe = await probe_cli("trapctl", "trapctl", env_with(binary))
    assert set(probe.helps) == {""}  # `get --help` printed the root help again: not recursed
    assert probe.refused == ["delete", "sync"]
    assert not marker.exists()


async def test_drafted_mutating_op_never_reaches_a_subprocess(
    trap: tuple[Path, Path], home_cfg: Config
) -> None:
    _binary, marker = trap
    tool = Tool.model_validate(
        {
            "name": "trapctl",
            "description": "d",
            "ops": [
                {
                    "name": "remove",
                    "when": "w",
                    "argv": ["trapctl", "delete", "{name}"],
                    "params": {"name": {"required": True}},
                    "examples": [{"params": {"name": "x"}}],
                }
            ],
        }
    )
    cfg = Config(home_cfg.home, home_cfg.registry_dir, None, {}, None)
    kept, detail = await _verify_op(
        tool,
        tool.ops[0],
        {},
        cfg,
        scripted({}),
        DraftContext("trapctl", "trapctl"),
        None,  # type: ignore[arg-type]
    )
    assert kept is None and "mutating" in detail
    assert not marker.exists()
    assert argv_problem(["kubectl", "delete", "pod", "{name}"])


def test_loader_refuses_declared_read_only_but_mutating_argv(tmp_path: Path) -> None:
    data = {
        "name": "sneaky",
        "description": "d",
        "ops": [{"name": "get", "when": "w", "argv": ["argocd", "app", "sync", "{app}"],
                 "params": {"app": {"required": True}}}],
    }  # fmt: skip
    (tmp_path / "sneaky.yaml").write_text(yaml.safe_dump(data))
    reg = load_registry(tmp_path)
    assert "sneaky" not in reg.tools and "looks mutating" in reg.problems[0]


def test_subcommands_are_read_from_common_help_formats() -> None:
    cobra = "Available Commands:\n  get    Show\n  logs   Print logs\n\nFlags:\n  -h  help\n"
    kubectl = "Basic Commands (Beginner):\n  create   Create\n  get   Display\n"
    argparse = "usage: acli [-h] {jira,confluence} ...\n"
    assert subcommand_names(cobra) == ["get", "logs"]
    assert subcommand_names(kubectl) == ["create", "get"]
    assert subcommand_names(argparse) == ["jira", "confluence"]


def test_returns_are_inferred_from_real_shapes() -> None:
    doc = {
        "issues": [
            {"key": "A-1", "summary": "s", "status": "Open"},
            {"key": "A-2", "summary": "t", "status": "Done"},
        ],
        "total": 2,
    }
    r = infer_returns(doc, "jira")
    assert r is not None and r.path == "$.issues[*]" and r.id_field == "key"
    assert r.pointer == "jira:{key}" and r.summary_fields[0] == "key"
    code = infer_returns(
        [{"chunk_id": "ab12", "path": "/a.py", "start_line": 3, "title": "f"}], "vex"
    )
    assert code is not None and code.pointer == "code:{path}:{start_line}"
    assert "chunk_id" not in code.summary_fields


def draft_for_workiq() -> dict:
    return {
        "name": "directory",
        "description": "People directory",
        "tags": ["people", "who"],
        "when": "who someone is",
        "ops": [
            {
                "name": "lookup",
                "when": "look up a person",
                "params": {"name": {"type": "str", "required": True}},
                # returns guessed wrong on purpose: verification must correct it from real output
                "argv": ["workiq", "people", "lookup", "--name", "{name}", "--json"],
                "returns": {"format": "json", "path": "$.results[*]"},
                "examples": [{"params": {"name": "andrew neville"}}],
            },
            {
                "name": "today",
                "when": "meetings today",
                "argv": ["workiq", "calendar", "today", "--json"],
                "examples": [{"params": {}}],
            },
        ],
    }


async def test_ingest_stages_a_grounded_manifest_and_activates_on_request(
    home_cfg: Config, tmp_path: Path
) -> None:
    from oracle.config import load_config

    cfg = load_config({"ORACLE_HOME": str(tmp_path / "home"), "ORACLE_PROFILE": "home",
                       "ORACLE_REGISTRY": str(tmp_path / "registry")})  # fmt: skip
    (tmp_path / "registry").mkdir()
    (tmp_path / "registry" / "acli.yaml").write_text((REGISTRY / "acli.yaml").read_text())
    registry = load_registry(tmp_path / "registry")

    def model_for(prompt_kind: str):  # type: ignore[no-untyped-def]
        return scripted(draft_for_workiq())

    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    def respond(messages, info):  # type: ignore[no-untyped-def]
        schema = info.output_tools[0].parameters_json_schema
        props = set(schema.get("properties", {}))
        if "ops" in props:
            body = draft_for_workiq()
        elif "proposals" in props:
            body = {
                "description": "People directory: titles, teams and managers.",
                "when": "Who someone is: title, team, manager.",
                "not_when": "Tickets: use acli.",
                "proposals": [],
            }
        elif "cases" in props:
            body = {"cases": [{"query": "who is andrew neville", "op": "lookup"}]}
        else:
            raise AssertionError(f"unexpected agent {props}")
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, body)])

    result = await ingest_cli("workiq", "directory", cfg, registry, FunctionModel(respond))
    assert result.manifest is not None and all(o.kept for o in result.outcomes), result.outcomes
    lookup = result.manifest.op("lookup")
    assert lookup.returns.path == "$.people[*]" and lookup.returns.id_field == "id"
    assert lookup.examples[0].expect.min_results == 1
    for f in (
        "probe.json",
        "draft.yaml",
        "manifest.yaml",
        "boundary.md",
        "eval_candidates.yaml",
        "report.md",
    ):
        assert (result.staged / f).is_file(), f
    assert "directory" not in load_registry(tmp_path / "registry").tools  # staged, not active
    target = activate(cfg, "directory")
    assert target.is_file() and "directory" in load_registry(tmp_path / "registry").tools
    with pytest.raises(IngestError, match="exists"):
        activate(cfg, "directory")
    with pytest.raises(IngestError, match="already has a manifest"):
        await ingest_cli("workiq", "acli", cfg, registry, FunctionModel(respond))


NOTES_SERVER = Path(__file__).parent / "fakes" / "mcp_notes_server.py"


def mcp_command() -> str:
    import shlex
    import sys

    return f"{shlex.quote(sys.executable)} {shlex.quote(str(NOTES_SERVER))}"


async def test_mcp_probe_refuses_mutating_tools(tmp_path: Path) -> None:
    from oracle.registry.importers.mcp_import import parse_command, probe_mcp

    marker = tmp_path / "MUTATED"
    probe = await probe_mcp(
        parse_command(mcp_command()), {**os.environ, "NOTES_MARKER": str(marker)}
    )
    assert {t.name for t in probe.tools} == {
        "search_notes",
        "list_notes",
        "delete_note",
        "rebuild_index",
    }
    assert set(probe.refused) == {"delete_note", "rebuild_index"}
    assert [t.name for t in probe.sampleable] == ["list_notes"]
    assert not marker.exists()


async def test_mcp_ingest_never_exposes_or_runs_a_mutating_tool(tmp_path: Path) -> None:
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    from oracle.config import load_config
    from oracle.workflows.ingest import ingest_mcp

    marker = tmp_path / "MUTATED"
    os.environ["NOTES_MARKER"] = str(marker)
    cfg = load_config(
        {"ORACLE_HOME": str(tmp_path / "home"), "ORACLE_REGISTRY": str(tmp_path / "reg")}
    )
    (tmp_path / "reg").mkdir()
    attempts: list[str] = []

    def respond(messages, info):  # type: ignore[no-untyped-def]
        props = set(info.output_tools[0].parameters_json_schema.get("properties", {}))
        if "expose" in props:
            sneaky = {"tool": "delete_note", "when": "remove a note", "example": {"note_id": "n1"}}
            good = [
                {
                    "tool": "search_notes",
                    "when": "find notes by words",
                    "example": {"query": "gateway"},
                },
                {"tool": "list_notes", "when": "every note", "example": {}},
            ]
            attempts.append("draft")
            body = {"description": "Notes.", "when": "Notes.", "not_when": "-", "tags": ["notes"],
                    "expose": ([sneaky] if len(attempts) == 1 else []) + good}  # fmt: skip
        elif "cases" in props:
            body = {"cases": [{"query": "notes about gateway timeouts", "op": "search_notes"}]}
        else:
            body = {"description": "Notes.", "when": "Notes.", "not_when": "-", "proposals": []}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, body)])

    try:
        result = await ingest_mcp(
            mcp_command(), "notes", cfg, load_registry(tmp_path / "reg"), FunctionModel(respond)
        )
    finally:
        os.environ.pop("NOTES_MARKER", None)
    assert attempts == ["draft", "draft"]  # the draft exposing delete_note was rejected and retried
    assert result.manifest is not None
    assert {op.mcp_tool for op in result.manifest.ops} == {"search_notes", "list_notes"}
    search = result.manifest.op("search_notes")
    assert search.returns.path == "$.notes[*]" and search.returns.id_field == "id"
    assert result.refused == ["delete_note", "rebuild_index"] and not marker.exists()


def test_flags_before_a_subcommand_do_not_hide_it() -> None:
    from oracle.registry.schema import Param

    assert argv_problem(["kubectl", "-n", "prod", "delete", "pod", "{name}"])
    assert argv_problem(["argocd", "--server", "{s}", "app", "sync", "{app}"])
    assert argv_problem(["kubectl", "{verb}", "pods"], {"verb": Param(enum=["get", "delete"])})
    assert not argv_problem(
        ["kubectl", "{verb}", "pods"], {"verb": Param(enum=["get", "describe"])}
    )
    assert not argv_problem(["csearch", "--json", "-n", "{limit}", "{query}"])  # data, not a verb
    # rendered: the command exactly as it would run
    assert rendered_problem(["kubectl", "delete", "pods"])  # from `kubectl {verb} pods`
    assert rendered_problem(["argocd", "app", "sync", "billing", "-o", "json"])  # nested slot
    assert not rendered_problem(["cmm-graph", "path", "--to", "publish"])  # a flag value: data
    assert not rendered_problem(["csearch", "delete old tickets"])  # one item: data
    assert not rendered_problem(["kubectl", "logs", "user-sync-worker", "-n", "prod"])  # a name
    assert argv_problem(["sh", "-c", "kubectl delete pods --all"])
    assert argv_problem(["mongosh", "--eval", "db.users.drop()"])
    assert argv_problem(["sqlite3", "x.db", "DELETE FROM users"])
    assert not argv_problem(["sqlite3", "x.db", "SELECT * FROM users"])
    for name in ("deleteProject", "createIssue", "issues.delete", "delete_project"):
        assert mcp_problem(name), name
    assert not mcp_problem("searchIssues")
    assert not argv_problem(["kubectl", "get", "pods", "-o", "json"])
    assert not argv_problem(["acli", "jira", "workitem", "view", "{key}", "--json"])
    assert not argv_problem(["kubectl", "-n", "{ns}", "get", "pods"])
    assert not argv_problem(["vex", "search", "--", "{query}"])
    home = load_registry(REGISTRY)
    assert home.problems == [] and len(home.tools) == 8


def test_own_op_references_are_normalised_and_dropped_ones_removed() -> None:
    from oracle.registry.schema import Op, Param
    from oracle.workflows.ingest import _chains

    kept = [
        Op(name="app.list", when="w", argv=["argocd", "app", "list"],
           chains_to=["app.get", "chunk.get", "acli.jira.search"]),
        Op(name="app.get", when="w", argv=["argocd", "app", "get", "{name}"],
           params={"name": Param(required=True, enum_from="app.list")}),
    ]  # fmt: skip
    drafted = {"app.list", "app.get", "chunk.get"}
    first = _chains("argocd", kept[0], kept, drafted)
    assert first.chains_to == ["argocd.app.get", "acli.jira.search"]  # chunk.get was dropped
    second = _chains("argocd", kept[1], kept, drafted)
    assert second.params["name"].enum_from == "argocd.app.list"
    bare = Op(name="find", when="w", argv=["mongo", "find"], chains_to=["collections"])
    assert _chains("mongo", bare, [bare], {"find", "collections"}).chains_to == []
