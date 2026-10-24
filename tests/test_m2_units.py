from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from oracle.exec.bindings import BindingError, binding_refs, fill_template, resolve
from oracle.exec.dag import PlanRunner
from oracle.exec.executor import Executor, OpResult
from oracle.exec.redact import MASK, redact
from oracle.plan.models import Step, step_errors
from oracle.registry.loader import Registry
from oracle.timewin import resolve_window
from tests.conftest import script_tool


def result(op: str, ids: list[str], records: list[dict] | None = None) -> OpResult:
    r = OpResult(op=op, params={})
    r.ids = ids
    r.records = records or [{"id": i, "name": f"n-{i}"} for i in ids]
    r.pointers = [f"t:{i}" for i in ids]
    return r


def test_bindings_resolve_ids_pointers_fields_and_concat() -> None:
    table = {"s1": [result("t.a", ["a", "b"])], "s2": [result("t.b", ["c"])]}
    get = lambda step: table.get(step, [])  # noqa: E731
    assert resolve("$s1.ids + $s2.ids", get) == ["a", "b", "c"]
    assert resolve("$s1.pointers", get) == ["t:a", "t:b"]
    assert resolve("$s1.results[0].name", get) == "n-a"
    assert resolve("$s1.results[*].name", get) == ["n-a", "n-b"]
    assert resolve("$s9.results[0].name", get) is None
    assert binding_refs({"x": "$s1.ids + $s2/stage.ids", "y": "plain"}) == {"s1", "s2/stage"}
    with pytest.raises(BindingError, match="bad binding"):
        binding_refs({"x": "$s1.nope"})


def test_recipe_templates_fill_params_and_rewrite_stage_bindings() -> None:
    values = {"name": "Andrew Neville", "ids": ["a", "b"]}
    assert fill_template('assignee = "{name}"', values, "s1", {"search"}) == (
        'assignee = "Andrew Neville"'
    )
    assert fill_template("{ids}", values, "s1", set()) == ["a", "b"]
    assert fill_template("$search.ids", values, "s1", {"search"}) == "$s1/search.ids"
    assert fill_template("$s0.ids", values, "s1", {"search"}) == "$s0.ids"
    assert fill_template("{unknown}", values, "s1", set()) == "{unknown}"


@pytest.mark.parametrize(
    ("question", "label", "hours"),
    [
        ("important events from today", "today", None),
        ("what broke yesterday", "yesterday", 24),
        ("tickets updated in the past 3 days", "past 3 days", 72),
        ("what happened this week", "this week", None),
    ],
)
def test_time_windows(question: str, label: str, hours: int | None) -> None:
    now = datetime(2026, 10, 22, 15, 30, tzinfo=timezone(timedelta(hours=-7)))  # a Thursday
    window = resolve_window(question, now)
    assert window is not None and window.label == label
    if hours:
        assert window.end - window.start == timedelta(hours=hours)
    assert window.start.tzinfo is not None and "-07:00" in window.describe()


def test_no_time_in_question_means_no_window() -> None:
    assert resolve_window("who is andrew neville", datetime.now().astimezone()) is None


def test_redaction_masks_common_secret_shapes() -> None:
    record = {
        "env": "DB_URL=postgres://app:hunter2@db:5432/x password=s3cret",
        "nested": ["token ghp_" + "a" * 36, {"k": "AKIAABCDEFGHIJKLMNOP"}],
        "count": 3,
    }
    out = redact(record)
    flat = str(out)
    assert "hunter2" not in flat and "s3cret" not in flat and "ghp_" not in flat
    assert "AKIA" not in flat and MASK in flat and out["count"] == 3


def test_step_errors_check_bindings_recipes_and_earlier_rounds(fake_registry: Registry) -> None:
    allowed = set(fake_registry.op_refs()) | {"people.profile"}
    steps = [
        Step(id="s1", op="workiq.people.lookup", params={"name": "andrew"}),
        Step(id="s2", op="acli.jira.search", params={"jql": "$s1.results[0].email"}),
        Step(id="s3", op="acli.jira.get", params={"key": "$s9.ids"}),
        Step(id="s4", op="people.profile", params={"nickname": "x"}),
    ]
    errors = step_errors(steps, fake_registry, allowed, 12)
    assert not any(e.startswith("s2") for e in errors)
    assert any("s3: binds to unknown steps ['s9']" in e for e in errors)
    assert any(e.startswith("s4") and "unknown params" in e for e in errors)
    later = [Step(id="s5", op="acli.jira.get", params={"key": "$s1.ids"})]
    assert step_errors(later, fake_registry, allowed, 12, earlier={"s1"}) == []
    assert any(
        "unique and new" in e for e in step_errors(later, fake_registry, allowed, 12, {"s5"})
    )


PRINT_ITEMS = """
import json, sys
ids = sys.argv[sys.argv.index("--") + 1].split(",") if "--" in sys.argv else []
print(json.dumps({"items": [{"id": i} for i in ids]}))
"""


def two_op_registry(tmp_path: Path) -> Registry:
    import sys

    tool = script_tool(
        tmp_path,
        PRINT_ITEMS,
        op={
            "argv": [sys.executable, str(tmp_path / "tool.py"), "{?ids:--,ids}"],
            "params": {"ids": {"type": "list[str]"}},
            "returns": {"path": "$.items[*]", "id_field": "id", "pointer": "t:{id}"},
        },
    )
    return Registry(tools={"t": tool})


async def test_dag_waves_bindings_skips_and_unavailable(tmp_path: Path) -> None:
    reg = two_op_registry(tmp_path)
    runner = PlanRunner(reg, Executor(reg, {"PATH": "/usr/bin"}, tmp_path / "spill"))
    runs = await runner.run(
        [
            Step(id="s1", op="t.run", params={"ids": ["a", "b"]}),
            Step(id="s2", op="t.run", params={"ids": "$s1.ids"}),
            Step(id="s3", op="t.run", params={}),
            Step(id="s4", op="t.run", params={"ids": "$s3.ids"}),
            Step(id="s5", op="nope.op", params={}),
        ]
    )
    by_id = {r.step_id: r.result for r in runs}
    assert by_id["s2"].ids == ["a", "b"]
    assert by_id["s3"].ok and by_id["s3"].records == []
    assert by_id["s4"].status == "skipped" and "s3" in (by_id["s4"].error or "")
    assert by_id["s5"].status == "unavailable"


async def test_recipe_runs_stages_aliases_outputs_and_falls_back(tmp_path: Path) -> None:
    from oracle.registry.schema import Recipe

    reg = two_op_registry(tmp_path)
    reg.recipes["t.chain"] = Recipe.model_validate(
        {
            "name": "t.chain",
            "when": "w",
            "params": {"first": {"type": "list[str]"}},
            "steps": [
                {"id": "one", "op": "t.run", "params": {"ids": "{first}"}},
                {"id": "two", "op": "t.run", "params": {"ids": "$one.ids"}},
                {"id": "gone", "op": "absent.op"},
            ],
            "output": ["two"],
            "on_empty": [{"id": "fallback", "op": "t.run", "params": {"ids": ["z"]}}],
        }
    )
    runner = PlanRunner(reg, Executor(reg, {"PATH": "/usr/bin"}, tmp_path / "spill"))
    runs = await runner.run(
        [
            Step(id="s1", op="t.chain", params={"first": ["x"]}),
            Step(id="s2", op="t.run", params={"ids": "$s1.ids"}),
        ]
    )
    by_id = {r.step_id: r for r in runs}
    assert by_id["s1/two"].result.ids == ["x"] and by_id["s1/two"].recipe == "t.chain"
    assert by_id["s1/gone"].result.status == "unavailable"
    assert by_id["s2"].result.ids == ["x"]  # bound through the recipe's output stage
    assert runner.recipes_used == ["t.chain"]
    empty = await runner.run([Step(id="s3", op="t.chain", params={})])
    assert [r.step_id for r in empty][-1] == "s3/fallback"
    assert runner.results_of("s3")[0].ids == ["z"]


async def test_identical_lookups_run_once_and_are_reused(tmp_path: Path) -> None:
    from oracle.render import trace_line

    reg = two_op_registry(tmp_path)
    runner = PlanRunner(reg, Executor(reg, {"PATH": "/usr/bin"}, tmp_path / "spill"))
    runs = await runner.run(
        [
            Step(id="s1", op="t.run", params={"ids": ["a"]}),
            Step(id="s2", op="t.run", params={"ids": ["b"]}),
        ]
    )
    again = await runner.run([Step(id="s3", op="t.run", params={"ids": ["a"]})], round_=1)
    assert again[0].same_as == "s1" and again[0].result is runs[0].result
    assert "same as s1, reused" in trace_line(again[0])
