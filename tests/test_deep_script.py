"""The deep tier's script path: a scripted script writer, the real runner and child process,
the fake tools (no model spend)."""

from typing import Any

from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel

from oracle.ask import AskRequest, Models, ask
from oracle.config import Config
from oracle.deep.bridge import flatten, report_runs
from oracle.log.store import open_store
from oracle.registry.loader import Registry
from oracle.workflows.protocol import Trace
from tests.conftest import _final, echo_distiller, prompt_text

QUESTION = "what is going on with the ingestion gateway 5xx errors"
CHAIN_SCRIPT = """
import re
from oracle_tools import acli, gather, report, workiq

msgs = await workiq.teams_search(query="ingestion gateway 5xx", since="-7d")
keys = sorted({k for m in msgs for k in re.findall(r"[A-Z]+-[0-9]+", m.get("text", ""))})
tickets = await gather(*(acli.jira_get(key=k) for k in keys))
pages = await acli.confluence_search(cql='text ~ "ingestion gateway"')
authors = sorted({p["author"] for p in pages if p.get("author")})
people = await gather(*(workiq.people_lookup(name=a) for a in authors))
report({
    "tickets": [t for group in tickets for t in group],
    "pages": pages,
    "people": [p for group in people for p in group],
    "authors": len(authors),
})
"""


def scripter(*codes: str, seen: list[str] | None = None) -> FunctionModel:
    """Returns the scripts in order, one per model call; records each prompt in `seen`."""
    queue = list(codes)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if seen is not None:
            seen.append(prompt_text(messages) + "\n" + _retry_text(messages))
        code = queue.pop(0) if len(queue) > 1 else queue[0]
        return _final(info, {"interpretation": "the gateway 5xx incident", "code": code})

    return FunctionModel(respond)


def _retry_text(messages: list[ModelMessage]) -> str:
    return "\n".join(
        str(getattr(p, "content", "")) for m in messages for p in m.parts
        if type(p).__name__ in ("RetryPromptPart", "ToolReturnPart")
    )  # fmt: skip


def models(**over: Any) -> Models:
    base = dict(planner=echo_distiller(), distiller=echo_distiller(), reducer=echo_distiller())
    return Models(**(base | over))


async def test_script_follows_the_chain(home_cfg: Config, fake_registry: Registry) -> None:
    seen: list[str] = []
    req = AskRequest(QUESTION, level="deep", script=True)
    answer = await ask(
        req, home_cfg, fake_registry, models(scripter=scripter(CHAIN_SCRIPT, seen=seen))
    )
    assert answer.script == CHAIN_SCRIPT and not answer.failures, answer.failures
    assert "async def teams_search" in seen[0] and "async def jira_get" in seen[0]
    assert answer.ops_run[:3] == ["workiq.teams.search", "acli.jira.get", "acli.confluence.search"]
    pointers = {f.pointer for f in answer.findings}
    assert {"jira:OPS-1206", "conf:90011", "people:u-1003"} <= pointers
    assert answer.searched == ["acli", "workiq"]  # the report is not a tool
    ids = [r["step"] for r in answer.step_results]
    assert len(ids) == len(set(ids)), ids  # calls started together under gather
    assert any(
        line.startswith("script: attempt 1,") and line.endswith("ok") for line in answer.trace
    )
    async with (
        open_store(home_cfg.log_path) as store,
        store.db.execute("SELECT script FROM calls WHERE call_id = ?", (answer.call_id,)) as cur,
    ):
        assert (await cur.fetchone())[0] == CHAIN_SCRIPT  # type: ignore[index]


async def test_runtime_failure_is_corrected_once(home_cfg: Config, fake_registry: Registry) -> None:
    broken = (
        "from oracle_tools import workiq, report\n"
        'msgs = await workiq.teams_search(query="ingestion gateway 5xx", since="-7d")\n'
        'report(msgs[0]["no_such_field"])\n'
    )
    seen: list[str] = []
    req = AskRequest(QUESTION, level="deep", script=True)
    answer = await ask(
        req, home_cfg, fake_registry, models(scripter=scripter(broken, CHAIN_SCRIPT, seen=seen))
    )
    assert len(seen) == 2 and "Your previous script failed" in seen[1] and "KeyError" in seen[1]
    assert any(line.startswith("script: attempt 2,") for line in answer.trace)
    teams = [r for r in answer.step_results if r["op"] == "workiq.teams.search"]
    assert len(teams) == 1  # the second attempt reused the first attempt's search
    assert "jira:OPS-1206" in {f.pointer for f in answer.findings}


async def test_refused_script_goes_back_before_running(
    home_cfg: Config, fake_registry: Registry
) -> None:
    seen: list[str] = []
    bad = "import os\nos.system('true')\n"
    req = AskRequest(QUESTION, level="deep", script=True)
    answer = await ask(
        req, home_cfg, fake_registry, models(scripter=scripter(bad, CHAIN_SCRIPT, seen=seen))
    )
    assert "import os is not allowed" in seen[1] and answer.script == CHAIN_SCRIPT


async def test_tools_outside_the_shortlist_are_refused(
    home_cfg: Config, fake_registry: Registry
) -> None:
    seen: list[str] = []
    req = AskRequest(QUESTION, level="deep", script=True, scope=["workiq"])
    ok = (
        "from oracle_tools import workiq, report\n"
        "report(await workiq.teams_search(query='gateway'))\n"
    )
    answer = await ask(
        req, home_cfg, fake_registry, models(scripter=scripter(CHAIN_SCRIPT, ok, seen=seen))
    )
    assert "oracle_tools.acli is not listed" in seen[1] and answer.ops_run == [
        "workiq.teams.search"
    ]


async def test_two_failures_keep_the_first_report(
    home_cfg: Config, fake_registry: Registry
) -> None:
    partial = (
        "from oracle_tools import acli, report\n"
        'report(await acli.jira_get(key="OPS-1206"))\n'
        "x = 1 / 0\n"
    )
    worse = "x = 1 / 0\n"
    req = AskRequest(QUESTION, level="deep", script=True)
    answer = await ask(req, home_cfg, fake_registry, models(scripter=scripter(partial, worse)))
    assert any("the script failed after 2 attempt(s): ZeroDivisionError" in g for g in answer.gaps)
    assert "jira:OPS-1206" in {f.pointer for f in answer.findings}


async def test_the_budget_bounds_the_correction(home_cfg: Config, fake_registry: Registry) -> None:
    seen: list[str] = []
    req = AskRequest(QUESTION, level="deep", script=True, budget="1000")
    answer = await ask(
        req, home_cfg, fake_registry, models(scripter=scripter("x = 1 / 0\n", seen=seen))
    )
    assert len(seen) == 1 and "no second attempt: the budget or the time is spent" in answer.gaps


async def test_script_raises_the_level(home_cfg: Config, fake_registry: Registry) -> None:
    req = AskRequest(QUESTION, level="standard", script=True)
    answer = await ask(req, home_cfg, fake_registry, models(scripter=scripter(CHAIN_SCRIPT)))
    assert answer.level == "deep" and any("--script runs at deep" in n for n in answer.notes)
    async with (
        open_store(home_cfg.log_path) as store,
        store.db.execute(
            "SELECT level, ran_level FROM calls WHERE call_id = ?", (answer.call_id,)
        ) as cur,
    ):
        assert await cur.fetchone() == ("standard", "deep")  # as asked, then as it ran


async def test_a_nested_script_may_rise_to_a_deep_parent(
    home_cfg: Config, fake_registry: Registry
) -> None:
    """Under a deep parent a nested call defaults to standard, and --script may still raise
    it to deep: the parent's tier is the ceiling, not the child's own."""
    trace = Trace.root("deep").enter("oracle", "ask", "outer")
    trace = trace.enter("oracle", "ask", QUESTION, level="standard")
    req = AskRequest(QUESTION, level="standard", script=True)
    answer = await ask(
        req, home_cfg, fake_registry, models(scripter=scripter(CHAIN_SCRIPT)), trace=trace
    )
    assert answer.level == "deep" and answer.script is not None


def test_reported_values_become_records() -> None:
    origin = {"jira:OPS-1": "acli.jira.get"}
    value = {
        "tickets": [
            {"key": "OPS-1", "_pointer": "jira:OPS-1"},
            {"key": "OPS-1", "_pointer": "jira:OPS-1"},
        ],
        "by_file": {"a.py": 2, "b.py": 1},
        "count": 3,
        "forged": {"_pointer": "jira:NOT-SEEN", "key": "NOT-SEEN"},
    }
    assert flatten(value)[0] == {"by_file": {"a.py": 2, "b.py": 1}, "count": 3}
    per_op, extra = report_runs([value, "done"], origin, "c1")
    assert [(r.op, r.result.pointers) for r in per_op] == [("acli.jira.get", ["jira:OPS-1"])]
    assert extra is not None and extra.result.pointers == [
        "script:c1#1",
        "script:c1#2",
        "script:c1#3",
    ]
    assert {"key": "NOT-SEEN"} in extra.result.records  # a pointer the script made up is dropped


async def test_a_rejected_query_raises_in_the_script(
    home_cfg: Config, fake_registry: Registry
) -> None:
    code = (
        "from oracle_tools import acli, report\n"
        "try:\n"
        '    await acli.jira_search(jql="statusCategory != Done")\n'
        "except ValueError as e:\n"
        '    report({"could not search": str(e)})\n'
    )
    req = AskRequest(QUESTION, level="deep", script=True)
    answer = await ask(req, home_cfg, fake_registry, models(scripter=scripter(code)))
    assert answer.script == code and "acli.jira.search (error)" in " ".join(answer.failures)
