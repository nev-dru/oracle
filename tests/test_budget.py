"""The per-call budget: each tier's bounds, the token reserve, the per-call timeout, the usage
charge, and how a Ctrl-C during a model call is read. A fake clock, no models."""

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from oracle.budget import BOUNDS, SUMMARY_RESERVE, Bounds, Budget, Interrupted, parse_budget


@dataclass
class Ran:
    """What `agent.run` returns, as far as the budget reads it."""

    usage: Any


def ran(tokens_in: int = 10, tokens_out: int = 5) -> Ran:
    return Ran(SimpleNamespace(input_tokens=tokens_in, output_tokens=tokens_out, requests=1))


async def done(value: Ran) -> Ran:
    return value


async def slow() -> Ran:
    await asyncio.sleep(1)
    return ran()


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_parsing_and_admission() -> None:
    assert parse_budget(None, 100) == 100
    assert parse_budget("unlimited", 100) is None
    assert parse_budget("5000", 100) == 5000
    with pytest.raises(ValueError):
        parse_budget("-1", 100)
    budget = Budget(tokens=10_000, reserve=2_000)
    assert budget.admits(8_000) and not budget.admits(8_001)
    assert Budget(tokens=None).admits(10**12)


@pytest.mark.parametrize(
    ("level", "tokens", "reserve", "follow_up", "tool_calls"),
    [
        ("quick", None, 0, None, float("inf")),
        ("standard", None, 0, 60.0, float("inf")),
        ("deep", 200_000, SUMMARY_RESERVE, 300.0, 400),
        ("deep", 4_000, 1_000, 300.0, 400),  # a small budget keeps a quarter back
        ("deep", None, SUMMARY_RESERVE, None, 400),  # unlimited drops the wall clock too
    ],
)
def test_each_tier_gets_its_bounds(
    level: str, tokens: int | None, reserve: int, follow_up: float | None, tool_calls: float
) -> None:
    budget = Budget.for_tier(level, tokens)
    assert budget.tokens == (tokens if level == "deep" else None)
    assert budget.reserve == reserve and budget.bounds.follow_up_s == follow_up
    assert budget.bounds.model_s == 90.0 and budget.tool_calls_left() == tool_calls


def test_tokens_count_only_at_deep() -> None:
    assert Budget.for_tier("standard", 5_000).unlimited


@pytest.mark.parametrize(
    ("level", "at", "out_of_time", "late"),
    [
        ("standard", 59.0, False, False),
        ("standard", 61.0, True, False),
        ("standard", 91.0, True, True),
        ("deep", 299.0, False, False),
        ("deep", 301.0, True, False),  # deep has no summarise-by bound; its summary has a timeout
        ("quick", 10_000.0, False, False),
    ],
)
def test_time_bounds_read_one_clock(level: str, at: float, out_of_time: bool, late: bool) -> None:
    clock = Clock()
    budget = Budget(bounds=BOUNDS[level], clock=clock)  # starts on its clock
    clock.now = at
    assert budget.out_of_time() is out_of_time and budget.late_for_summary() is late
    assert budget.line().endswith(f"{at:.0f} s")


async def test_a_model_call_is_charged() -> None:
    budget = Budget(tokens=1_000)
    await budget.model(done(ran(30, 20)))
    await budget.model(done(ran(30, 20)))
    assert budget.spent == 100 and budget.usage.model_calls == 2 and budget.available() == 900


async def test_a_model_call_is_bounded() -> None:
    budget = Budget(bounds=Bounds(model_s=0.01))
    with pytest.raises(TimeoutError):
        await budget.model(slow())
    assert budget.usage.model_calls == 0


async def test_a_failure_while_the_stop_is_set_is_interrupted() -> None:
    stop = asyncio.Event()
    budget = Budget(stop=stop)

    async def killed() -> Ran:
        stop.set()  # the Ctrl-C lands while the call runs, and kills its process
        raise RuntimeError("claude exited with -2")

    with pytest.raises(Interrupted, match="claude exited with -2"):
        await budget.model(killed())
    assert budget.stopping()


async def test_a_call_started_after_the_stop_fails_for_its_own_reasons() -> None:
    stop = asyncio.Event()
    stop.set()
    budget = Budget(stop=stop)

    async def broken() -> Ran:
        raise RuntimeError("bad output")

    with pytest.raises(RuntimeError, match="bad output"):
        await budget.model(broken())


async def test_cancellation_passes_through() -> None:
    stop = asyncio.Event()
    budget = Budget(stop=stop)

    async def cancelled() -> Ran:
        stop.set()
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await budget.model(cancelled())


async def test_the_deep_summary_has_one_bound_for_all_its_calls() -> None:
    budget = Budget.for_tier("deep", 10_000)
    async with budget.summary() as spend:
        assert spend.bounds.model_s is None  # no single call has less than the whole summary
        await spend.model(done(ran()))
    assert budget.usage.model_calls == 1  # the view charges the call's own budget
    short = Budget(bounds=Bounds(summary_s=0.01))
    with pytest.raises(TimeoutError):
        async with short.summary() as spend:
            await spend.model(slow())


async def test_the_standard_summary_keeps_the_per_call_bound() -> None:
    budget = Budget.for_tier("standard")
    async with budget.summary() as spend:
        assert spend is budget
