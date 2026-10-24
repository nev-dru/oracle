"""`oracle tool test --ranker`: top-1 and top-3 hit rates on ranker_eval.yaml.

A hit counts only tools the ranker matched (score > 0) or pinned, never padding: with a small
registry every tool is shortlisted, and a padded hit says nothing about the ranker.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from oracle.rank.index import Ranker
from oracle.registry.loader import Registry

GATE = 0.9


@dataclass
class RankerCase:
    query: str
    expect: list[str]
    scored: list[str] = field(default_factory=list)

    @property
    def top1(self) -> bool:
        return bool(self.scored) and self.scored[0] in self.expect

    @property
    def top3(self) -> bool:
        return set(self.expect) <= set(self.scored[:3])


@dataclass
class RankerReport:
    cases: list[RankerCase]
    skipped: list[str]

    def rate(self, attr: str) -> float:
        return sum(getattr(c, attr) for c in self.cases) / len(self.cases) if self.cases else 0.0

    @property
    def passed(self) -> bool:
        return bool(self.cases) and self.rate("top3") >= GATE

    def lines(self) -> list[str]:
        out = [
            f"MISS  {c.query!r}: expected {c.expect}, top-3 matched {c.scored[:3]}"
            for c in self.cases
            if not c.top3
        ]
        out.append(
            f"top-1 {self.rate('top1'):.0%}  top-3 {self.rate('top3'):.0%} "
            f"over {len(self.cases)} cases (gate {GATE:.0%}); "
            f"{len(self.skipped)} skipped: tools not registered here"
        )
        return out


def run_ranker_eval(registry: Registry, path: Path) -> RankerReport:
    data = yaml.safe_load(path.read_text()) or {}
    ranker = Ranker(registry)
    cases, skipped = [], []
    for raw in data.get("cases", []):
        if not set(raw["expect"]) <= set(registry.tools):
            skipped.append(raw["query"])
            continue
        ranked = ranker.rank(raw["query"])
        scored = [t.tool for t in ranked if t.score > 0 or t.pinned]
        cases.append(RankerCase(raw["query"], list(raw["expect"]), scored))
    return RankerReport(cases, skipped)
