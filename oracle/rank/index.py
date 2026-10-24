"""Deterministic op ranking: exact name, then BM25, fused with reciprocal rank fusion.

One document per op; results are grouped by tool so the planner sees each selected tool's whole
op list (hierarchical tool -> op). No model call, so a wrong pick can be debugged from here.
Embeddings are a later stage (DESIGN.md: only once BM25 misses show up in ranker_eval).
A tool scores by its best op, so tools with many ops are not favoured for their size. The
shortlist is padded to `top` with unmatched tools, because domain words ("salesforce", a
person's name) rarely appear in tool docs, and a missing tool costs accuracy while an extra one
costs only planner tokens.
"""

import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from oracle.registry.loader import Registry

RRF_K = 60
TAG_WEIGHT = 2  # tags are the curated routing words; prose in `when` lines is incidental
EXACT_BONUS = 1.0
PREFER_BONUS = 0.5
# "who" stays: it routes to people sources (workiq is tagged `who`). The other interrogatives
# go, because tool docs use them as plain English ("when the exact name is unknown"). "x" goes
# because docs use it as a placeholder ("who is X").
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "why",
        "with",
        "x",
    ]
)


@dataclass
class RankedTool:
    tool: str
    score: float
    ops: list[str]
    matched_ops: list[str] = field(default_factory=list)
    pinned: bool = False


def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(w) for w in words if w not in STOPWORDS]


def _stem(word: str) -> str:
    """Crude plural folding so 'tickets' meets 'ticket'; enough for tool docs."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


class Ranker:
    def __init__(self, registry: Registry) -> None:
        self.registry = registry
        self.refs: list[str] = []
        docs: list[list[str]] = []
        for tool in registry.tools.values():
            for op in tool.ops:
                self.refs.append(f"{tool.name}.{op.name}")
                text = " ".join(
                    [tool.name, op.name, tool.description, " ".join(tool.tags * TAG_WEIGHT)]
                    + [tool.when, op.when]
                    + [f"{n} {p.desc}" for n, p in op.params.items()]
                    + [" ".join(str(v) for v in ex.params.values()) for ex in op.examples]
                )
                docs.append(tokens(text))
        self._bm25 = BM25Okapi(docs) if docs else None

    def rank(
        self,
        query: str,
        *,
        scope: list[str] | None = None,
        prefer: list[str] | None = None,
        top: int = 5,
    ) -> list[RankedTool]:
        if self._bm25 is None:
            return []
        fused, matched = self._bm25_by_tool(self._bm25, query)
        pinned = self._exact(query) | set(scope or [])
        fused = self._adjust(fused, pinned, prefer or [], scope, top)
        ranked = [
            RankedTool(
                tool=name,
                score=round(score, 4),
                ops=[f"{name}.{op.name}" for op in self.registry.tools[name].ops],
                matched_ops=matched.get(name, []),
                pinned=name in pinned,
            )
            for name, score in fused.items()
        ]
        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked[:top]

    def _adjust(
        self,
        fused: dict[str, float],
        pinned: set[str],
        prefer: list[str],
        scope: list[str] | None,
        top: int,
    ) -> dict[str, float]:
        """Exact-name and --prefer bonuses, the --scope filter, and padding to `top`."""
        out = dict(fused)
        for name in pinned:
            out[name] = out.get(name, 0.0) + EXACT_BONUS
        for name in prefer:
            out[name] = out.get(name, 0.0) + PREFER_BONUS
        allowed = [n for n in self.registry.tools if not scope or n in scope]
        out = {n: score for n, score in out.items() if n in allowed}
        for name in allowed:
            if len(out) >= top:
                break
            out.setdefault(name, 0.0)
        return out

    def _bm25_by_tool(
        self, bm25: BM25Okapi, query: str
    ) -> tuple[dict[str, float], dict[str, list[str]]]:
        """RRF score per tool from its best-ranked op, and the ops that matched at all."""
        q = tokens(query)
        scores = bm25.get_scores(q) if q else [0.0] * len(self.refs)
        order = sorted(range(len(self.refs)), key=lambda i: scores[i], reverse=True)
        fused: dict[str, float] = {}
        matched: dict[str, list[str]] = {}
        for rank, i in enumerate(i for i in order if scores[i] > 0):
            tool = self.refs[i].split(".", 1)[0]
            fused[tool] = max(fused.get(tool, 0.0), 1.0 / (RRF_K + rank + 1))
            matched.setdefault(tool, []).append(self.refs[i])
        return fused, matched

    def _exact(self, query: str) -> set[str]:
        """Tools named in the query, by tool name or a full op ref (`acli`, `jira.search`)."""
        q = query.lower()
        hits = set()
        for tool in self.registry.tools.values():
            names = [tool.name] + [op.name for op in tool.ops if "." in op.name]
            if any(re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", q) for n in names):
                hits.add(tool.name)
        return hits
