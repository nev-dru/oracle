"""Skills for the oracle's own agents: short markdown files in skills/<name>/SKILL.md.

A skill reaches an agent three ways, and each agent run sees at most MAX_PER_RUN of them:
role-static (`scope: role`, always for the roles in `applies_to`), tool-scoped (`scope: tool:x`
or listed in a manifest's `skills:`, when that tool is in the shortlist), and query-ranked
(`scope: query`, the best match to the question above a threshold). The agents run in a fresh
context on every call, so a skill costs its tokens once per call and is never buried.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from oracle.rank.index import Lexical, tokens
from oracle.registry.loader import Registry

MAX_CHARS = 3200  # about 800 tokens
MAX_PER_RUN = 4
MAX_TOOL_SCOPED = 2


@dataclass(frozen=True)
class Skill:
    name: str
    applies_to: tuple[str, ...]
    scope: str
    when: str
    body: str

    @property
    def tool(self) -> str | None:
        return self.scope.removeprefix("tool:") if self.scope.startswith("tool:") else None


def load_skills(directory: Path) -> list[Skill]:
    """Every readable skill; a malformed one is skipped (and reported by `skill_problems`)."""
    out = []
    for path in sorted(directory.glob("*/SKILL.md")) if directory.is_dir() else []:
        try:
            out.append(parse_skill(path))
        except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError):
            continue
    return out


def parse_skill(path: Path) -> Skill:
    text = path.read_text(encoding="utf-8")
    head: dict = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) < 3:
            raise ValueError(f"{path}: frontmatter is not closed with ---")
        head, body = yaml.safe_load(parts[1]) or {}, parts[2]
        if not isinstance(head, dict):
            raise ValueError(f"{path}: frontmatter is not a mapping")
    return Skill(
        name=str(head.get("name", path.parent.name)),
        applies_to=tuple(head.get("applies_to", [])),
        scope=str(head.get("scope", "role")),
        when=str(head.get("when", "")),
        body=body.strip()[:MAX_CHARS],
    )


def skill_text(directory: Path, name: str) -> str:
    """One skill's body by name (the ingest drafter's role-static skill)."""
    return next((s.body for s in load_skills(directory) if s.name == name), "")


def select(
    skills: list[Skill], role: str, tools: list[str], registry: Registry, question: str
) -> list[Skill]:
    """Role-static first, then up to two for shortlisted tools (in shortlist order), then the
    best query-ranked match."""
    mine = [s for s in skills if role in s.applies_to]
    chosen = [s for s in mine if s.scope == "role"]
    chosen += _tool_scoped(mine, tools, registry)[:MAX_TOOL_SCOPED]
    chosen += _best_for(question, [s for s in mine if s.scope == "query"])
    return list(dict.fromkeys(chosen))[:MAX_PER_RUN]


def _tool_scoped(mine: list[Skill], tools: list[str], registry: Registry) -> list[Skill]:
    """Skills scoped to a shortlisted tool, or attached by its manifest's `skills:` list, one
    per tool first (in shortlist order), so a tool with two skills (acli: jql, cql) cannot take
    every slot from the tools ranked after it."""
    per_tool = [_for_tool(mine, tool, registry) for tool in tools]
    rounds = max((len(group) for group in per_tool), default=0)
    ordered = [group[i] for i in range(rounds) for group in per_tool if i < len(group)]
    return list(dict.fromkeys(ordered))


def _for_tool(mine: list[Skill], tool: str, registry: Registry) -> list[Skill]:
    attached = set(registry.tools[tool].skills) if tool in registry.tools else set()
    return [s for s in mine if s.tool == tool or (s.name in attached and s.tool in (None, tool))]


def _best_for(question: str, ranked: list[Skill]) -> list[Skill]:
    if not ranked:
        return []
    scores = Lexical([tokens(f"{s.name} {s.when}") for s in ranked]).scores(tokens(question))
    best = max(range(len(ranked)), key=lambda i: scores[i])
    return [ranked[best]] if scores[best] > 0 else []


def render(skills: list[Skill]) -> str:
    if not skills:
        return ""
    blocks = [f"### {s.name}\n{s.body}" for s in skills]
    return "Guidance for this question:\n\n" + "\n\n".join(blocks)
