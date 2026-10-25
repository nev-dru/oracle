"""The crawl's extractor (Haiku): from a hop's new results, the terms worth searching next, of
any kind, described in plain words. No list of entity types exists anywhere: the description is
what routes a term to a tool."""

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model

from oracle.budget import Budget
from oracle.exec.dag import StepRun
from oracle.plan.digest import DATA_RULE, defang, digest
from oracle.plan.planner import prompt
from oracle.registry.loader import Registry

MAX_TERMS = 20
SLICE_CHARS = 200


class Term(BaseModel):
    value: str = Field(max_length=160, description="the term exactly as it appears")
    description: str = Field(max_length=240, description="what it is and where it came from")
    relevance: float = Field(ge=0, le=1)


class Terms(BaseModel):
    terms: list[Term] = Field(default_factory=list, max_length=MAX_TERMS)


def build_extractor(model: Model | str) -> Agent[None, Terms]:
    return Agent(
        model, output_type=Terms, instructions=prompt("extract") + "\n" + DATA_RULE, retries=2
    )


async def extract_terms(
    agent: Agent[None, Terms],
    *,
    question: str,
    runs: list[StepRun],
    seen: set[str],
    registry: Registry,
    budget: Budget,
) -> list[Term]:
    """New terms, best first; a term already searched (case-insensitively) is dropped."""
    lines = [f"Question: {question}"]
    if seen:
        lines.append("Already searched (do not return): " + defang("; ".join(sorted(seen))[:2000]))
    # summary fields alone hide what a crawl follows: a search result's author is in the slice
    lines += ["", digest(runs, registry, per_step=12, slice_chars=SLICE_CHARS)]
    result = await budget.model(agent, "\n".join(lines))
    fresh = [t for t in result.output.terms if key(t.value) not in seen and t.value.strip()]
    return sorted(fresh, key=lambda t: t.relevance, reverse=True)


def key(value: str) -> str:
    return " ".join(value.lower().split())
