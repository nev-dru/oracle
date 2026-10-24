"""Model strings from config -> PydanticAI models. Model names live in config, never in code."""

from pydantic_ai.models import Model

from oracle.llm.claude_code import PREFIX, claude_code_model


def resolve_model(spec: str) -> Model | str:
    """`claude-code:sonnet` uses the subscription; anything else is a PydanticAI model string
    (`anthropic:...`, `openai:...` with OPENAI_BASE_URL pointing at LiteLLM, `bedrock:...`)."""
    if spec.startswith(PREFIX):
        return claude_code_model(spec.removeprefix(PREFIX))
    return spec
