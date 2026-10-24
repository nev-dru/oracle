"""`claude-code:<alias>` models: the Claude subscription via the Claude Code CLI (claude-agent-sdk).

A PydanticAI FunctionModel, so agents stay plain PydanticAI. Each request is one Claude Code
process with no tools, no MCP servers and no settings. When the agent wants structured output,
the output tool's JSON schema goes to the CLI's `output_format` and the structured result comes
back as a call to that output tool, so PydanticAI validates it and retries as usual.
The oracle's agents never call function tools, so none are forwarded.
"""

import asyncio
import json
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryFeedbackPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

PREFIX = "claude-code:"
MAX_TURNS = 3  # one answer plus the CLI's structured-output turn, with slack
CALL_TIMEOUT_S = 120.0  # a hung `claude` process must not hang the main agent's Bash call
NO_THINKING = {"haiku"}
RETRY = "Your previous answer was rejected. Fix it and answer again:\n"


class ClaudeCodeError(RuntimeError):
    pass


def claude_code_model(alias: str) -> FunctionModel:
    async def call(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return await _complete(alias, messages, info)

    return FunctionModel(call, model_name=f"{PREFIX}{alias}")


async def _complete(alias: str, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    try:
        from claude_agent_sdk import ResultMessage, query
    except ImportError:  # pragma: no cover - depends on the install
        raise ClaudeCodeError(
            "claude-code models need the extra: uv sync --extra claude-code"
        ) from None
    output = info.output_tools[0] if info.output_tools else None
    result: Any = None
    try:
        async with asyncio.timeout(CALL_TIMEOUT_S):
            stream = query(prompt=transcript(messages), options=_options(alias, info, output))
            async for msg in stream:
                if isinstance(msg, ResultMessage):
                    result = msg
    except TimeoutError:
        raise ClaudeCodeError(
            f"claude-code {alias} gave no result in {CALL_TIMEOUT_S:g} s"
        ) from None
    if result is None or result.is_error:
        detail = getattr(result, "result", None) or "no result message"
        raise ClaudeCodeError(f"claude-code {alias} failed: {detail}")
    return ModelResponse(
        parts=[_part(result, output)], usage=_usage(result.usage), model_name=f"{PREFIX}{alias}"
    )


def _part(result: Any, output: ToolDefinition | None) -> TextPart | ToolCallPart:
    """Structured output becomes a call to the agent's output tool, so PydanticAI validates it."""
    if output is not None and result.structured_output is not None:
        return ToolCallPart(output.name, result.structured_output)
    return TextPart(result.result or "")


def _options(alias: str, info: AgentInfo, output: ToolDefinition | None) -> Any:
    from claude_agent_sdk import ClaudeAgentOptions, ThinkingConfigDisabled

    return ClaudeAgentOptions(
        model=alias,
        system_prompt=info.instructions or None,
        tools=[],
        allowed_tools=[],
        setting_sources=[],
        strict_mcp_config=True,
        max_turns=MAX_TURNS,
        output_format=(
            {"type": "json_schema", "schema": output.parameters_json_schema} if output else None
        ),
        # Haiku only extracts and formats here; thinking adds latency without better output.
        thinking=ThinkingConfigDisabled(type="disabled") if alias in NO_THINKING else None,
    )


def _usage(raw: dict[str, Any] | None) -> RequestUsage:
    """Cache reads and writes count as input, so token totals match what the model saw."""
    u = raw or {}
    cached = int(u.get("cache_read_input_tokens", 0)) + int(u.get("cache_creation_input_tokens", 0))
    return RequestUsage(
        input_tokens=int(u.get("input_tokens", 0)) + cached,
        output_tokens=int(u.get("output_tokens", 0)),
    )


def transcript(messages: list[ModelMessage]) -> str:
    """Flatten the conversation into one prompt; retries arrive as feedback to fix the answer."""
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            lines += [_request_part(p) for p in msg.parts]
        else:
            lines += [_response_part(p) for p in msg.parts]
    return "\n\n".join(line for line in lines if line)


def _request_part(part: Any) -> str:
    if isinstance(part, UserPromptPart):
        return part.content if isinstance(part.content, str) else str(part.content)
    if isinstance(part, ToolReturnPart) and part.outcome == "retried":
        return RETRY + str(part.content)
    if isinstance(part, RetryFeedbackPart):
        return RETRY + part.model_response()
    return ""


def _response_part(part: Any) -> str:
    if isinstance(part, TextPart):
        return f"(your previous answer)\n{part.content}"
    if isinstance(part, ToolCallPart):
        return f"(your previous answer)\n{json.dumps(part.args_as_dict())}"
    return ""
