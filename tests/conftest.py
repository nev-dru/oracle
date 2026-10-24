import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from oracle.config import REPO_ROOT, Config, load_config
from oracle.registry.loader import Registry, load_registry
from oracle.registry.schema import Tool

FAKES = REPO_ROOT / "tests" / "fakes" / "bin"
REGISTRY = REPO_ROOT / "profiles" / "home" / "tools"


@pytest.fixture
def home_cfg(tmp_path: Path) -> Config:
    """The home profile with an isolated ORACLE_HOME; no model configured."""
    return load_config({"ORACLE_HOME": str(tmp_path / "home"), "ORACLE_PROFILE": "home"})


@pytest.fixture
def fake_registry() -> Registry:
    """Only the fake-backed tools, so tests never need Ollama or the code graph."""
    return load_registry(REGISTRY, {"acli", "workiq"})


def script_tool(tmp_path: Path, body: str, **manifest: object) -> Tool:
    """A one-op tool whose binary is a Python script written from `body`."""
    script = tmp_path / "tool.py"
    script.write_text(textwrap.dedent(body))
    op = {
        "name": "run",
        "when": "test op",
        "argv": [sys.executable, str(script)],
        "returns": {"format": "json", "path": "$.items[*]", "id_field": "id"},
    }
    op.update(manifest.pop("op", {}))  # type: ignore[arg-type]
    data = {"name": "t", "description": "test tool", "ops": [op], **manifest}
    return Tool.model_validate(yaml.safe_load(yaml.safe_dump(data)))


def _final(info, body):  # type: ignore[no-untyped-def]
    from pydantic_ai.messages import ModelResponse, ToolCallPart

    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, body)])


def scripted(*outputs: dict):  # type: ignore[no-untyped-def]
    """A model answering with each output in turn (the last one repeats)."""
    from pydantic_ai.models.function import FunctionModel

    queue = list(outputs)

    def respond(messages, info):  # type: ignore[no-untyped-def]
        return _final(info, queue.pop(0) if len(queue) > 1 else queue[0])

    return FunctionModel(respond)


def prompt_text(messages) -> str:  # type: ignore[no-untyped-def]
    from pydantic_ai.messages import UserPromptPart

    return "\n".join(
        str(p.content) for m in messages for p in m.parts if isinstance(p, UserPromptPart)
    )


def echo_distiller(confidence: str = "high", limit: int = 15):  # type: ignore[no-untyped-def]
    """A distiller citing the first pointers it was shown, in order, one finding each."""
    import re

    from pydantic_ai.models.function import FunctionModel

    def respond(messages, info):  # type: ignore[no-untyped-def]
        pointers = re.findall(r"^  (\S+) \| ", prompt_text(messages), flags=re.M)
        findings = [{"text": f"found {p}", "pointer": p} for p in pointers[:limit]]
        return _final(info, {"findings": findings, "confidence": confidence, "gaps": [],
                             "refine": ['oracle "narrower question"']})  # fmt: skip

    return FunctionModel(respond)


def plan(*steps: dict, breadth: bool = False, expect: str = "") -> dict:
    return {"interpretation": "i", "steps": list(steps), "breadth": breadth, "expect": expect}


DONE = {"complete": True, "sufficient": True, "gaps": [], "next_steps": []}
