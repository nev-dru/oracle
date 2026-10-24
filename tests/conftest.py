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
