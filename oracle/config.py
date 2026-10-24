"""~/.oracle/config.toml plus environment overrides, and the active environment profile.

Model names are never hard-coded: each role reads a PydanticAI model string from
ORACLE_MODEL_<ROLE> or `[models]` in config.toml. A profile picks the active manifests and
adjusts PATH and env for tool subprocesses; it may also point at its own registry directory.
Registry precedence: ORACLE_REGISTRY, then the profile's `registry`, then config.toml, then
the repo's registry/tools.
"""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
ROLES = ("planner", "extract", "deep")


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Profile:
    name: str
    registry: Path | None = None
    tools: frozenset[str] | None = None
    path_prepend: tuple[Path, ...] = ()
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Config:
    home: Path
    registry_dir: Path
    profile: Profile | None
    models: dict[str, str]
    timezone: str | None

    @property
    def log_path(self) -> Path:
        return self.home / "log.sqlite"

    @property
    def spill_dir(self) -> Path:
        return self.home / "spill"

    def model(self, role: str) -> str:
        if role not in self.models:
            var = f"ORACLE_MODEL_{role.upper()}"
            raise ConfigError(f"no model for {role}: set {var} or [models].{role} in config.toml")
        return self.models[role]

    def tool_env(self, base: dict[str, str] | None = None) -> dict[str, str]:
        """Environment for op subprocesses: the parent env plus the profile's PATH and env."""
        env = dict(os.environ if base is None else base)
        if self.profile:
            env.update(self.profile.env)
            prefix = os.pathsep.join(str(p) for p in self.profile.path_prepend)
            if prefix:
                env["PATH"] = prefix + os.pathsep + env.get("PATH", "")
        return env


def load_config(env: dict[str, str] | None = None) -> Config:
    env = dict(os.environ if env is None else env)
    home = Path(env.get("ORACLE_HOME", "~/.oracle")).expanduser()
    data = _read_toml(home / "config.toml")
    profile_name = _pick("ORACLE_PROFILE", "profile", env, data)
    profile = load_profile(profile_name, home) if profile_name else None
    registry = (
        env.get("ORACLE_REGISTRY")
        or (profile.registry if profile else None)
        or data.get("registry")
        or REPO_ROOT / "registry" / "tools"
    )
    return Config(
        home=home,
        registry_dir=Path(registry).expanduser(),
        profile=profile,
        models=_models(env, data),
        timezone=_pick("ORACLE_TZ", "timezone", env, data),
    )


def _pick(var: str, key: str, env: dict[str, str], data: dict[str, Any]) -> str | None:
    """Environment wins over config.toml."""
    return env.get(var) or data.get(key) or None


def _models(env: dict[str, str], data: dict[str, Any]) -> dict[str, str]:
    table = data.get("models") or {}
    models = {}
    for role in ROLES:
        spec = _pick(f"ORACLE_MODEL_{role.upper()}", role, env, table)
        if spec:
            models[role] = spec
    return models


def load_profile(name: str, home: Path) -> Profile:
    """A path, or a name looked up in ~/.oracle/profiles/ and then the repo's profiles/."""
    candidates = (
        [Path(name)]
        if name.endswith(".toml")
        else [
            home / "profiles" / f"{name}.toml",
            REPO_ROOT / "profiles" / f"{name}.toml",
        ]
    )
    for path in candidates:
        if path.is_file():
            return _parse_profile(path)
    raise ConfigError(f"profile {name!r} not found in {[str(c) for c in candidates]}")


def _parse_profile(path: Path) -> Profile:
    data = _read_toml(path)
    base = path.resolve().parent
    tools = data.get("tools")
    registry = data.get("registry")
    return Profile(
        name=path.stem,
        registry=(base / registry).resolve() if registry else None,
        tools=frozenset(tools) if tools is not None else None,
        path_prepend=tuple((base / p).resolve() for p in data.get("path_prepend", [])),
        env={
            k: str(v).replace("${PROFILE_DIR}", str(base)) for k, v in data.get("env", {}).items()
        },
    )


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from None
