"""Validate and coerce op params against a manifest; the planner and executor share this check."""

import json
import re
from typing import Any, Protocol

from oracle.registry.schema import Param


class ParamError(ValueError):
    pass


class HasParams(Protocol):
    """An op or a recipe: anything with a name and declared params."""

    @property
    def name(self) -> str: ...

    @property
    def params(self) -> dict[str, Param]: ...


def validate_params(op: HasParams, raw: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(raw) - set(op.params))
    if unknown:
        raise ParamError(f"unknown params {unknown}; {op.name} takes {sorted(op.params)}")
    out: dict[str, Any] = {}
    for name, spec in op.params.items():
        value = raw.get(name)
        if value is None:  # absent and explicit null both mean "use the default"
            value = spec.default
        if value is None:
            if spec.required:
                raise ParamError(f"missing required param {name!r}")
            out[name] = None
            continue
        out[name] = _check(name, spec, _coerce(name, spec, value))
    return out


def _coerce(name: str, spec: Param, value: Any) -> Any:
    try:
        return _COERCE[spec.type](value)
    except (TypeError, ValueError):
        raise ParamError(f"param {name!r} must be {spec.type}, got {value!r}") from None


def _int(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError(value)
    return int(value)


def _list(value: Any) -> list[str]:
    items = value.split(",") if isinstance(value, str) else list(value)
    return [str(v).strip() for v in items if str(v).strip()]


def _json(value: Any) -> dict[str, Any]:
    obj = json.loads(value) if isinstance(value, str) else value
    if not isinstance(obj, dict):
        raise TypeError(value)
    return obj


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("true", "1", "yes"):
        return True
    if str(value).lower() in ("false", "0", "no"):
        return False
    raise ValueError(value)


def _float(value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError(value)
    return float(value)


_COERCE = {
    "str": str,
    "int": _int,
    "float": _float,
    "bool": _bool,
    "list[str]": _list,
    "json": _json,
}


# `kubectl get secrets` and friends return credentials; no op may ask for them by name
DENIED_VALUES = frozenset({"secret", "secrets"})


CODE_OPERATORS = frozenset({"$where", "$function", "$accumulator", "$eval"})


def _denied(value: Any) -> bool:
    """A secrets resource named in a short identifier (`secret`, `secret/db-creds`,
    `secrets,configmaps`, `Secret.v1`). Prose with spaces is data, so a search for "where do
    we store secrets" passes."""
    values = value if isinstance(value, list) else [value]
    return any(
        isinstance(v, str)
        and len(v.split()) == 1
        and bool(set(re.split(r"[/,.:]", v.strip().lower())) & DENIED_VALUES)
        for v in values
    )


def _code_operator(value: Any) -> str | None:
    """Mongo-style operators that run server-side code, anywhere in a JSON param."""
    if isinstance(value, dict):
        for key, inner in value.items():
            if key in CODE_OPERATORS:
                return key
            if found := _code_operator(inner):
                return found
    if isinstance(value, list):
        return next((f for v in value if (f := _code_operator(v))), None)
    return None


def _refuse_unsafe(name: str, value: Any) -> None:
    """Values no op may send, whatever its manifest says: secrets and server-side code."""
    if _denied(value):
        raise ParamError(f"param {name!r}: reading secrets is refused")
    if op := _code_operator(value):
        raise ParamError(f"param {name!r}: {op} runs code on the server and is refused")


def _text(value: Any) -> str:
    """How an enum value is written in a manifest: `true`, `2`, `callers`."""
    return str(value).lower() if isinstance(value, bool) else str(value)


def _check(name: str, spec: Param, value: Any) -> Any:
    _refuse_unsafe(name, value)
    if spec.enum is not None and value not in spec.enum and _text(value) not in spec.enum:
        raise ParamError(f"param {name!r} must be one of {spec.enum}, got {value!r}")
    if spec.max is not None and isinstance(value, int) and value > spec.max:
        raise ParamError(f"param {name!r} must be at most {spec.max}, got {value}")
    if isinstance(value, str) and not value.strip():
        raise ParamError(f"param {name!r} is empty")
    return value


TEMPLATE = re.compile(r"^\$|\{[a-z_]+\}")
_DUMMY: dict[str, Any] = {
    "str": "x",
    "int": 1,
    "float": 1.0,
    "bool": True,
    "list[str]": ["x"],
    "json": {},
}


def is_deferred(value: Any) -> bool:
    """A binding (`$s1.ids`) or a template (`assignee = "{name}"`): known only at run time."""
    return isinstance(value, str) and bool(TEMPLATE.search(value))


def validate_shape(op: HasParams, raw: dict[str, Any]) -> dict[str, Any]:
    """validate_params with deferred values stood in for by a value of the declared type, so a
    plan or recipe is checked for names, required params and literal values before it runs."""
    stand_in = {
        k: _stand_in(op.params[k]) if k in op.params and is_deferred(v) else v
        for k, v in raw.items()
    }
    return validate_params(op, stand_in)


def _stand_in(spec: Param) -> Any:
    """A valid value of the declared type: the first enum value, or the default, or a dummy."""
    if spec.enum:
        return spec.enum[0]
    return spec.default if spec.default is not None else _DUMMY[spec.type]
