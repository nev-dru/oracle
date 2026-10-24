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


_COERCE = {"str": str, "int": _int, "bool": _bool, "list[str]": _list, "json": _json}


def _check(name: str, spec: Param, value: Any) -> Any:
    if spec.enum is not None and value not in spec.enum:
        raise ParamError(f"param {name!r} must be one of {spec.enum}, got {value!r}")
    if spec.max is not None and isinstance(value, int) and value > spec.max:
        raise ParamError(f"param {name!r} must be at most {spec.max}, got {value}")
    if isinstance(value, str) and not value.strip():
        raise ParamError(f"param {name!r} is empty")
    return value


TEMPLATE = re.compile(r"^\$|\{[a-z_]+\}")
_DUMMY: dict[str, Any] = {"str": "x", "int": 1, "bool": True, "list[str]": ["x"], "json": {}}


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
