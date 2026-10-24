"""Validate and coerce op params against a manifest; the planner and executor share this check."""

import json
from typing import Any

from oracle.registry.schema import Op, Param


class ParamError(ValueError):
    pass


def validate_params(op: Op, raw: dict[str, Any]) -> dict[str, Any]:
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
