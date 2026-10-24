"""Regex redaction of secrets in op results, before they are cached, logged or shown to a model.

Env-var auth keeps secrets out of argv, but a config dump or `kubectl get secret` can still
return them. This is a net for the common shapes, not a guarantee.
"""

import re
from typing import Any

MASK = "[REDACTED]"
PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),  # GitHub tokens
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),  # Slack tokens
    # sk- style API keys: long, with digits and capitals, unlike a lower-case slug
    re.compile(r"\bsk-(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[A-Z])[A-Za-z0-9_-]{32,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{20,}"),
    re.compile(r"\b[A-Za-z][A-Za-z0-9+.-]*://[^:/\s]+:[^@/\s]+@"),  # user:password@ in URLs
]
# `password=...`: the value must look like a credential (a digit or symbol, 6+ chars), so prose
# such as "secret: rotate the key" is left alone
KEY_VALUE = re.compile(
    r"(?i)\b(password|passwd|secret|api[_-]?key|access[_-]?token|client[_-]?secret)"
    r"(\s*[:=]\s*)(\"[^\"]+\"|'[^']+'|(?=\S*[\d!@#$%^&*+/=_-])\S{6,})"
)


def redact_text(text: str) -> str:
    for pattern in PATTERNS:
        text = pattern.sub(MASK, text)
    return KEY_VALUE.sub(lambda m: f"{m[1]}{m[2]}{MASK}", text)


def redact(value: Any) -> Any:
    """Redact every string inside a JSON-like value; keys are left alone."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value
