"""Small normalization helpers shared by the process planner facade."""

import re
from typing import Iterable, List


def _user_facing_detail(value) -> str:
    """Remove planner schema and backend terms from a clarification prompt."""
    text = str(value or "").strip()
    replacements = (
        (r"\bthickness_um\b", "thickness"),
        (r"\bdepth_um\b", "etch depth"),
        (r"\btemperature_c\b", "temperature"),
        (r"\btime_s\b", "time"),
        (r"\bwindowing_mode\b", "lithography mode"),
        (r"\binvert_lithography\b", "inverted lithography"),
        (r"\bcommand compiler\b", "process assistant"),
        (r"\bcommands\b", "steps"),
        (r"\bcommand\b", "step"),
        (r"\bJSON\b", "process plan"),
        (r"\bmodel\b", "assistant"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _normalize_user_request(text: str) -> str:
    text = str(text or "").strip()
    return re.sub(r"(?<=\d),(?=\d)", ".", text)


def _unique(values: Iterable[str]) -> List[str]:
    out = []
    seen = set()
    for value in values:
        txt = str(value or "").strip()
        key = txt.lower()
        if not txt or key in seen:
            continue
        seen.add(key)
        out.append(txt)
    return out


def _coerce_float(value):
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not (out == out):
        return None
    return out


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    txt = str(value).strip().lower()
    if txt in {"true", "yes", "y", "1", "invert", "inverted", "inverse"}:
        return True
    if txt in {"false", "no", "n", "0", "normal", "positive"}:
        return False
    return None


def _measure_was_requested(value: float, requested_values: List[float]) -> bool:
    if value is None:
        return False
    for requested in requested_values:
        if requested is None:
            continue
        tolerance = max(1e-9, abs(requested) * 1e-6)
        if abs(float(value) - float(requested)) <= tolerance:
            return True
    return False

__all__ = [
    "_coerce_bool",
    "_coerce_float",
    "_measure_was_requested",
    "_normalize_user_request",
    "_unique",
    "_user_facing_detail",
]
