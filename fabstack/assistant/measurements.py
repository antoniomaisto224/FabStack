"""Measurement parsing primitives for the process planner."""

import re
from typing import List, Tuple

from fabstack.assistant.normalization import _coerce_float, _normalize_user_request


_UNIT_PATTERN = r"nm|nanometer|nanometers|um|\u00b5m|µm|micron|microns|mm"


_NUMBER_PATTERN = r"\d+(?:\.\d+)?"


def _first_explicit_unit(text: str) -> str:
    match = re.search(rf"\b(?:{_NUMBER_PATTERN})\s*({_UNIT_PATTERN})\b", _normalize_user_request(text), re.IGNORECASE)
    return match.group(1).lower() if match else ""


def _last_explicit_unit(text: str) -> str:
    unit = ""
    for match in re.finditer(rf"\b(?:{_NUMBER_PATTERN})\s*({_UNIT_PATTERN})\b", _normalize_user_request(text), re.IGNORECASE):
        unit = match.group(1).lower()
    return unit


def _unit_value_to_um(value: float, unit: str) -> float:
    unit = str(unit or "").lower()
    if unit in {"nm", "nanometer", "nanometers"}:
        return value / 1000.0
    if unit == "mm":
        return value * 1000.0
    return value


def _extract_measure_mentions_um(text: str, infer_unitless: bool = False) -> List[Tuple[float, int, int]]:
    text = _normalize_user_request(text)
    explicit_pattern = re.compile(rf"({_NUMBER_PATTERN})\s*({_UNIT_PATTERN})\b", re.IGNORECASE)
    values = []
    explicit_spans = []
    for match in explicit_pattern.finditer(text):
        value = _unit_value_to_um(float(match.group(1)), match.group(2))
        values.append((value, match.start(), match.end()))
        explicit_spans.append((match.start(), match.end()))

    if not infer_unitless:
        return values

    clause_boundary = re.compile(r"\b(?:then|after that)\b|[;\n]", re.IGNORECASE)
    starts = [0]
    ends = []
    for match in clause_boundary.finditer(text):
        ends.append(match.start())
        starts.append(match.end())
    ends.append(len(text))

    number_pattern = re.compile(rf"(?<![a-z0-9_.])({_NUMBER_PATTERN})(?![a-z0-9_.%])", re.IGNORECASE)
    for clause_start, clause_end in zip(starts, ends):
        clause = text[clause_start:clause_end]
        default_unit = _last_explicit_unit(clause)
        if not default_unit:
            continue
        for match in number_pattern.finditer(clause):
            abs_start = clause_start + match.start()
            abs_end = clause_start + match.end()
            if any(abs_start < span_end and abs_end > span_start for span_start, span_end in explicit_spans):
                continue
            if _number_span_is_identifier(text, abs_start, abs_end):
                continue
            value = _unit_value_to_um(float(match.group(1)), default_unit)
            values.append((value, abs_start, abs_end))

    values.sort(key=lambda item: item[1])
    return values


def _extract_unitless_measure_um(text: str, default_unit: str):
    if not default_unit:
        return None
    text = _normalize_user_request(text)
    explicit_spans = [
        (match.start(), match.end())
        for match in re.finditer(rf"({_NUMBER_PATTERN})\s*({_UNIT_PATTERN})\b", text, re.IGNORECASE)
    ]
    number_pattern = re.compile(rf"(?<![a-z0-9_.])({_NUMBER_PATTERN})(?![a-z0-9_.%])", re.IGNORECASE)
    for match in number_pattern.finditer(text):
        if any(match.start() < span_end and match.end() > span_start for span_start, span_end in explicit_spans):
            continue
        if _number_span_is_identifier(text, match.start(), match.end()):
            continue
        return _unit_value_to_um(float(match.group(1)), default_unit)
    return None


def _number_span_is_identifier(text: str, start: int, end: int) -> bool:
    text = str(text or "")
    before = text[max(0, start - 8):start].lower()
    after = text[end:min(len(text), end + 8)].lower()
    token_start = start
    while token_start > 0 and re.match(r"[a-z0-9_+\-.]", text[token_start - 1], re.IGNORECASE):
        token_start -= 1
    token_end = end
    while token_end < len(text) and re.match(r"[a-z0-9_+\-.]", text[token_end], re.IGNORECASE):
        token_end += 1
    token = text[token_start:token_end].lower().strip(".,;:()[]{}")

    if re.search(r"(?:su|az|hsq|dnq|pr_?|si|sio|si3n|aln|sic|tin|cr|au|ti|al)[-_+\s]*$", before):
        return True
    if re.match(r"^[-_+\s]*(?:series|developer|photoresist|resist|oxide|nitride|carbide)\b", after):
        return True
    if re.search(r"[a-z]", token) and re.search(r"\d", token):
        return True
    return False


def _extract_measure_um(text: str):
    measures = _extract_measure_mentions_um(text, infer_unitless=False)
    if not measures:
        return None
    return measures[0][0]


def _extract_all_measure_um(text: str) -> List[float]:
    return [value for value, _start, _end in _extract_measure_mentions_um(text, infer_unitless=True)]


def _extract_substrate_like_unitless_um(text: str):
    low = _normalize_user_request(text).lower()
    if not re.search(r"\b(wafer|substrate)\b", low):
        return None
    explicit_spans = [
        (match.start(), match.end())
        for match in re.finditer(rf"({_NUMBER_PATTERN})\s*({_UNIT_PATTERN})\b", low, re.IGNORECASE)
    ]
    for match in re.finditer(rf"(?<![a-z0-9_.])({_NUMBER_PATTERN})(?![a-z0-9_.%])", low, re.IGNORECASE):
        if any(match.start() < span_end and match.end() > span_start for span_start, span_end in explicit_spans):
            continue
        value = _coerce_float(match.group(1))
        if value is not None and value > 1.0:
            return float(value)
    return None


def _extract_measure(text: str) -> str:
    text = _normalize_user_request(text)
    pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*(nm|nanometer|nanometers|um|\u00b5m|micron|microns|mm)\b",
        flags=re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        return ""
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit in {"nm", "nanometer", "nanometers"}:
        um = value / 1000.0
        return f"{um:g} µm ({value:g} nm)"
    if unit == "mm":
        um = value * 1000.0
        return f"{um:g} µm ({value:g} mm)"
    return f"{value:g} µm"

__all__ = [
    "_NUMBER_PATTERN",
    "_UNIT_PATTERN",
    "_extract_all_measure_um",
    "_extract_measure",
    "_extract_measure_mentions_um",
    "_extract_measure_um",
    "_extract_substrate_like_unitless_um",
    "_extract_unitless_measure_um",
    "_first_explicit_unit",
    "_last_explicit_unit",
    "_number_span_is_identifier",
    "_unit_value_to_um",
]
