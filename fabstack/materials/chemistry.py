"""Utility helpers for chemical text formatting and legend labels for FabStack.

This module keeps all chemistry-related string helpers in one place so they can
be reused by both the GUI and any future tools or scripts.
"""

import re

# ---------- Chem text helpers (pretty subscripts in UI, ASCII internally) ----------
_SUPS_UI = str.maketrans("0123456789+-()", "₀₁₂₃₄₅₆₇₈₉₊₋₍₎")
_SUPS_BACK = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

def chem_subs_unicode(s: str) -> str:
    """Render digits as Unicode subscripts only when right after a letter/paren."""
    if not s:
        return s
    def repl(m: re.Match) -> str:
        return m.group(1) + m.group(2).translate(_SUPS_UI)
    return re.sub(r"([A-Za-z\)\]\}])(\d+)", repl, s)

def chem_normalize_ascii(s: str) -> str:
    """Convert any Unicode subscript digits back to ASCII digits for storage/use."""
    return (s or "").translate(_SUPS_BACK)

def legend_label(name: str) -> str:
    """If name contains an acronym in parentheses, return only that for legend."""
    if "(" in name and ")" in name:
        start = name.rfind("(")
        end = name.rfind(")")
        if 0 <= start < end:
            inner = name[start+1:end].strip()
            if inner:
                return inner
    return name
