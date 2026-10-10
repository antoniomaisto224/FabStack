"""Compatibility facade for chemical-name formatting helpers."""

from fabstack.materials.chemistry import (
    chem_normalize_ascii,
    chem_subs_unicode,
    legend_label,
)

__all__ = ["chem_normalize_ascii", "chem_subs_unicode", "legend_label"]
