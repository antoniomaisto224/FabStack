"""Scalar transforms between physical dimensions and display rows."""

import numpy as np

from fabstack.rendering.constants import (
    FEATURE_SCALE_COEFFICIENT,
    FEATURE_SCALE_EXPONENT,
    MAX_THICKNESS_NM,
    MIN_LAYER_PX,
    MIN_VOID_PX,
)


def _raw_feature_scale_um(thickness_um):
    """Continuous illustrative scale; the public helpers perform validation."""
    thickness_um = max(0.001, float(thickness_um))
    return float(FEATURE_SCALE_COEFFICIENT) * (
        thickness_um / 0.001
    ) ** float(FEATURE_SCALE_EXPONENT)


MAX_LAYER_PX = _raw_feature_scale_um(MAX_THICKNESS_NM / 1000.0)


def thickness_to_raw_visual_px(thickness_nm):
    """
    Map a positive physical thickness through the illustrative power law.

    1 nm is the readability endpoint, not a population-dependent floor.  The
    1 mm value is a reference rather than a cap; larger scenes can grow and
    scroll instead of silently flattening physical distinctions.
    """
    try:
        thickness_nm = float(thickness_nm)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(thickness_nm) or thickness_nm <= 0.0:
        return 0.0
    thickness_um = max(0.001, thickness_nm / 1000.0)
    return max(float(MIN_LAYER_PX), _raw_feature_scale_um(thickness_um))


def physical_span_to_raw_visual_px(thickness_um, *, minimum_px=MIN_VOID_PX):
    """Map an etched/empty physical span with the same law as solid layers."""
    try:
        thickness_um = float(thickness_um)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(thickness_um) or thickness_um <= 0.0:
        return 0.0
    return max(float(minimum_px), _raw_feature_scale_um(max(0.001, thickness_um)))


def _visual_height_to_display_rows(height_px: float) -> int:
    try:
        height_px = float(height_px)
    except (TypeError, ValueError):
        return 0
    if not np.isfinite(height_px) or height_px <= 0.0:
        return 0
    return max(1, int(np.ceil(height_px - 1e-9)))


__all__ = [
    "MAX_LAYER_PX",
    "_raw_feature_scale_um",
    "_visual_height_to_display_rows",
    "physical_span_to_raw_visual_px",
    "thickness_to_raw_visual_px",
]
