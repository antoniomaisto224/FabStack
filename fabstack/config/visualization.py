"""Shared configuration for FabStack's physical-to-visual mapping.

Keeping these values in one module ensures that UI controls, persistence, and
rendering use identical limits and defaults.  The visualization scale is
illustrative only; physical dimensions remain stored in micrometers by the
process engine.
"""

PREVIEW_EXPORT_AS_MAIN_VIEW = True
EXPORT_PREVIEW_FIGSIZE = (6, 4)
EXPORT_PREVIEW_SUBPLOTS = {
    "left": 0.06,
    "right": 0.94,
    "top": 0.95,
    "bottom": 0.05,
}

VISUAL_SCALE_MIN = 0.5
VISUAL_SCALE_MAX = 1.5
VISUAL_SCALE_DEFAULT = 1.0

FEATURE_EMPHASIS_MIN = 0.25
FEATURE_EMPHASIS_MAX = 1.0
FEATURE_EMPHASIS_DEFAULT = 0.5


def clamp_visual_scale(value) -> float:
    """Return a valid global visualization scale."""
    return max(VISUAL_SCALE_MIN, min(VISUAL_SCALE_MAX, float(value)))


def clamp_feature_emphasis(value) -> float:
    """Return a valid per-project feature-emphasis value."""
    return max(FEATURE_EMPHASIS_MIN, min(FEATURE_EMPHASIS_MAX, float(value)))
