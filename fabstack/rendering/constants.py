"""Stable numeric constants shared by display and export rendering."""

DISPLAY_TOP_HEADROOM_PX = 12
DYNAMIC_TARGET_STACK_PX = 700
MAX_AUTO_CANVAS_HEIGHT_SCALE = 6.0
VISUAL_RENDER_SCALE_BASE = 0.5

WORD_EXPORT_IMAGE_WIDTH_IN = 7.1
WORD_EXPORT_DPI = 150
WORD_EXPORT_VERTICAL_PAD_PX = 8
WORD_EXPORT_HORIZONTAL_PAD_IN = 0.08
WORD_EXPORT_PAGE_WIDTH_IN = 8.27
WORD_EXPORT_PAGE_HEIGHT_IN = 11.69
WORD_EXPORT_PAGE_MARGIN_IN = 0.55

# A single calibrated fifth-root law is used for every process feature.  A
# linear scale cannot show 1 nm and 1 mm in one finite overview: either the
# nano-film disappears or the substrate becomes enormous.  This monotone law
# makes all six decades distinct while preserving their physical order.  Its
# coefficient is calibrated so the common 525 um wafer is 175 display rows --
# about two thirds of the former 263-row body.
#
# Atomic layers and etched substrate depths are resolved scene-wide.  Explicit
# layer/stack-versus-gap comparisons receive an eight-row separation (four
# visible pixels at the default 0.5 render transform).  The renderer may grow
# the canvas for a dense stack, but it never invents a different trench bottom
# merely because a neighbouring column contains different films.
# Manual display-height edits use visual-only overrides; automatic display
# mode ignores legacy overrides.
MIN_THICKNESS_NM = 1.0            # 1 nm
MAX_THICKNESS_NM = 1_000_000.0    # 1000 um
FEATURE_SCALE_EXPONENT = 0.20
FEATURE_SCALE_COEFFICIENT = 175.0 / ((525.0 / 0.001) ** FEATURE_SCALE_EXPONENT)
SUBSTRATE_ROWS_PER_UM = 1.0 / 3.0
MIN_LAYER_PX = 13.0
MIN_VOID_PX = MIN_LAYER_PX
ORDER_TIE_RATIO = 1.05
ORDER_MARGIN_PX = 8
MIN_SUBSTRATE_CORE_PX = 13
MAX_CONSTRAINT_PASSES = 512

__all__ = [
    "DISPLAY_TOP_HEADROOM_PX",
    "DYNAMIC_TARGET_STACK_PX",
    "FEATURE_SCALE_COEFFICIENT",
    "FEATURE_SCALE_EXPONENT",
    "MAX_AUTO_CANVAS_HEIGHT_SCALE",
    "MAX_CONSTRAINT_PASSES",
    "MAX_THICKNESS_NM",
    "MIN_LAYER_PX",
    "MIN_SUBSTRATE_CORE_PX",
    "MIN_THICKNESS_NM",
    "MIN_VOID_PX",
    "ORDER_MARGIN_PX",
    "ORDER_TIE_RATIO",
    "SUBSTRATE_ROWS_PER_UM",
    "VISUAL_RENDER_SCALE_BASE",
    "WORD_EXPORT_DPI",
    "WORD_EXPORT_HORIZONTAL_PAD_IN",
    "WORD_EXPORT_IMAGE_WIDTH_IN",
    "WORD_EXPORT_PAGE_HEIGHT_IN",
    "WORD_EXPORT_PAGE_MARGIN_IN",
    "WORD_EXPORT_PAGE_WIDTH_IN",
    "WORD_EXPORT_VERTICAL_PAD_PX",
]
