"""Stable aggregate facade for plotting, canvas, visual adjustment, and export."""

from fabstack.config.visualization import (
    EXPORT_PREVIEW_SUBPLOTS,
    FEATURE_EMPHASIS_DEFAULT,
    FEATURE_EMPHASIS_MAX,
    FEATURE_EMPHASIS_MIN,
    PREVIEW_EXPORT_AS_MAIN_VIEW,
    VISUAL_SCALE_MAX,
    VISUAL_SCALE_MIN,
)
from fabstack.presentation.mixins.canvas_rendering_mixin import CanvasRenderingMixin
from fabstack.presentation.mixins.display_profile_mixin import DisplayProfileMixin
from fabstack.presentation.mixins.export_actions_mixin import ExportActionsMixin
from fabstack.presentation.mixins.visual_adjustment_mixin import VisualAdjustmentMixin
from fabstack.rendering.constants import (
    DISPLAY_TOP_HEADROOM_PX,
    DYNAMIC_TARGET_STACK_PX,
    FEATURE_SCALE_COEFFICIENT,
    FEATURE_SCALE_EXPONENT,
    MAX_AUTO_CANVAS_HEIGHT_SCALE,
    MAX_CONSTRAINT_PASSES,
    MAX_THICKNESS_NM,
    MIN_LAYER_PX,
    MIN_SUBSTRATE_CORE_PX,
    MIN_THICKNESS_NM,
    MIN_VOID_PX,
    ORDER_MARGIN_PX,
    ORDER_TIE_RATIO,
    SUBSTRATE_ROWS_PER_UM,
    VISUAL_RENDER_SCALE_BASE,
    WORD_EXPORT_DPI,
    WORD_EXPORT_HORIZONTAL_PAD_IN,
    WORD_EXPORT_IMAGE_WIDTH_IN,
    WORD_EXPORT_PAGE_HEIGHT_IN,
    WORD_EXPORT_PAGE_MARGIN_IN,
    WORD_EXPORT_PAGE_WIDTH_IN,
    WORD_EXPORT_VERTICAL_PAD_PX,
)
from fabstack.rendering.scaling import (
    MAX_LAYER_PX,
    _raw_feature_scale_um,
    _visual_height_to_display_rows,
    physical_span_to_raw_visual_px,
    thickness_to_raw_visual_px,
)


class PlottingExportMixin(
    DisplayProfileMixin,
    VisualAdjustmentMixin,
    CanvasRenderingMixin,
    ExportActionsMixin,
):
    """Compatibility aggregate preserving MainWindow's historical direct base."""

    pass


__all__ = [
    "CanvasRenderingMixin",
    "DISPLAY_TOP_HEADROOM_PX",
    "DYNAMIC_TARGET_STACK_PX",
    "DisplayProfileMixin",
    "EXPORT_PREVIEW_SUBPLOTS",
    "ExportActionsMixin",
    "FEATURE_EMPHASIS_DEFAULT",
    "FEATURE_EMPHASIS_MAX",
    "FEATURE_EMPHASIS_MIN",
    "FEATURE_SCALE_COEFFICIENT",
    "FEATURE_SCALE_EXPONENT",
    "MAX_AUTO_CANVAS_HEIGHT_SCALE",
    "MAX_CONSTRAINT_PASSES",
    "MAX_LAYER_PX",
    "MAX_THICKNESS_NM",
    "MIN_LAYER_PX",
    "MIN_SUBSTRATE_CORE_PX",
    "MIN_THICKNESS_NM",
    "MIN_VOID_PX",
    "ORDER_MARGIN_PX",
    "ORDER_TIE_RATIO",
    "PREVIEW_EXPORT_AS_MAIN_VIEW",
    "PlottingExportMixin",
    "SUBSTRATE_ROWS_PER_UM",
    "VISUAL_RENDER_SCALE_BASE",
    "VISUAL_SCALE_MAX",
    "VISUAL_SCALE_MIN",
    "VisualAdjustmentMixin",
    "WORD_EXPORT_DPI",
    "WORD_EXPORT_HORIZONTAL_PAD_IN",
    "WORD_EXPORT_IMAGE_WIDTH_IN",
    "WORD_EXPORT_PAGE_HEIGHT_IN",
    "WORD_EXPORT_PAGE_MARGIN_IN",
    "WORD_EXPORT_PAGE_WIDTH_IN",
    "WORD_EXPORT_VERTICAL_PAD_PX",
    "_raw_feature_scale_um",
    "_visual_height_to_display_rows",
    "physical_span_to_raw_visual_px",
    "thickness_to_raw_visual_px",
]

