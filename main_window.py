"""Compatibility facade for the packaged Qt main window."""

from fabstack.presentation.main_window import (
    HEIGHT_UM_DEFAULT,
    LEFT_PANEL_TOP_MAX_HEIGHT_PX,
    LEFT_PANEL_TOP_MIN_HEIGHT_PX,
    PROCESS_SUBSTRATE_VIEW_FRACTION,
    PROCESS_TOP_HEADROOM_UM,
    PROCESS_UM_PER_PX,
    WIDTH_UM_DEFAULT,
    CrossSectionCanvas,
    MainWindow,
)

__all__ = [
    "CrossSectionCanvas",
    "HEIGHT_UM_DEFAULT",
    "LEFT_PANEL_TOP_MAX_HEIGHT_PX",
    "LEFT_PANEL_TOP_MIN_HEIGHT_PX",
    "MainWindow",
    "PROCESS_SUBSTRATE_VIEW_FRACTION",
    "PROCESS_TOP_HEADROOM_UM",
    "PROCESS_UM_PER_PX",
    "WIDTH_UM_DEFAULT",
]
