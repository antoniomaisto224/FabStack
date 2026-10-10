"""Main Qt application window for FabStack.

Contains the PyQt5 main window and plotting logic. It relies on:
- ``fabstack.domain.CrossSection`` for the geometry/stack engine
- ``fabstack.materials.chemistry`` for chemical text formatting
"""

import json
import logging
import os
import math

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSlider, QDoubleSpinBox, QFormLayout,
    QGridLayout, QHBoxLayout, QInputDialog, QColorDialog, QMessageBox, QLineEdit, QAction, QDialog,
    QVBoxLayout, QDialogButtonBox, QFontDialog
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QIcon, QKeySequence, QFont, QColor

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.font_manager as mpl_font_manager

from fabstack.resources import (
    APP_ICON_PATH,
    LOGO_PATH,
    MATERIALS_LIBRARY_PATH,
    METHOD_DEFAULTS_PATH,
    METHOD_RECIPES_PATH,
    PARAM_DEFAULTS_PATH,
)
import numpy as np

from fabstack.presentation.mixins import (
    LlmAssistantMixin,
    MaterialsLibraryMixin,
    OperationsMixin,
    PlottingExportMixin,
    ProcessPersistenceMixin,
    ProjectTabsMixin,
    UiLayoutMixin,
    UiThemeMixin,
)
from fabstack.domain.cross_section import CrossSection
from fabstack.materials.chemistry import chem_subs_unicode, chem_normalize_ascii
from fabstack.operations.execution import (
    apply_process_step_to_current_xsec,
    execute_ui_flip_operation,
)
from fabstack.config.visualization import (
    EXPORT_PREVIEW_FIGSIZE,
    FEATURE_EMPHASIS_DEFAULT,
    FEATURE_EMPHASIS_MAX,
    FEATURE_EMPHASIS_MIN,
    PREVIEW_EXPORT_AS_MAIN_VIEW,
    VISUAL_SCALE_MAX,
    VISUAL_SCALE_MIN,
)
from fabstack.terminology import operation_label

# fixed wafer size defaults (µm)
WIDTH_UM_DEFAULT = 300
HEIGHT_UM_DEFAULT = 2500
PROCESS_UM_PER_PX = 1.0
PROCESS_TOP_HEADROOM_UM = 50.0
PROCESS_SUBSTRATE_VIEW_FRACTION = 0.25
LOGGER = logging.getLogger(__name__)
# ---- left panel placement for preview mode ----
LEFT_PANEL_TOP_MAX_HEIGHT_PX = 140
LEFT_PANEL_TOP_MIN_HEIGHT_PX = 140

class CrossSectionCanvas(FigureCanvas):
    def __init__(self, parent=None, figsize=None):
        if figsize is None:
            figsize = EXPORT_PREVIEW_FIGSIZE if PREVIEW_EXPORT_AS_MAIN_VIEW else (6, 8)
        self.fig = Figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)


class MainWindow(
    UiThemeMixin,
    UiLayoutMixin,
    LlmAssistantMixin,
    MaterialsLibraryMixin,
    OperationsMixin,
    PlottingExportMixin,
    ProcessPersistenceMixin,
    ProjectTabsMixin,
    QMainWindow,
):
    def __init__(self):
        super().__init__()
        # Global modern style (auto-adapts to current light/dark palette).
        self._apply_modern_theme()

        # Set window icon: always use dedicated app icon (iconlogo.png)
        if os.path.exists(APP_ICON_PATH):
            self.setWindowIcon(QIcon(APP_ICON_PATH))

        # Application identity.
        self.base_title = "FabStack"
        self.setWindowTitle(self.base_title)
        self._set_startup_window_size()

        # materials
        self.material_names = ["Void", "Silicon (Si)"]
        self.material_colors = ["#ffffff", "#b0b0b0"]
        self.VOID_INDEX = 0

        # Track materials "deleted" from the current file UI (soft delete)
        # This avoids breaking internal indexing/history grids.
        self.deleted_materials = set()

        # Recently used materials (names only, most recent first)
        self.recent_materials = []

        # Categories hidden from "Add from library" material selections
        # (but still available for methods / etchants logic)
        self.hidden_material_categories = set()

        # library (with categories)
        self.library_path = MATERIALS_LIBRARY_PATH
        self.material_library_by_cat = {}
        self.material_library_flat = []

        # compatibility maps loaded from JSON
        self.compat_etch_by_material = {}
        self.compat_etch_by_category = {}
        self.compat_deposit_by_material = {}
        self.compat_deposit_by_category = {}
        # additional compatibility maps for future extensions (bond/strip/clean)
        self.compat_bond_by_material = {}
        self.compat_bond_by_category = {}
        self.compat_strip_by_material = {}
        self.compat_strip_by_category = {}
        self.compat_liftoff_by_material = {}
        self.compat_liftoff_by_category = {}
        self.compat_clean_by_material = {}
        self.compat_clean_by_category = {}
        self.operation_restrictions_by_operation = {}

        # etch type modes by etchant name (from library)
        # keys stored in normalized ASCII (chem_normalize_ascii)
        self.etch_modes_by_etchant = {}
        # current etch mode used when applying an Etch step
        self.current_etch_mode = "anisotropic"

        self.load_material_library()
        LOGGER.info("Using material library: %s", self.library_path)
        LOGGER.debug(
            "Loaded material categories: %s",
            list(self.material_library_by_cat.keys()),
        )

        # state
        self.history = []
        self.process_steps = []
        self.process_steps_replay_enabled = False
        self._substrate_change_guard = 0
        self.redo_stack = []  # for Edit → Redo
        self.current_title = ""
        self.litho_windows = []
        # Optional external process selected for Bond operation.
        self.bond_source_info = None

        # expert user mode flag (library manager / operation parameters)
        self.is_expert_user = False

        # floating dialogs
        self.window_dialog = None
        self.op_params_dialog = None
        self.text_options_dialog = None
        self.process_flow_dialog = None

        # parameter classes (for future descriptive use)
        # Initialize from persistent defaults (Temperature/Time if none saved).
        self.parameter_classes = self.load_default_parameter_classes()
        # Persisted method defaults and named recipes (global, across sessions).
        self.default_method_settings = self.load_default_method_settings()
        self.method_recipes = self.load_method_recipes()
        # widgets for custom parameter values in the Parameters tab
        self.custom_param_widgets = {}

        # file save/load state
        self.current_process_path = None
        self.project_sessions = []  # one session per tab
        self.active_project_tab_index = -1
        self.tab_switch_in_progress = False
        self.untitled_tab_counter = 1

        # draggable step title (axes coords)
        self.text_pos = (0.5, -0.08)
        self.text_fontsize = 10
        self.text_artist = None
        self.dragging_title = False

        # labels
        self.material_fontsize = 8
        default_font_family = self.font().family() or "Sans Serif"
        self.title_font_family = default_font_family
        self.title_color = "#000000"
        self.material_font_family = default_font_family
        self.material_color = "#000000"
        self.material_legend_pos = (0.8, 0.0)
        self.material_legend_artist = None
        self.dragging_material_legend = False
        self.material_legend_drag_offset = (0.0, 0.0)

        # Vertical visual scale (applies to main view + exports)
        self.visual_scale_factor = 1.0
        # Per-project illustrative emphasis for process features.  This is
        # deliberately independent from global view zoom and never changes
        # the physical process dimensions.
        self.feature_scale_factor = FEATURE_EMPHASIS_DEFAULT
        self.visualisation_mode = "dynamic"
        self.visual_thickness_overrides = {}
        self.visual_thickness_step_overrides = {}
        self.visual_thickness_pick_mode = False
        self.visual_thickness_highlight_artists = []
        self.visual_adjustment_notice_dismissed = False
        # Dialog and controls for vertical scale settings.
        self.scale_dialog = None
        self.global_scale_spin = None
        self.feature_scale_dialog_spin = None
        # Feature emphasis is controlled from the canvas overlay. Keep these
        # legacy toolbar references empty so state synchronisation remains
        # compatible without rendering a duplicate control near Substrate.
        self.feature_scale_slider = None
        self.feature_scale_value_label = None
        # Legacy finalise/accept/reject controls are no longer exposed. A
        # single Feature button starts direct visual-thickness picking.
        self.finalise_visualisation_button = None
        self.accept_visualisation_button = None
        self.reject_visualisation_button = None
        self.adjust_visual_thickness_button = None

        # draggable substrate label (data coords)
        self.substrate_text_artist = None
        the_substrate_text_pos = None
        self.substrate_text_pos = the_substrate_text_pos
        self.dragging_substrate = False

        # draggable layer labels (data coords)
        self.layer_text_positions = {}   # {material_idx: (x,y)}
        self.layer_text_artists = {}     # {material_idx: TextArtist}
        self.dragging_layer_idx = None
        # canvas drag for lithography windows
        self.dragging_litho_window = False
        self.litho_drag_start_frac = None
        self.litho_drag_current_frac = None
        self.litho_drag_patch = None
        self._last_op_name_for_view = None

        # Project preference for including method details in step titles.
        self.include_method_in_step_titles = True
        self.include_logo_in_exports = True
        # Controls whether restricted materials are visible (read-only) in operation selector.
        self.show_restricted_materials_in_selector = False
        # Per-tab one-click restore point captured before "Reset wafer".
        self._reset_restore_info_by_tab = {}
        # Autosave settings.
        self.autosave_enabled = True
        self.autosave_interval_minutes = 15
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(int(self.autosave_interval_minutes * 60 * 1000))
        self.autosave_timer.timeout.connect(self._on_autosave_timer)
        self.unsaved_save_prompt_widget = None
        self._init_process_assistant_state()

        # ---------- UI ----------
        self._build_main_ui(
            canvas_cls=CrossSectionCanvas,
            preview_export_as_main_view=PREVIEW_EXPORT_AS_MAIN_VIEW,
            left_panel_top_min_height_px=LEFT_PANEL_TOP_MIN_HEIGHT_PX,
            left_panel_top_max_height_px=LEFT_PANEL_TOP_MAX_HEIGHT_PX,
            height_um_default=HEIGHT_UM_DEFAULT,
        )

        # Logo overlay
        self._init_fixed_overlay_logo()
        self._init_canvas_scale_overlay()
        self._init_visual_adjustment_notice_overlay()
        self.figure_scroll.verticalScrollBar().valueChanged.connect(
            lambda _: self._sync_canvas_overlay_positions()
        )

        # menus
        self._build_menus()

        # mouse events for dragging texts
        self.canvas.mpl_connect("button_press_event", self.on_canvas_mouse_press)
        self.canvas.mpl_connect("button_release_event", self.on_canvas_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_canvas_mouse_move)

        self._connect_main_ui_signals()
        self._initialize_main_ui_state(
            preview_export_as_main_view=PREVIEW_EXPORT_AS_MAIN_VIEW
        )
        self._sync_canvas_scale_overlay_from_state()

    def _set_startup_window_size(self):
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1400, 800)
            return

        available = screen.availableGeometry()
        width = min(1400, max(720, int(available.width() * 0.96)))
        height = min(900, max(520, int(available.height() * 0.92)))
        width = min(width, max(1, available.width()))
        height = min(height, max(1, available.height()))
        self.resize(width, height)
        self.move(
            available.x() + max(0, (available.width() - width) // 2),
            available.y() + max(0, (available.height() - height) // 2),
        )

    # ---------- Thickness mapping ----------
    def process_grid_height_um_for_substrate(self, substrate_thickness_um: float) -> float:
        substrate_um = max(0.0, float(substrate_thickness_um))
        if substrate_um <= 0.0:
            return float(PROCESS_TOP_HEADROOM_UM)
        fraction = max(0.05, min(0.95, float(PROCESS_SUBSTRATE_VIEW_FRACTION)))
        target_height_um = max(
            substrate_um + float(PROCESS_TOP_HEADROOM_UM),
            substrate_um / fraction,
        )
        return min(float(HEIGHT_UM_DEFAULT), target_height_um)

    def um_to_process_px(
        self,
        um_value: float,
        *,
        mode: str = "ceil",
        minimum: int = 0,
    ) -> int:
        """
        Linear physical thickness/depth mapping (µm -> process px).
        Process operations use a fixed physical simulation resolution.
        """
        um = max(0.0, float(um_value))
        if um <= 0.0:
            return max(int(minimum), 0)

        px_float = um / max(1e-12, float(PROCESS_UM_PER_PX))

        mode_norm = str(mode or "ceil").strip().lower()
        if mode_norm == "floor":
            px = int(math.floor(px_float + 1e-12))
        elif mode_norm == "round":
            px = int(round(px_float))
        else:
            px = int(math.ceil(px_float - 1e-12))

        return max(px, int(minimum))

    # ---------- FIXED OVERLAY LOGO ----------
    def _init_fixed_overlay_logo(self):
        self.overlay_logo_label = QLabel(self.figure_scroll.viewport())
        self.overlay_logo_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.overlay_logo_label.setStyleSheet("background: transparent;")
        self.overlay_logo_label.setScaledContents(True)
        self.overlay_logo_pixmap = None

        if os.path.exists(LOGO_PATH):
            pm = QPixmap(LOGO_PATH)
            if not pm.isNull():
                self.overlay_logo_pixmap = pm

        self._update_fixed_overlay_logo_size()
        self._position_fixed_overlay_logo()
        self.overlay_logo_label.show()
        self.overlay_logo_label.raise_()

    def _clamped_visual_scale(self, value=None):
        if value is None:
            value = getattr(self, "visual_scale_factor", 1.0)
        return max(VISUAL_SCALE_MIN, min(VISUAL_SCALE_MAX, float(value)))

    def _clamped_feature_scale(self, value=None):
        if value is None:
            value = getattr(self, "feature_scale_factor", FEATURE_EMPHASIS_DEFAULT)
        return max(
            FEATURE_EMPHASIS_MIN,
            min(FEATURE_EMPHASIS_MAX, float(value)),
        )

    def _feature_emphasis_name(self, value=None):
        value = self._clamped_feature_scale(value)
        if abs(value - 0.25) <= 1e-9:
            return "Compact"
        if abs(value - 0.50) <= 1e-9:
            return "Balanced"
        if abs(value - 1.00) <= 1e-9:
            return "Expanded"
        return "Custom"

    def _feature_emphasis_text(self, value=None):
        value = self._clamped_feature_scale(value)
        return f"{self._feature_emphasis_name(value)} · {value:.2f}"

    def _sync_feature_scale_controls_from_state(self):
        value = self._clamped_feature_scale()
        slider_value = int(round(value * 100.0))

        for slider_name in ("feature_scale_slider", "feature_scale_overlay_slider"):
            slider = getattr(self, slider_name, None)
            if slider is None:
                continue
            blocked = slider.blockSignals(True)
            try:
                slider.setValue(slider_value)
            finally:
                slider.blockSignals(blocked)

        if self.feature_scale_value_label is not None:
            self.feature_scale_value_label.setText(self._feature_emphasis_text(value))
        overlay_value = getattr(self, "feature_scale_overlay_value", None)
        if overlay_value is not None:
            overlay_value.setText(f"{value:.2f}")

        if self.feature_scale_dialog_spin is not None:
            blocked = self.feature_scale_dialog_spin.blockSignals(True)
            try:
                self.feature_scale_dialog_spin.setValue(value)
            finally:
                self.feature_scale_dialog_spin.blockSignals(blocked)

    def _set_feature_scale_factor(self, value, *, confirm_override_reset=True):
        value = self._clamped_feature_scale(value)
        previous = self._clamped_feature_scale()
        if abs(value - previous) <= 1e-9:
            self._sync_feature_scale_controls_from_state()
            return False

        mode = (
            self._normalised_visualisation_mode()
            if hasattr(self, "_normalised_visualisation_mode")
            else "dynamic"
        )
        if mode in {"pending", "finalised"} and confirm_override_reset:
            answer = QMessageBox.question(
                self,
                "Reset custom display heights?",
                "Changing feature detail will reset your custom display heights.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                self._sync_feature_scale_controls_from_state()
                return False

        self.feature_scale_factor = value
        if mode in {"pending", "finalised"}:
            self.visual_thickness_overrides = {}
            self.visual_thickness_step_overrides = {}
            self.visual_adjustment_notice_dismissed = False
            self._set_visualisation_mode("dynamic")

        self._sync_feature_scale_controls_from_state()
        self.update_plot(self.current_title or "")
        if PREVIEW_EXPORT_AS_MAIN_VIEW:
            self._apply_export_preview_canvas_aspect()
        if hasattr(self, "_snapshot_active_tab_state"):
            self._snapshot_active_tab_state()
        if hasattr(self, "statusBar"):
            self.statusBar().showMessage(
                f"Feature detail: {self._feature_emphasis_text(value)}.",
                4000,
            )
        return True

    def _init_canvas_scale_overlay(self):
        self.canvas_scale_overlay = QWidget(self.figure_scroll.viewport())
        self.canvas_scale_overlay.setObjectName("canvasScaleOverlay")
        self.canvas_scale_overlay.setStyleSheet(
            "QWidget#canvasScaleOverlay {"
            " background-color: rgba(30, 38, 50, 170);"
            " border: 1px solid rgba(120, 135, 160, 150);"
            " border-radius: 6px;"
            "}"
            "QWidget#canvasScaleOverlay QLabel { color: #e8ecf3; }"
        )

        layout = QGridLayout(self.canvas_scale_overlay)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(3)
        layout.setColumnMinimumWidth(0, 48)
        layout.setColumnMinimumWidth(1, 110)
        layout.setColumnMinimumWidth(2, 44)

        self.global_scale_overlay_slider = QSlider(Qt.Horizontal, self.canvas_scale_overlay)
        self.global_scale_overlay_slider.setRange(
            int(round(VISUAL_SCALE_MIN * 100.0)),
            int(round(VISUAL_SCALE_MAX * 100.0)),
        )
        self.global_scale_overlay_slider.setSingleStep(5)
        self.global_scale_overlay_slider.setPageStep(10)
        self.global_scale_overlay_slider.setFixedWidth(110)
        self.global_scale_overlay_value = QLabel("1.00×")
        self.global_scale_overlay_label = QLabel("Zoom")
        self.global_scale_overlay_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.global_scale_overlay_label, 0, 0)
        layout.addWidget(self.global_scale_overlay_slider, 0, 1)
        layout.addWidget(self.global_scale_overlay_value, 0, 2)

        self.feature_scale_overlay_label = QLabel("Detail")
        self.feature_scale_overlay_slider = QSlider(Qt.Horizontal, self.canvas_scale_overlay)
        self.feature_scale_overlay_slider.setRange(
            int(round(FEATURE_EMPHASIS_MIN * 100.0)),
            int(round(FEATURE_EMPHASIS_MAX * 100.0)),
        )
        self.feature_scale_overlay_slider.setSingleStep(5)
        self.feature_scale_overlay_slider.setPageStep(10)
        self.feature_scale_overlay_slider.setFixedWidth(110)
        self.feature_scale_overlay_value = QLabel(f"{FEATURE_EMPHASIS_DEFAULT:.2f}")
        self.feature_scale_overlay_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.feature_scale_overlay_label, 1, 0)
        layout.addWidget(self.feature_scale_overlay_slider, 1, 1)
        layout.addWidget(self.feature_scale_overlay_value, 1, 2)

        self.global_scale_overlay_slider.setToolTip("Zoom the stack")
        self.feature_scale_overlay_slider.setToolTip(
            "Emphasise thin layers and etched depths without changing dimensions"
        )

        self.global_scale_overlay_slider.valueChanged.connect(self._on_overlay_global_scale_changed)
        self.feature_scale_overlay_slider.valueChanged.connect(
            lambda value: self._set_feature_scale_factor(float(value) / 100.0)
        )

        self.canvas_scale_overlay.adjustSize()
        self._sync_canvas_scale_overlay_from_state()
        self._position_canvas_scale_overlay()
        self.canvas_scale_overlay.show()
        self.canvas_scale_overlay.raise_()

    def _on_overlay_global_scale_changed(self, value: int):
        scale = self._clamped_visual_scale(float(value) / 100.0)
        self.visual_scale_factor = scale
        self.global_scale_overlay_value.setText(f"{scale:.2f}×")
        if self.global_scale_spin is not None:
            block = self.global_scale_spin.blockSignals(True)
            try:
                self.global_scale_spin.setValue(scale)
            finally:
                self.global_scale_spin.blockSignals(block)
        self.update_plot(self.current_title or "")

    def _sync_canvas_scale_overlay_from_state(self):
        if not hasattr(self, "global_scale_overlay_slider"):
            return

        g = self._clamped_visual_scale()

        bg = self.global_scale_overlay_slider.blockSignals(True)
        try:
            self.global_scale_overlay_slider.setValue(int(round(g * 100.0)))
        finally:
            self.global_scale_overlay_slider.blockSignals(bg)

        self.global_scale_overlay_value.setText(f"{g:.2f}×")
        self._sync_feature_scale_controls_from_state()

    def _position_canvas_scale_overlay(self):
        if not hasattr(self, "canvas_scale_overlay") or self.canvas_scale_overlay is None:
            return
        margin = 4
        vp = self.figure_scroll.viewport()
        self.canvas_scale_overlay.adjustSize()
        x = vp.width() - self.canvas_scale_overlay.width() - margin
        y = margin
        self.canvas_scale_overlay.move(max(x, margin), y)
        self.canvas_scale_overlay.raise_()

    def _init_visual_adjustment_notice_overlay(self):
        self.visual_adjustment_notice_overlay = QWidget(self.figure_scroll.viewport())
        self.visual_adjustment_notice_overlay.setObjectName("visualAdjustmentNotice")
        self.visual_adjustment_notice_overlay.setStyleSheet(
            "QWidget#visualAdjustmentNotice {"
            " background-color: rgba(255, 255, 255, 220);"
            " border: 1px solid rgba(156, 163, 175, 180);"
            " border-radius: 6px;"
            "}"
            "QWidget#visualAdjustmentNotice QLabel { color: #374151; }"
            "QWidget#visualAdjustmentNotice QPushButton {"
            " color: #4b5563;"
            " background: transparent;"
            " border: none;"
            " font-weight: 600;"
            " padding: 0px;"
            "}"
            "QWidget#visualAdjustmentNotice QPushButton:hover {"
            " background-color: rgba(229, 231, 235, 180);"
            " border-radius: 4px;"
            "}"
        )

        layout = QHBoxLayout(self.visual_adjustment_notice_overlay)
        layout.setContentsMargins(10, 6, 8, 6)
        layout.setSpacing(8)

        self.visual_adjustment_notice_label = QLabel(
            "Custom display heights are active. Physical dimensions are unchanged.",
            self.visual_adjustment_notice_overlay,
        )
        self.visual_adjustment_notice_label.setWordWrap(True)
        self.visual_adjustment_notice_close = QPushButton(
            "×",
            self.visual_adjustment_notice_overlay,
        )
        self.visual_adjustment_notice_close.setFixedSize(18, 18)
        self.visual_adjustment_notice_close.setToolTip("Dismiss")
        self.visual_adjustment_notice_close.clicked.connect(
            self._dismiss_visual_adjustment_notice
        )

        layout.addWidget(self.visual_adjustment_notice_label, 1)
        layout.addWidget(self.visual_adjustment_notice_close, 0)
        self.visual_adjustment_notice_overlay.hide()

    def _dismiss_visual_adjustment_notice(self):
        self.visual_adjustment_notice_dismissed = True
        if hasattr(self, "visual_adjustment_notice_overlay"):
            self.visual_adjustment_notice_overlay.hide()

    def _set_visual_adjustment_notice_visible(self, active: bool):
        if not hasattr(self, "visual_adjustment_notice_overlay"):
            return
        if not active:
            self.visual_adjustment_notice_dismissed = False
            self.visual_adjustment_notice_overlay.hide()
            return
        if self.visual_adjustment_notice_dismissed:
            self.visual_adjustment_notice_overlay.hide()
            return
        self._position_visual_adjustment_notice_overlay()
        self.visual_adjustment_notice_overlay.show()
        self.visual_adjustment_notice_overlay.raise_()

    def _position_visual_adjustment_notice_overlay(self):
        if not hasattr(self, "visual_adjustment_notice_overlay") or self.visual_adjustment_notice_overlay is None:
            return
        margin = 8
        vp = self.figure_scroll.viewport()
        available = max(180, vp.width() - 2 * margin)
        width = min(560, available)
        self.visual_adjustment_notice_overlay.setFixedWidth(width)
        self.visual_adjustment_notice_overlay.adjustSize()
        x = max(margin, (vp.width() - self.visual_adjustment_notice_overlay.width()) // 2)
        y = margin
        self.visual_adjustment_notice_overlay.move(x, y)
        self.visual_adjustment_notice_overlay.raise_()

    def _sync_canvas_overlay_positions(self):
        self._position_canvas_scale_overlay()
        self._position_visual_adjustment_notice_overlay()
        self._position_fixed_overlay_logo()

    def _update_fixed_overlay_logo_size(self):
        if self.overlay_logo_pixmap is None:
            self.overlay_logo_label.hide()
            return

        vw = self.figure_scroll.viewport().width()
        vh = self.figure_scroll.viewport().height()
        if vw <= 0 or vh <= 0:
            return

        target_w = max(48, int(vw * 0.28))
        scaled = self.overlay_logo_pixmap.scaled(
            target_w,
            target_w,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.overlay_logo_label.setPixmap(scaled)
        self.overlay_logo_label.setFixedSize(scaled.size())
        self.overlay_logo_label.show()
        self.overlay_logo_label.raise_()

    def _position_fixed_overlay_logo(self):
        if not hasattr(self, "overlay_logo_label") or self.overlay_logo_label is None:
            return
        if self.overlay_logo_label.pixmap() is None:
            return

        margin = 3
        vp = self.figure_scroll.viewport()
        x = vp.width() - self.overlay_logo_label.width() - margin
        y = margin
        if hasattr(self, "canvas_scale_overlay") and self.canvas_scale_overlay is not None:
            y += self.canvas_scale_overlay.height() + margin
        self.overlay_logo_label.move(max(x, margin), y)
        self.overlay_logo_label.raise_()

    def update_logo_pixmap(self):
        if not hasattr(self, "logo_label") or getattr(self, "logo_pixmap", None) is None:
            return
        if self.logo_pixmap.isNull():
            return

        width = self.logo_label.width()
        if width <= 0:
            return

        target_height = self.logo_label.height() if self.logo_label.height() > 0 else width
        scaled = self.logo_pixmap.scaled(
            width,
            target_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.logo_label.setPixmap(scaled)

    def _apply_preview_left_panel_width(self):
        if not PREVIEW_EXPORT_AS_MAIN_VIEW:
            return
        if not hasattr(self, "top_panel") or self.top_panel is None:
            return

        wanted_h = self.top_panel.sizeHint().height()
        wanted_h = max(80, min(wanted_h, 170))
        self.top_panel.setMinimumHeight(wanted_h)
        self.top_panel.setMaximumHeight(wanted_h)

    def _apply_export_preview_canvas_aspect(self):
        """
        Adjust the preview canvas aspect ratio according to EXPORT_PREVIEW_FIGSIZE
        and the current vertical visual scale factor.
        """
        if not PREVIEW_EXPORT_AS_MAIN_VIEW:
            return
        if not hasattr(self, "figure_scroll") or self.figure_scroll is None:
            return
        if not hasattr(self, "canvas") or self.canvas is None:
            return

        vw = self.figure_scroll.viewport().width()
        vh = self.figure_scroll.viewport().height()
        if vw <= 0 or vh <= 0:
            return

        base_w, base_h = EXPORT_PREVIEW_FIGSIZE
        scale = (
            self._visual_render_scale_factor()
            if hasattr(self, "_visual_render_scale_factor")
            else self._clamped_visual_scale()
        )
        canvas_scale = max(1.0, scale)
        visual_height_scale = (
            self._visual_override_canvas_height_scale()
            if hasattr(self, "_visual_override_canvas_height_scale")
            else 1.0
        )
        effective_h = base_h * canvas_scale * visual_height_scale
        ratio = base_w / effective_h  # width / height

        target_w = vw
        target_h = int(vw / ratio)
        # Keep the canvas at least viewport-sized for scales below 1.0. The
        # stack itself is scaled by plot coordinates, not by shrinking canvas.
        target_h = max(target_h, int(vh * canvas_scale * visual_height_scale), vh, 240)

        self.canvas.setFixedSize(target_w, target_h)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_logo_pixmap()
        self._apply_preview_left_panel_width()
        self._sync_stack_viewport_geometry()

    def showEvent(self, event):
        super().showEvent(event)
        # Ensure final viewport size is known before applying preview geometry.
        QTimer.singleShot(0, self._sync_stack_viewport_geometry)

    def on_main_splitter_moved(self, _pos, _index):
        # Splitter drags do not trigger MainWindow.resizeEvent.
        self._sync_stack_viewport_geometry()

    def _sync_stack_viewport_geometry(self):
        self._apply_export_preview_canvas_aspect()
        if hasattr(self, "canvas_scale_overlay"):
            self._position_canvas_scale_overlay()
        if hasattr(self, "visual_adjustment_notice_overlay"):
            self._position_visual_adjustment_notice_overlay()
        if hasattr(self, "overlay_logo_label"):
            self._update_fixed_overlay_logo_size()
            self._position_fixed_overlay_logo()

    # ---------- Project tabs ----------
    def _default_new_process_data(self) -> dict:
        substrate_name = "Silicon (Si)"
        substrate_thickness_um = 200
        width_px = WIDTH_UM_DEFAULT
        height_um = self.process_grid_height_um_for_substrate(substrate_thickness_um)
        height_px = self.um_to_process_px(height_um, mode="ceil", minimum=1)
        substrate_thickness_px = self.um_to_process_px(substrate_thickness_um, mode="ceil", minimum=1)
        xsec = CrossSection(
            width_px=width_px,
            height_px=height_px,
            substrate_index=1,
            void_index=0,
            substrate_thickness_px=substrate_thickness_px,
            substrate_thickness_um=substrate_thickness_um,
            top_headroom_um=max(0.0, float(height_um) - float(substrate_thickness_um)),
        )
        init_title = f"Substrate: {substrate_name} — {substrate_thickness_um:g} µm"
        return {
            "format_version": 1,
            "material_names": ["Void", "Silicon (Si)"],
            "material_colors": ["#ffffff", "#b0b0b0"],
            "void_index": 0,
            "substrate_name": substrate_name,
            "substrate_thickness": substrate_thickness_um,
            "history": [{
                "grid": xsec.copy_state(),
                "row_heights_um": xsec.copy_row_heights(),
                "visual_geometry": xsec.copy_visual_geometry(),
                "structural_role_grid": xsec.copy_structural_roles(),
                "structural_bodies": xsec.copy_structural_bodies(),
                "planarizer_role_grid": xsec.copy_planarizer_roles(),
                "planarizer_roles": xsec.copy_planarizer_metadata(),
                "title": init_title,
            }],
            "process_steps": None,
            "text_pos": (0.5, -0.08),
            "text_fontsize": int(getattr(self, "text_fontsize", 10)),
            "material_fontsize": int(getattr(self, "material_fontsize", 8)),
            "title_font_family": str(getattr(self, "title_font_family", self.font().family() or "Sans Serif")),
            "title_color": str(getattr(self, "title_color", "#000000")),
            "material_font_family": str(getattr(self, "material_font_family", self.font().family() or "Sans Serif")),
            "material_color": str(getattr(self, "material_color", "#000000")),
            "material_legend_pos": tuple(getattr(self, "material_legend_pos", (0.8, 0.0))),
            "litho_windows": [],
            "window_start": 0.35,
            "window_end": 0.65,
            "deleted_materials": [],
            "recent_materials": [],
            "hidden_material_categories": list(getattr(self, "hidden_material_categories", set())),
            "parameter_classes": self.load_default_parameter_classes(),
            "include_method_in_step_titles": bool(getattr(self, "include_method_in_step_titles", True)),
            "include_logo_in_exports": bool(getattr(self, "include_logo_in_exports", True)),
            "visual_scale_factor": self._clamped_visual_scale(),
            "feature_scale_factor": self._clamped_feature_scale(),
            "visualisation_mode": "dynamic",
            "visual_thickness_overrides": {},
            "visual_thickness_step_overrides": {},
            "llm_chat_messages": [],
        }

    # ---------- Menus ----------
    def _build_menus(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        action_new = QAction("New process", self)
        action_open = QAction("Open process…", self)
        action_save = QAction("Save", self)
        action_save_as = QAction("Save As…", self)
        action_quit = QAction("Quit FabStack", self)

        action_new.triggered.connect(self.new_process)
        action_open.triggered.connect(self.load_process)
        action_save.triggered.connect(self.save_process)
        action_save_as.triggered.connect(self.save_process_as)
        action_quit.triggered.connect(self.close)

        file_menu.addAction(action_new)
        file_menu.addSeparator()
        file_menu.addAction(action_open)
        file_menu.addSeparator()
        file_menu.addAction(action_save)
        file_menu.addAction(action_save_as)
        file_menu.addSeparator()

        self.act_show_restricted_materials = QAction("Include restricted materials", self)
        self.act_show_restricted_materials.setCheckable(True)
        self.act_show_restricted_materials.setChecked(
            bool(getattr(self, "show_restricted_materials_in_selector", False))
        )
        self.act_show_restricted_materials.toggled.connect(
            self.on_toggle_show_restricted_materials
        )
        file_menu.addAction(self.act_show_restricted_materials)
        self.act_autosave_15min = QAction("Autosave every 15 minutes", self)
        self.act_autosave_15min.setCheckable(True)
        self.act_autosave_15min.setChecked(bool(getattr(self, "autosave_enabled", True)))
        self.act_autosave_15min.toggled.connect(self.on_toggle_autosave)
        file_menu.addAction(self.act_autosave_15min)

        file_menu.addSeparator()

        # Expert user activation (password: 0000)
        self.action_expert_user = QAction("Expert mode…", self)
        self.action_expert_user.triggered.connect(self.handle_expert_user_action)
        file_menu.addAction(self.action_expert_user)

        file_menu.addSeparator()
        file_menu.addAction(action_quit)

        edit_menu = menubar.addMenu("Edit")
        act_undo = QAction("Undo", self)
        act_redo = QAction("Redo", self)
        act_undo.triggered.connect(self.undo_last_step)
        act_redo.triggered.connect(self.redo_step)

        act_undo.setShortcut(QKeySequence.Undo)
        act_redo.setShortcut(QKeySequence.Redo)

        edit_menu.addAction(act_undo)
        edit_menu.addAction(act_redo)
        self.act_restore_reset = QAction("Undo process reset", self)
        self.act_restore_reset.triggered.connect(self.restore_before_reset_wafer)
        self.act_restore_reset.setEnabled(False)
        edit_menu.addAction(self.act_restore_reset)

        edit_menu.addSeparator()
        act_set_dep_step = QAction("Display settings…", self)
        act_set_dep_step.triggered.connect(self.set_deposition_thickness_step)
        edit_menu.addAction(act_set_dep_step)

        # Text options (moved here from Methods dialog)
        act_text_options = QAction("Text style…", self)
        act_text_options.triggered.connect(self.open_text_options_dialog)
        edit_menu.addAction(act_text_options)

        # Toggle method details in generated process-step titles.
        self.act_toggle_method_in_titles = QAction(
            "Show methods in step titles", self
        )
        self.act_toggle_method_in_titles.setCheckable(True)
        self.act_toggle_method_in_titles.setChecked(
            self.include_method_in_step_titles
        )
        self.act_toggle_method_in_titles.toggled.connect(
            self.on_toggle_method_in_titles
        )
        edit_menu.addAction(self.act_toggle_method_in_titles)

        lib_menu = menubar.addMenu("Library")
        lib_manage = QAction("Manage materials…", self)
        lib_add_material = QAction("Create material…", self)
        lib_add_from_lib = QAction("Add library material…", self)
        lib_recent = QAction("Recent materials…", self)

        lib_manage.triggered.connect(self.open_library_manager)
        lib_add_material.triggered.connect(self.add_material)
        lib_add_from_lib.triggered.connect(self.add_material_from_library)
        lib_recent.triggered.connect(self.show_recent_materials_dialog)

        # Keep a reference for expert gating
        self.lib_manage_action = lib_manage

        lib_menu.addAction(lib_manage)
        lib_menu.addAction(lib_add_material)
        lib_menu.addAction(lib_add_from_lib)
        lib_menu.addSeparator()
        lib_menu.addAction(lib_recent)

        export_menu = menubar.addMenu("Export")
        self.act_include_logo_in_exports = QAction("Include logo in exports", self)
        self.act_include_logo_in_exports.setCheckable(True)
        self.act_include_logo_in_exports.setChecked(
            bool(getattr(self, "include_logo_in_exports", True))
        )
        self.act_include_logo_in_exports.toggled.connect(
            self.on_toggle_logo_in_exports
        )

        exp_png = QAction("Process image (PNG)…", self)
        exp_word = QAction("Process document (Word)…", self)
        exp_copy = QAction("Copy current view", self)
        # Export only the process-step titles as plain text or Word.
        exp_list_txt = QAction("Step list (TXT)…", self)
        exp_list_word = QAction("Step list (Word)…", self)

        exp_png.triggered.connect(
            lambda _checked=False: self._run_export_action("export the process image", self.export_history_image)
        )
        exp_word.triggered.connect(
            lambda _checked=False: self._run_export_action("export the process document", self.export_process_word)
        )
        exp_copy.triggered.connect(
            lambda _checked=False: self._run_export_action("copy the current view", self.export_current_view)
        )
        exp_list_txt.triggered.connect(
            lambda _checked=False: self._run_export_action("export the step list", self.export_step_list_txt)
        )
        exp_list_word.triggered.connect(
            lambda _checked=False: self._run_export_action("export the step list", self.export_step_list_word)
        )

        export_menu.addAction(self.act_include_logo_in_exports)
        export_menu.addSeparator()
        export_menu.addAction(exp_png)
        export_menu.addAction(exp_word)
        export_menu.addAction(exp_copy)
        export_menu.addSeparator()
        export_menu.addAction(exp_list_txt)
        export_menu.addAction(exp_list_word)

    def handle_expert_user_action(self):
        """
        File → Expert user…
        - If not expert: asks for password "0000" and enables expert mode on success.
        - If already expert: allows disabling expert mode.
        """
        if self.is_expert_user:
            reply = QMessageBox.question(
                self,
                "Turn off Expert mode?",
                "Advanced editing controls will be hidden.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.is_expert_user = False
                self.statusBar().showMessage("Expert mode turned off.", 3000)
                self.update_expert_user_controls()
            return

        pwd = self._prompt_expert_password()
        if pwd is None:
            return
        if pwd == "0000":
            self.is_expert_user = True
            self.statusBar().showMessage("Expert mode turned on.", 3000)
            self.update_expert_user_controls()
        else:
            QMessageBox.warning(
                self,
                "Incorrect password",
                "Expert mode was not enabled.",
            )

    def ensure_expert_user_mode(self, feature_name: str = "this feature") -> bool:
        """
        Ensure expert mode is active, offering inline login when needed.

        Returns True when expert mode is active, False otherwise.
        """
        if self.is_expert_user:
            return True

        reply = QMessageBox.question(
            self,
            "Expert mode required",
            f"Turn on Expert mode to use {feature_name}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return False

        pwd = self._prompt_expert_password()
        if pwd is None:
            return False
        if pwd == "0000":
            self.is_expert_user = True
            self.statusBar().showMessage("Expert mode turned on.", 3000)
            self.update_expert_user_controls()
            return True

        QMessageBox.warning(
            self,
            "Incorrect password",
            "Expert mode was not enabled.",
        )
        return False

    def _prompt_expert_password(self):
        """Show a styled in-app password prompt for expert mode."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Turn on Expert mode")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Password:"))

        pwd_edit = QLineEdit()
        pwd_edit.setEchoMode(QLineEdit.Password)
        pwd_edit.setClearButtonEnabled(True)
        layout.addWidget(pwd_edit)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        layout.addWidget(btn_box)

        pwd_edit.returnPressed.connect(dlg.accept)
        pwd_edit.setFocus()

        if dlg.exec_() != QDialog.Accepted:
            return None
        return pwd_edit.text()

    def update_expert_user_controls(self):
        """
        Enable/disable expert-only controls:
        - Library → Manage library
        - Method dialog → Parameters tab
        """
        # Update menu text so it's clear when mode is active
        if hasattr(self, "action_expert_user") and self.action_expert_user is not None:
            if self.is_expert_user:
                self.action_expert_user.setText("Expert mode (active)…")
            else:
                self.action_expert_user.setText("Expert mode…")

        # Library manager remains clickable and will prompt for login if needed.
        if hasattr(self, "lib_manage_action") and self.lib_manage_action is not None:
            self.lib_manage_action.setEnabled(True)
            if self.is_expert_user:
                self.lib_manage_action.setToolTip("Manage the material library")
            else:
                self.lib_manage_action.setToolTip("Requires Expert mode")

        # Method settings are available to all users; only the Parameters tab is gated.
        if hasattr(self, "op_params_button") and self.op_params_button is not None:
            self.op_params_button.setEnabled(True)
            self.op_params_button.setToolTip("Configure this process step")

        if getattr(self, "op_params_dialog", None) is not None and hasattr(self, "_sync_op_params_expert_state"):
            self._sync_op_params_expert_state()

    def on_toggle_method_in_titles(self, checked: bool):
        """Toggle whether method details are included in newly created step titles."""
        self.include_method_in_step_titles = checked

    def on_toggle_logo_in_exports(self, checked: bool):
        """Toggle whether exported images include the FabStack logo."""
        self.include_logo_in_exports = bool(checked)

    def show_recent_materials_dialog(self):
        """Show a simple selector for recent materials from the Library menu."""
        if not self.recent_materials:
            QMessageBox.information(
                self,
                "No recent materials",
                "Materials you use will appear here."
            )
            return

        items = [
            n for n in self.recent_materials
            if n != "Void" and n not in self.deleted_materials
        ]
        if not items:
            QMessageBox.information(
                self,
                "No recent materials",
                "Add a material to this process first."
            )
            return

        name, ok = QInputDialog.getItem(
            self,
            "Recent materials",
            "Select material:",
            items,
            0,
            False
        )
        if not ok or not name:
            return

        op = self.op_combo.currentText() if hasattr(self, "op_combo") else ""
        if self._is_material_restricted_for_operation(op, name):
            QMessageBox.information(
                self,
                "Choose another material",
                f"{name} cannot be used for {operation_label(op)}."
            )
            return

        idx = self._material_combo_index_by_full_name(name)
        if idx >= 0:
            self.material_combo.setCurrentIndex(idx)

    def set_deposition_thickness_step(self):
        """
        Open vertical scale settings.

        Process geometry remains physical; the overview uses an illustrative
        monotone scale so nanometres and millimetres remain readable together.
        """
        if self.scale_dialog is None:
            dlg = QDialog(self)
            dlg.setWindowTitle("Display settings")
            dlg.setModal(False)
            self.scale_dialog = dlg

            layout = QVBoxLayout(dlg)
            form = QFormLayout()
            layout.addLayout(form)

            self.global_scale_spin = QDoubleSpinBox(dlg)
            self.global_scale_spin.setRange(VISUAL_SCALE_MIN, VISUAL_SCALE_MAX)
            self.global_scale_spin.setDecimals(2)
            self.global_scale_spin.setSingleStep(0.05)
            self.global_scale_spin.setSuffix("×")
            form.addRow("View scale:", self.global_scale_spin)

            self.feature_scale_dialog_spin = QDoubleSpinBox(dlg)
            self.feature_scale_dialog_spin.setRange(
                FEATURE_EMPHASIS_MIN,
                FEATURE_EMPHASIS_MAX,
            )
            self.feature_scale_dialog_spin.setDecimals(2)
            self.feature_scale_dialog_spin.setSingleStep(0.05)
            form.addRow("Feature detail:", self.feature_scale_dialog_spin)

            info = QLabel(
                "View scale zooms the stack. Feature detail keeps thin layers "
                "and etched depths visible. Physical dimensions stay unchanged."
            )
            info.setWordWrap(True)
            layout.addWidget(info)

            btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            btn_box.button(QDialogButtonBox.Ok).setText("Apply")
            btn_box.accepted.connect(self._apply_scale_dialog_values)
            btn_box.rejected.connect(dlg.close)
            layout.addWidget(btn_box)

        if self.global_scale_spin is not None:
            self.global_scale_spin.setValue(self._clamped_visual_scale())
        if self.feature_scale_dialog_spin is not None:
            self.feature_scale_dialog_spin.setValue(self._clamped_feature_scale())

        self.scale_dialog.show()
        self.scale_dialog.raise_()
        self.scale_dialog.activateWindow()

    def _apply_scale_dialog_values(self):
        """Apply vertical visual scale from settings dialog."""
        prev_global = float(getattr(self, "visual_scale_factor", 1.0))
        requested_feature = self._clamped_feature_scale()
        if self.global_scale_spin is not None:
            self.visual_scale_factor = self._clamped_visual_scale(self.global_scale_spin.value())
        if self.feature_scale_dialog_spin is not None:
            requested_feature = self._clamped_feature_scale(
                self.feature_scale_dialog_spin.value()
            )
        if abs(float(self.visual_scale_factor) - prev_global) > 1e-9:
            self.update_plot(self.current_title or "")
            if PREVIEW_EXPORT_AS_MAIN_VIEW:
                self._apply_export_preview_canvas_aspect()
        self._sync_canvas_scale_overlay_from_state()
        self._set_feature_scale_factor(requested_feature)

        if self.scale_dialog is not None:
            self.scale_dialog.close()

    def _normalize_color_hex(self, value: str) -> str:
        c = QColor(str(value))
        if not c.isValid():
            c = QColor("#000000")
        return c.name()

    def _matplotlib_available_fonts(self):
        cached = getattr(self, "_mpl_font_names_cache", None)
        if cached is not None:
            return cached
        try:
            names = {f.name for f in mpl_font_manager.fontManager.ttflist if getattr(f, "name", "")}
        except Exception:
            names = set()
        self._mpl_font_names_cache = names
        return names

    def _matplotlib_font_family(self, requested_family: str) -> str:
        fam = str(requested_family or "").strip()
        if not fam:
            return "DejaVu Sans"
        blocked = {"-apple-system", "system-ui", "ui-sans-serif"}
        if fam in blocked or fam.startswith(".AppleSystemUIFont") or fam.startswith(".SF"):
            return "DejaVu Sans"
        available = self._matplotlib_available_fonts()
        if available and fam not in available:
            return "DejaVu Sans"
        return fam

    def _style_color_button(self, button: QPushButton, color_hex: str):
        if button is None:
            return
        c = QColor(self._normalize_color_hex(color_hex))
        lum = (0.2126 * c.red()) + (0.7152 * c.green()) + (0.0722 * c.blue())
        fg = "#000000" if lum > 145 else "#ffffff"
        button.setText(c.name())
        button.setStyleSheet(
            f"QPushButton {{ background-color: {c.name()}; color: {fg}; border: 1px solid palette(mid); padding: 2px 8px; }}"
        )

    def _style_font_button(self, button: QPushButton, family: str):
        if button is None:
            return
        fam = str(family or "").strip() or (self.font().family() or "Sans Serif")
        button.setText(fam)
        button.setFont(QFont(fam, button.font().pointSize()))

    def _refresh_plot_from_text_options(self):
        self.update_plot(self.current_title or "", auto_scroll_to_bottom=False)
        if self.text_options_dialog is not None and self.text_options_dialog.isVisible():
            self.text_options_dialog.raise_()
            self.text_options_dialog.activateWindow()

    def on_text_size_spin_changed(self, value: int):
        self.text_fontsize = int(value)
        if self.text_options_dialog is not None and self.text_options_dialog.isVisible():
            self._refresh_plot_from_text_options()
        else:
            self.update_plot(self.current_title or "")

    def on_material_text_size_spin_changed(self, value: int):
        self.material_fontsize = int(value)
        if self.text_options_dialog is not None and self.text_options_dialog.isVisible():
            self._refresh_plot_from_text_options()
        else:
            self.update_plot(self.current_title or "")

    def _choose_title_font(self):
        parent = self.text_options_dialog if self.text_options_dialog is not None else self
        initial = QFont(str(getattr(self, "title_font_family", "")) or (self.font().family() or "Sans Serif"))
        font, ok = QFontDialog.getFont(initial, parent, "Choose title font")
        if not ok:
            return
        self.title_font_family = font.family()
        self._sync_text_options_widgets_from_state()
        self._refresh_plot_from_text_options()

    def _choose_material_font(self):
        parent = self.text_options_dialog if self.text_options_dialog is not None else self
        initial = QFont(str(getattr(self, "material_font_family", "")) or (self.font().family() or "Sans Serif"))
        font, ok = QFontDialog.getFont(initial, parent, "Choose material label font")
        if not ok:
            return
        self.material_font_family = font.family()
        self._sync_text_options_widgets_from_state()
        self._refresh_plot_from_text_options()

    def _choose_title_color(self):
        parent = self.text_options_dialog if self.text_options_dialog is not None else self
        c = QColorDialog.getColor(QColor(self._normalize_color_hex(self.title_color)), parent, "Choose title colour")
        if not c.isValid():
            return
        self.title_color = c.name()
        self._sync_text_options_widgets_from_state()
        self._refresh_plot_from_text_options()

    def _choose_material_color(self):
        parent = self.text_options_dialog if self.text_options_dialog is not None else self
        c = QColorDialog.getColor(
            QColor(self._normalize_color_hex(self.material_color)),
            parent,
            "Choose material label colour",
        )
        if not c.isValid():
            return
        self.material_color = c.name()
        self._sync_text_options_widgets_from_state()
        self._refresh_plot_from_text_options()

    def _sync_text_options_widgets_from_state(self):
        if not hasattr(self, "title_font_button"):
            return
        self._style_font_button(self.title_font_button, self.title_font_family)
        self._style_color_button(self.title_color_button, self.title_color)
        self._style_font_button(self.material_font_button, self.material_font_family)
        self._style_color_button(self.material_color_button, self.material_color)

    def open_text_options_dialog(self):
        """Floating dialog to edit text fonts, colors and sizes."""
        if self.text_options_dialog is None:
            dlg = QDialog(self)
            dlg.setWindowTitle("Text style")
            dlg.setModal(False)
            self.text_options_dialog = dlg

            layout = QVBoxLayout(dlg)
            form = QFormLayout()
            self.title_font_button = QPushButton(dlg)
            self.title_font_button.clicked.connect(self._choose_title_font)
            self.title_font_button.setAutoDefault(False)
            self.title_font_button.setDefault(False)
            self.title_color_button = QPushButton(dlg)
            self.title_color_button.clicked.connect(self._choose_title_color)
            self.title_color_button.setAutoDefault(False)
            self.title_color_button.setDefault(False)
            self.material_font_button = QPushButton(dlg)
            self.material_font_button.clicked.connect(self._choose_material_font)
            self.material_font_button.setAutoDefault(False)
            self.material_font_button.setDefault(False)
            self.material_color_button = QPushButton(dlg)
            self.material_color_button.clicked.connect(self._choose_material_color)
            self.material_color_button.setAutoDefault(False)
            self.material_color_button.setDefault(False)
            form.addRow("Title font:", self.title_font_button)
            form.addRow("Title font size:", self.text_size_spin)
            form.addRow("Title colour:", self.title_color_button)
            form.addRow(QLabel(""))
            form.addRow("Material font:", self.material_font_button)
            form.addRow("Material font size:", self.material_font_size_spin)
            form.addRow("Material colour:", self.material_color_button)
            layout.addLayout(form)

            btn_box = QDialogButtonBox(QDialogButtonBox.Close)
            btn_box.rejected.connect(dlg.close)
            for btn in btn_box.buttons():
                btn.setAutoDefault(False)
                btn.setDefault(False)
            layout.addWidget(btn_box)

        self.text_size_spin.setValue(self.text_fontsize)
        self.material_font_size_spin.setValue(self.material_fontsize)
        self._sync_text_options_widgets_from_state()
        self.text_options_dialog.show()
        self.text_options_dialog.raise_()
        self.text_options_dialog.activateWindow()

    # ---------- Parameter defaults (global, across sessions) ----------
    def load_default_parameter_classes(self):
        """
        Load globally saved default parameter definitions from disk, if available.

        Returns a list of {'name': ..., 'unit': ...} dicts.
        Falls back to Temperature/Time if no file or on error.
        """
        defaults = [
            {"name": "Temperature", "unit": "°C"},
            {"name": "Time", "unit": "s"},
        ]
        path = PARAM_DEFAULTS_PATH
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    cleaned = []
                    for entry in data:
                        if not isinstance(entry, dict):
                            continue
                        name = str(entry.get("name", "")).strip()
                        if not name:
                            continue
                        unit = str(entry.get("unit", ""))
                        cleaned.append({"name": name, "unit": unit})
                    if cleaned:
                        return cleaned
        except Exception as exc:
            LOGGER.warning("Could not read parameter defaults from %s: %s", path, exc)
        return defaults

    def save_default_parameter_classes(self, entries=None):
        """
        Persist the given list of parameter class definitions as global defaults.
        If 'entries' is None, use self.parameter_classes.
        """
        if entries is None:
            entries = getattr(self, "parameter_classes", [])
        cleaned = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).strip()
            if not name:
                continue
            unit = str(entry.get("unit", ""))
            cleaned.append({"name": name, "unit": unit})
        if not cleaned:
            return
        path = PARAM_DEFAULTS_PATH
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp_path = path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(cleaned, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, path)
            LOGGER.info("Saved parameter defaults to %s", path)
        except Exception as exc:
            LOGGER.warning("Could not save parameter defaults to %s: %s", path, exc)

    def save_current_parameter_defaults(self):
        """
        Save the current list of parameter classes as the application-wide defaults.
        These defaults will be used for new processes and at next startup.
        """
        if not self.ensure_expert_user_mode("Parameters"):
            return
        if not getattr(self, "parameter_classes", None):
            QMessageBox.warning(self, "Nothing to save", "Add a parameter first.")
            return
        self.save_default_parameter_classes(self.parameter_classes)
        self.statusBar().showMessage("Parameter defaults saved for new processes.", 3500)

    # ---------- Method defaults and recipes (global, across sessions) ----------
    def _supported_method_operations(self):
        return (
            "Deposit",
            "Spincoating",
            "Bond",
            "Etch",
            "Strip material",
            "Lift-off",
            "Clean / Surface prep",
            "Bake/Annealing",
            "Lithography",
        )

    def _sanitize_method_state(self, state):
        if not isinstance(state, dict):
            return {}

        cleaned = {}

        dep = str(state.get("deposit_method", "")).strip()
        if dep and not dep.lower().startswith("no compatible"):
            cleaned["deposit_method"] = dep

        etchant = chem_normalize_ascii(str(state.get("etchant", "")).strip())
        if etchant and not etchant.lower().startswith("no compatible"):
            cleaned["etchant"] = etchant

        mode = str(state.get("etch_mode", "")).strip().lower()
        if mode in ("anisotropic", "isotropic"):
            cleaned["etch_mode"] = mode

        if "temperature" in state:
            try:
                cleaned["temperature"] = float(state.get("temperature"))
            except Exception:
                pass

        if "time" in state:
            try:
                cleaned["time"] = float(state.get("time"))
            except Exception:
                pass

        params_in = state.get("params", {})
        if isinstance(params_in, dict):
            params = {}
            for name, val in params_in.items():
                key = str(name).strip()
                if not key:
                    continue
                try:
                    params[key] = float(val)
                except Exception:
                    continue
            if params:
                cleaned["params"] = params

        return cleaned

    def _default_method_state_for_operation(self, op: str):
        raw = {}
        if isinstance(getattr(self, "default_method_settings", None), dict):
            raw = self.default_method_settings.get(op, {})
        return self._sanitize_method_state(raw)

    def load_default_method_settings(self):
        settings = {}
        path = METHOD_DEFAULTS_PATH
        allowed_ops = set(self._supported_method_operations())
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                raw = data.get("operations", data) if isinstance(data, dict) else {}
                if isinstance(raw, dict):
                    for op, state in raw.items():
                        op_name = str(op).strip()
                        if op_name not in allowed_ops:
                            continue
                        cleaned = self._sanitize_method_state(state)
                        if cleaned:
                            settings[op_name] = cleaned
        except Exception as exc:
            LOGGER.warning("Could not read method defaults from %s: %s", path, exc)
        return settings

    def save_default_method_settings(self):
        if not isinstance(getattr(self, "default_method_settings", None), dict):
            return
        cleaned_map = {}
        allowed_ops = set(self._supported_method_operations())
        for op, state in self.default_method_settings.items():
            op_name = str(op).strip()
            if op_name not in allowed_ops:
                continue
            cleaned = self._sanitize_method_state(state)
            if cleaned:
                cleaned_map[op_name] = cleaned

        path = METHOD_DEFAULTS_PATH
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp_path = path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump({"operations": cleaned_map}, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, path)
        except Exception as exc:
            LOGGER.warning("Could not save method defaults to %s: %s", path, exc)

    def load_method_recipes(self):
        recipes = {}
        path = METHOD_RECIPES_PATH
        allowed_ops = set(self._supported_method_operations())
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                raw = data.get("recipes", data) if isinstance(data, dict) else {}
                if isinstance(raw, dict):
                    for recipe_name, payload in raw.items():
                        name = str(recipe_name).strip()
                        if not name:
                            continue
                        op = ""
                        state_obj = payload
                        if isinstance(payload, dict) and "state" in payload:
                            op = str(payload.get("operation", "")).strip()
                            state_obj = payload.get("state", {})
                        if op and op not in allowed_ops:
                            op = ""
                        state = self._sanitize_method_state(state_obj)
                        if state:
                            recipes[name] = {"operation": op, "state": state}
        except Exception as exc:
            LOGGER.warning("Could not read method recipes from %s: %s", path, exc)
        return recipes

    def save_method_recipes(self):
        if not isinstance(getattr(self, "method_recipes", None), dict):
            return
        cleaned_recipes = {}
        allowed_ops = set(self._supported_method_operations())
        for recipe_name, payload in self.method_recipes.items():
            name = str(recipe_name).strip()
            if not name:
                continue
            op = ""
            state_obj = payload
            if isinstance(payload, dict) and "state" in payload:
                op = str(payload.get("operation", "")).strip()
                state_obj = payload.get("state", {})
            if op and op not in allowed_ops:
                op = ""
            state = self._sanitize_method_state(state_obj)
            if not state:
                continue
            cleaned_recipes[name] = {"operation": op, "state": state}

        path = METHOD_RECIPES_PATH
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp_path = path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump({"recipes": cleaned_recipes}, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, path)
        except Exception as exc:
            LOGGER.warning("Could not save method recipes to %s: %s", path, exc)

    def _capture_current_method_state(self):
        state = {
            "deposit_method": (self.deposit_method_combo.currentText() or "").strip(),
            "etchant": chem_normalize_ascii(
                (self.etchant_combo.currentData() or self.etchant_combo.currentText() or "").strip()
            ),
            "etch_mode": "isotropic" if getattr(self, "current_etch_mode", "anisotropic") == "isotropic" else "anisotropic",
            "temperature": float(self.temp_spin.value()),
            "time": float(self.exposure_spin.value()),
        }
        params = {}
        for name, widget in getattr(self, "custom_param_widgets", {}).items():
            if widget is None:
                continue
            try:
                params[str(name)] = float(widget.value())
            except Exception:
                continue
        if params:
            state["params"] = params
        return self._sanitize_method_state(state)

    def _set_etchant_combo_value(self, etchant_ascii: str) -> bool:
        target = chem_normalize_ascii((etchant_ascii or "").strip())
        if not target:
            return False

        idx = -1
        for i in range(self.etchant_combo.count()):
            item_val = self.etchant_combo.itemData(i) or self.etchant_combo.itemText(i)
            if chem_normalize_ascii((item_val or "").strip()) == target:
                idx = i
                break

        if idx >= 0:
            self.etchant_combo.setCurrentIndex(idx)
            self.on_etchant_changed()
            return True

        if self.etchant_combo.isEditable():
            self.etchant_combo.setEditText(chem_subs_unicode(target))
            self.on_etchant_changed()
            return True
        return False

    def _apply_method_state_to_widgets(self, state):
        cleaned = self._sanitize_method_state(state)
        if not cleaned:
            return False

        dep_method = cleaned.get("deposit_method", "")
        if dep_method:
            idx = self.deposit_method_combo.findText(dep_method, Qt.MatchFixedString)
            if idx >= 0:
                self.deposit_method_combo.setCurrentIndex(idx)

        etchant = cleaned.get("etchant", "")
        if etchant:
            self._set_etchant_combo_value(etchant)

        etch_mode = cleaned.get("etch_mode", "")
        if etch_mode in ("isotropic", "anisotropic"):
            self.current_etch_mode = etch_mode
            if hasattr(self, "isotropic_etch_checkbox") and self.isotropic_etch_checkbox is not None:
                block = self.isotropic_etch_checkbox.blockSignals(True)
                try:
                    self.isotropic_etch_checkbox.setChecked(etch_mode == "isotropic")
                finally:
                    self.isotropic_etch_checkbox.blockSignals(block)

        if "temperature" in cleaned:
            t_val = float(cleaned["temperature"])
            t_val = max(self.temp_spin.minimum(), min(self.temp_spin.maximum(), t_val))
            self.temp_spin.setValue(t_val)

        if "time" in cleaned:
            e_val = float(cleaned["time"])
            e_val = max(self.exposure_spin.minimum(), min(self.exposure_spin.maximum(), e_val))
            self.exposure_spin.setValue(e_val)

        params = cleaned.get("params", {})
        if isinstance(params, dict):
            for name, val in params.items():
                widget = self.custom_param_widgets.get(name)
                if widget is None:
                    continue
                try:
                    w_val = float(val)
                except Exception:
                    continue
                w_val = max(widget.minimum(), min(widget.maximum(), w_val))
                widget.setValue(w_val)

        return True

    def save_default_method_for_current_operation(self):
        op = self.op_combo.currentText()
        if op not in self._supported_method_operations():
            QMessageBox.information(
                self,
                "Default unavailable",
                "This step has no method settings to save.",
            )
            return

        state = self._capture_current_method_state()
        if not state:
            QMessageBox.warning(self, "Nothing to save", "Set at least one method value first.")
            return

        self.default_method_settings[op] = state
        self.save_default_method_settings()
        self.statusBar().showMessage(
            f"Default method saved for {operation_label(op)}.",
            3500,
        )

    def load_default_method_for_current_operation(self, notify_when_missing: bool = True):
        op = self.op_combo.currentText()
        state = self._default_method_state_for_operation(op)
        if not state:
            if notify_when_missing:
                QMessageBox.information(
                    self,
                    "No default saved",
                    f"Save a default for {operation_label(op)} first.",
                )
            return False
        self._apply_method_state_to_widgets(state)
        return True

    def refresh_method_recipe_combo(self, preferred_name: str = ""):
        if not hasattr(self, "method_recipe_combo"):
            return

        combo = self.method_recipe_combo
        current = preferred_name.strip() if preferred_name else (combo.currentText() or "").strip()
        names = sorted(self.method_recipes.keys(), key=lambda x: x.lower())

        block = combo.blockSignals(True)
        try:
            combo.clear()
            for name in names:
                combo.addItem(name)
            if combo.count() > 0:
                idx = combo.findText(current, Qt.MatchFixedString)
                if idx < 0:
                    idx = 0
                combo.setCurrentIndex(idx)
        finally:
            combo.blockSignals(block)

        has_recipes = combo.count() > 0
        if hasattr(self, "load_method_recipe_btn"):
            self.load_method_recipe_btn.setEnabled(has_recipes)
        if hasattr(self, "delete_method_recipe_btn"):
            self.delete_method_recipe_btn.setEnabled(has_recipes)

    def save_method_recipe_from_dialog(self):
        if not hasattr(self, "method_recipe_name_edit"):
            return

        name = (self.method_recipe_name_edit.text() or "").strip()
        if not name:
            QMessageBox.warning(self, "Name required", "Enter a recipe name.")
            return

        if name in self.method_recipes:
            reply = QMessageBox.question(
                self,
                f"Replace “{name}”?",
                "A recipe with this name already exists.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        self.method_recipes[name] = {
            "operation": self.op_combo.currentText(),
            "state": self._capture_current_method_state(),
        }
        self.save_method_recipes()
        self.refresh_method_recipe_combo(preferred_name=name)

    def load_selected_method_recipe(self):
        if not hasattr(self, "method_recipe_combo"):
            return

        name = (self.method_recipe_combo.currentText() or "").strip()
        if not name or name not in self.method_recipes:
            QMessageBox.information(self, "Select a recipe", "Choose a saved recipe first.")
            return

        recipe = self.method_recipes.get(name, {})
        target_op = str(recipe.get("operation", "")).strip()
        if target_op and target_op in self._supported_method_operations():
            idx = self.op_combo.findText(target_op, Qt.MatchFixedString)
            if idx >= 0 and idx != self.op_combo.currentIndex():
                self.op_combo.setCurrentIndex(idx)

        self._apply_method_state_to_widgets(recipe.get("state", {}))
        self.open_operation_parameters_dialog()

    def delete_selected_method_recipe(self):
        if not hasattr(self, "method_recipe_combo"):
            return

        name = (self.method_recipe_combo.currentText() or "").strip()
        if not name or name not in self.method_recipes:
            QMessageBox.information(self, "Select a recipe", "Choose a saved recipe first.")
            return

        reply = QMessageBox.question(
            self,
            f"Delete “{name}”?",
            "This recipe will be removed.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self.method_recipes.pop(name, None)
        self.save_method_recipes()
        self.refresh_method_recipe_combo()

    # ---------- Wafer / History ----------
    def restore_before_reset_wafer(self):
        idx = int(getattr(self, "active_project_tab_index", -1))
        if idx < 0 or idx >= len(self.project_sessions):
            return
        sid = id(self.project_sessions[idx])
        backup = self._reset_restore_info_by_tab.get(sid)
        if not isinstance(backup, dict):
            QMessageBox.information(
                self,
                "Nothing to restore",
                "This process has no state from before a reset."
            )
            return

        data = self._deepcopy_process_data(backup.get("data", {}))
        path = backup.get("path")
        self._load_process_dict(data)
        self.current_process_path = path

        session = self.project_sessions[idx]
        session["data"] = self._deepcopy_process_data(data)
        session["path"] = path
        if path:
            session["name"] = os.path.basename(path)
        current_hash = self._process_data_hash(session["data"])
        if session.get("saved_hash") is None:
            session["saved_hash"] = current_hash
        session["dirty"] = (current_hash != session.get("saved_hash"))
        self._update_project_tab_ui(idx)
        self._refresh_window_title_from_active_tab()

        self._reset_restore_info_by_tab.pop(sid, None)
        self._update_restore_reset_action_state()
        self.statusBar().showMessage("Restored the process state from before reset.", 3500)

    def reset_wafer(self):
        idx = int(getattr(self, "active_project_tab_index", -1))
        if self.xsec is not None and self.history and 0 <= idx < len(self.project_sessions):
            sid = id(self.project_sessions[idx])
            self._reset_restore_info_by_tab[sid] = {
                "data": self._deepcopy_process_data(self._make_process_dict()),
                "path": self.current_process_path,
            }

        width_um = WIDTH_UM_DEFAULT
        substrate_name = self.substrate_combo.currentText()
        sub_thick_um = self.substrate_thick_spin.value()
        height_um = self.process_grid_height_um_for_substrate(sub_thick_um)
        substrate_index = self.material_index(substrate_name)
        if substrate_index < 0:
            substrate_index = 1
        if sub_thick_um > height_um:
            sub_thick_um = height_um
            self.substrate_thick_spin.setValue(height_um)

        self.text_pos = (0.5, -0.08)
        self.material_legend_pos = (0.8, 0.0)
        self.material_legend_artist = None
        self.dragging_material_legend = False
        self.material_legend_drag_offset = (0.0, 0.0)
        self.substrate_text_pos = None
        self.layer_text_positions = {}
        self.layer_text_artists = {}
        self.dragging_layer_idx = None

        height_px = self.um_to_process_px(height_um, mode="ceil", minimum=1)
        sub_thick_px = self.um_to_process_px(sub_thick_um, mode="ceil", minimum=1)
        width_px = width_um

        self.xsec = CrossSection(
            width_px=width_px, height_px=height_px,
            substrate_index=substrate_index, void_index=self.VOID_INDEX,
            substrate_thickness_px=sub_thick_px,
            substrate_thickness_um=sub_thick_um,
            top_headroom_um=max(0.0, float(height_um) - float(sub_thick_um)),
        )

        self.history = []
        self.process_steps = []
        self.process_steps_replay_enabled = False
        self.visualisation_mode = "dynamic"
        self.visual_thickness_overrides = {}
        self.visual_thickness_step_overrides = {}
        self.redo_stack = []
        self.history_list.clear()
        self.clear_litho_windows(refresh_plot=False)
        self.clear_bond_source_selection()

        init_title = f"Substrate: {substrate_name} — {sub_thick_um:g} µm"
        self.save_state(init_title)

        self.update_plot(init_title)

        self.refresh_materials_list()
        self._snapshot_active_tab_state()
        self._update_restore_reset_action_state()
        self._update_visualisation_controls_state()

    def _base_process_cross_section(self):
        substrate_name = self.substrate_combo.currentText() or "Silicon (Si)"
        substrate_index = self.material_index(substrate_name)
        if substrate_index < 0:
            substrate_index = 1

        sub_thick_um = self.substrate_thick_spin.value()
        height_um = self.process_grid_height_um_for_substrate(sub_thick_um)
        if sub_thick_um > height_um:
            sub_thick_um = height_um

        xsec = CrossSection(
            width_px=WIDTH_UM_DEFAULT,
            height_px=self.um_to_process_px(height_um, mode="ceil", minimum=1),
            substrate_index=substrate_index,
            void_index=self.VOID_INDEX,
            substrate_thickness_px=self.um_to_process_px(sub_thick_um, mode="ceil", minimum=1),
            substrate_thickness_um=sub_thick_um,
            top_headroom_um=max(0.0, float(height_um) - float(sub_thick_um)),
        )
        return xsec, f"Substrate: {substrate_name} — {sub_thick_um:g} µm"

    def _begin_substrate_control_update(self):
        self._substrate_change_guard = int(getattr(self, "_substrate_change_guard", 0)) + 1

    def _end_substrate_control_update(self):
        self._substrate_change_guard = max(0, int(getattr(self, "_substrate_change_guard", 0)) - 1)

    def _substrate_control_updates_blocked(self) -> bool:
        return int(getattr(self, "_substrate_change_guard", 0)) > 0

    def _initial_history_substrate(self):
        history = getattr(self, "history", []) or []
        if not history or not isinstance(history[0], dict):
            return None
        state = history[0]
        grid = state.get("grid")
        heights = state.get("row_heights_um")
        if grid is None or heights is None:
            return None
        grid = np.asarray(grid)
        heights = np.asarray(heights, dtype=float)
        if grid.ndim != 2 or heights.ndim != 1 or heights.shape[0] != grid.shape[0]:
            return None

        totals = {}
        for material_idx in np.unique(grid):
            material_idx = int(material_idx)
            if material_idx == self.VOID_INDEX:
                continue
            total = 0.0
            for row_idx in np.where(np.any(grid == material_idx, axis=1))[0]:
                if np.any(grid[int(row_idx), :] == material_idx):
                    total += float(heights[int(row_idx)])
            if total > 0.0:
                totals[material_idx] = total
        if not totals:
            return None

        material_idx = max(totals, key=totals.get)
        material_name = (
            self.material_names[material_idx]
            if 0 <= material_idx < len(getattr(self, "material_names", []))
            else ""
        )
        return material_name, float(totals[material_idx])

    def _set_substrate_controls_silently(self, material_name: str, thickness_um):
        self._begin_substrate_control_update()
        try:
            if material_name and hasattr(self, "substrate_combo"):
                idx = self.substrate_combo.findText(material_name)
                if idx >= 0:
                    self.substrate_combo.setCurrentIndex(idx)
            if thickness_um is not None and hasattr(self, "substrate_thick_spin"):
                self.substrate_thick_spin.setValue(int(round(float(thickness_um))))
        finally:
            self._end_substrate_control_update()

    def _current_substrate_matches_initial_history(self) -> bool:
        initial = self._initial_history_substrate()
        if initial is None:
            return False
        material_name, thickness_um = initial
        current_name = str(self.substrate_combo.currentText() or "")
        try:
            current_thickness = float(self.substrate_thick_spin.value())
        except (TypeError, ValueError):
            return False
        return material_name == current_name and abs(float(thickness_um) - current_thickness) <= 1e-9

    def _ensure_process_steps_for_substrate_rebuild(self) -> bool:
        if self._process_replay_enabled():
            return True

        steps = self.process_steps_from_history_metadata(getattr(self, "history", []) or [])
        if steps is None:
            return False

        self.process_steps = [dict(step) for step in steps]
        self.process_steps_replay_enabled = True
        return True

    def apply_substrate_change_process_wide(
        self,
        *,
        active_index: int = None,
        warn_on_failure: bool = True,
        snapshot: bool = True,
    ) -> bool:
        if self._substrate_control_updates_blocked():
            return True

        if self._current_substrate_matches_initial_history():
            return True

        previous_substrate = self._initial_history_substrate()
        if not self._ensure_process_steps_for_substrate_rebuild():
            if previous_substrate is not None:
                self._set_substrate_controls_silently(*previous_substrate)
            if warn_on_failure:
                QMessageBox.warning(
                    self,
                    "Start a new process to change the substrate",
                    "This older process cannot update every step safely. Reset the process to use the selected substrate.",
                )
            return False

        if active_index is None:
            active_index = self._active_history_index_for_view() if hasattr(self, "_active_history_index_for_view") else None
        if hasattr(self, "reset_visualisation_for_process_change"):
            self.reset_visualisation_for_process_change()
        self.redo_stack = []
        self.rebuild_process_from_steps(active_index=active_index, update_view=True)

        if snapshot and hasattr(self, "_snapshot_active_tab_state"):
            self._snapshot_active_tab_state()
        if hasattr(self, "statusBar"):
            self.statusBar().showMessage("Substrate updated in all steps.", 3000)
        return True

    def on_substrate_changed(self):
        if self._substrate_control_updates_blocked():
            return
        if not hasattr(self, "substrate_combo") or not hasattr(self, "substrate_thick_spin"):
            return
        if self.xsec is None and not getattr(self, "history", None):
            return
        self.apply_substrate_change_process_wide(warn_on_failure=True, snapshot=True)

    def _process_replay_enabled(self) -> bool:
        return bool(getattr(self, "process_steps_replay_enabled", False))

    def _history_metadata_from_process_step(self, step: dict) -> dict:
        metadata = {}
        for key, value in step.items():
            if key in {"title", "bond_source_info"}:
                continue
            metadata[key] = value
        return metadata

    def _xsec_snapshot_state(self, title: str, extra: dict = None) -> dict:
        state = {"grid": self.xsec.copy_state(), "title": title}
        if hasattr(self.xsec, "copy_row_heights"):
            state["row_heights_um"] = self.xsec.copy_row_heights()
        if hasattr(self.xsec, "copy_visual_geometry"):
            state["visual_geometry"] = self.xsec.copy_visual_geometry()
        if hasattr(self.xsec, "copy_structural_roles"):
            state["structural_role_grid"] = self.xsec.copy_structural_roles()
        if hasattr(self.xsec, "copy_structural_bodies"):
            state["structural_bodies"] = self.xsec.copy_structural_bodies()
        if hasattr(self.xsec, "copy_planarizer_roles"):
            state["planarizer_role_grid"] = self.xsec.copy_planarizer_roles()
        if hasattr(self.xsec, "copy_planarizer_metadata"):
            state["planarizer_roles"] = self.xsec.copy_planarizer_metadata()
        if isinstance(extra, dict):
            state.update(extra)
        return state

    def _restore_xsec_snapshot(self, state: dict):
        self.xsec.restore_state(
            state["grid"],
            state.get("row_heights_um"),
            state.get("visual_geometry"),
            state.get("structural_role_grid"),
            state.get("structural_bodies"),
            state.get("planarizer_role_grid"),
            state.get("planarizer_roles"),
        )

    def process_steps_from_history_metadata(self, history):
        if not isinstance(history, list) or len(history) <= 1:
            return []

        steps = []
        for state in history[1:]:
            if not isinstance(state, dict):
                return None
            op = str(state.get("operation", "")).strip()
            if not op:
                return None

            step = {
                "operation": op,
            }
            material_name = str(state.get("material_name", "")).strip()
            if material_name:
                step["material_name"] = material_name
            method = str(state.get("method", "")).strip()
            if method:
                step["method"] = method

            if op in {"Deposit", "Spincoating", "Bond"}:
                if "thickness_um" not in state:
                    return None
                step["thickness_um"] = float(state.get("thickness_um", 0.0))
                if (
                    "bond_source" in state
                    and not isinstance(state.get("bond_source_info"), dict)
                ):
                    return None
                if op == "Bond" and isinstance(state.get("bond_source_info"), dict):
                    if hasattr(self, "_copy_bond_source_info"):
                        step["bond_source_info"] = self._copy_bond_source_info(state.get("bond_source_info"))
                    else:
                        step["bond_source_info"] = dict(state.get("bond_source_info"))
            elif op == "Etch":
                if "depth_um" not in state:
                    return None
                step["depth_um"] = float(state.get("depth_um", 0.0))
                step["etch_mode"] = str(state.get("etch_mode", "anisotropic") or "anisotropic")
            elif op == "Lithography":
                windows = state.get("windows", [])
                if isinstance(windows, list) and windows:
                    step["windows"] = [(float(s), float(e)) for s, e in windows]
                else:
                    step["window_start"] = float(state.get("window_start", 0.35))
                    step["window_end"] = float(state.get("window_end", 0.65))
                if "invert_lithography" in state:
                    step["invert_lithography"] = bool(state.get("invert_lithography"))
            elif op in {"Strip material", "Lift-off", "Bake/Annealing", "Clean / Surface prep", "Flip"}:
                pass
            else:
                return None

            old_title = str(state.get("title", "")).strip()
            generated_title = self._generated_process_step_title(step)
            step["title"] = (
                generated_title
                if self._history_title_looks_generated(old_title, op)
                else (old_title or generated_title)
            )
            steps.append(step)
        return steps

    def _format_step_um(self, value) -> str:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return str(value)
        text = f"{value:.9f}".rstrip("0").rstrip(".")
        return text if text else "0"

    def _history_title_looks_generated(self, title: str, operation: str) -> bool:
        title = str(title or "").strip()
        if not title:
            return True
        low = title.lower()
        operation = str(operation or "").strip().lower()
        generated_prefixes = (
            "deposit ",
            "spincoating ",
            "spin coat ",
            "bond ",
            "etch ",
            "pr lithography",
            "lithography and development",
            "strip ",
            "lift-off ",
            "clean/surface prep ",
            "clean / surface prep ",
            "bake/annealing ",
            "bake / anneal ",
            "flip",
        )
        if any(low.startswith(prefix) for prefix in generated_prefixes):
            return True
        return low == operation

    def _generated_process_step_title(self, step: dict) -> str:
        op = str(step.get("operation", "") or "").strip()
        material = str(step.get("material_name", "") or "").strip()
        method = str(step.get("method", "") or "").strip()

        if op in {"Deposit", "Spincoating", "Bond"}:
            title = (
                f"{operation_label(op)} {material} "
                f"({self._format_step_um(step.get('thickness_um', 0.0))} µm)"
            ).strip()
            return f"{title} — {method}" if method else title
        if op == "Etch":
            mode = str(step.get("etch_mode", "anisotropic") or "anisotropic").strip().lower()
            mode = "isotropic" if mode == "isotropic" else "anisotropic"
            title = f"Etch {material} ({self._format_step_um(step.get('depth_um', 0.0))} µm) {mode}".strip()
            return f"{title} — {method}" if method else title
        if op == "Lithography":
            return "Lithography and development"
        if op in {"Strip material", "Lift-off", "Bake/Annealing"}:
            title = f"{operation_label(op)} {material}".strip()
            return f"{title} — {method}" if method else title
        if op == "Clean / Surface prep":
            target = material or "wafer"
            title = f"Clean / surface prep {target}".strip()
            return f"{title} — {method}" if method else title
        if op == "Flip":
            return "Flip stack"
        return op or "Step"

    def _apply_process_step_to_current_xsec(self, step: dict) -> bool:
        return apply_process_step_to_current_xsec(self, step)

    def rebuild_process_from_steps(self, *, active_index: int = None, update_view: bool = True):
        if not self._process_replay_enabled():
            return

        self.xsec, init_title = self._base_process_cross_section()
        rebuilt_history = [self._xsec_snapshot_state(init_title)]

        for index, step in enumerate(list(getattr(self, "process_steps", [])), 1):
            if not isinstance(step, dict):
                raise ValueError(f"Process step {index} is malformed.")
            if not self._apply_process_step_to_current_xsec(step):
                operation = str(step.get("operation", "unknown") or "unknown")
                raise ValueError(
                    f"Process step {index} ({operation}) could not be replayed."
                )
            title = str(step.get("title", "")).strip() or str(step.get("operation", "Step")).strip() or "Step"
            state = self._xsec_snapshot_state(title)
            state.update(self._history_metadata_from_process_step(step))
            rebuilt_history.append(state)

        self.history = rebuilt_history
        if hasattr(self, "history_list") and self.history_list is not None:
            block = self.history_list.blockSignals(True)
            try:
                self.history_list.clear()
                for st in self.history:
                    self.history_list.addItem(st.get("title", ""))
                if active_index is None:
                    active_index = len(self.history) - 1
                active_index = max(0, min(int(active_index), len(self.history) - 1))
                self.history_list.setCurrentRow(active_index)
            finally:
                self.history_list.blockSignals(block)

        if update_view and self.history:
            active = self.history[-1] if active_index is None else self.history[max(0, min(int(active_index), len(self.history) - 1))]
            self._restore_xsec_snapshot(active)
            self.update_plot(active["title"])

    def record_process_step(self, title: str, step: dict):
        if hasattr(self, "reset_visualisation_for_process_change"):
            self.reset_visualisation_for_process_change()
        if not self._process_replay_enabled():
            self.save_state(title, extra=step)
            self.update_plot(title)
            return
        step_record = dict(step or {})
        step_record["title"] = title
        self.process_steps.append(step_record)
        self.redo_stack = []
        self.rebuild_process_from_steps(active_index=len(self.process_steps), update_view=True)

    def flip_active_stack_from_top(self):
        if self.xsec is None:
            return
        if not execute_ui_flip_operation(self):
            return
        title = "Flip stack"
        self.record_process_step(title, {"operation": "Flip"})

    def save_state(self, title: str, extra: dict = None):
        if self.xsec is None:
            return
        self.redo_stack = []
        state = self._xsec_snapshot_state(title, extra)
        self.history.append(state)
        self.history_list.addItem(title)
        self.history_list.setCurrentRow(len(self.history) - 1)

    def rename_step(self, item):
        row = self.history_list.row(item)
        old = item.text()
        new, ok = QInputDialog.getText(self, "Rename step", "Step name:", text=old)
        if not ok or not new.strip():
            return
        new = new.strip()
        item.setText(new)
        self.history[row]["title"] = new
        if self._process_replay_enabled() and row > 0:
            step_idx = row - 1
            if 0 <= step_idx < len(self.process_steps):
                self.process_steps[step_idx]["title"] = new
        if row == self.history_list.currentRow():
            self.update_plot(new)

    def on_history_row_changed(self, row: int):
        if row < 0 or row >= len(self.history):
            return
        state = self.history[row]
        self._restore_xsec_snapshot(state)
        self.update_plot(state["title"])

    def undo_last_step(self):
        if self._process_replay_enabled():
            if not self.process_steps:
                return
            if hasattr(self, "reset_visualisation_for_process_change"):
                self.reset_visualisation_for_process_change()
            step = self.process_steps.pop()
            self.redo_stack.append(step)
            self.rebuild_process_from_steps(active_index=len(self.process_steps), update_view=True)
            return

        if len(self.history) <= 1:
            return
        if hasattr(self, "reset_visualisation_for_process_change"):
            self.reset_visualisation_for_process_change()
        state = self.history.pop()
        self.redo_stack.append(state)
        self.history_list.takeItem(self.history_list.count() - 1)
        last = self.history[-1]
        self._restore_xsec_snapshot(last)
        self.update_plot(last["title"])
        self.history_list.setCurrentRow(len(self.history) - 1)

    def redo_step(self):
        if self._process_replay_enabled():
            if not self.redo_stack:
                return
            if hasattr(self, "reset_visualisation_for_process_change"):
                self.reset_visualisation_for_process_change()
            step = self.redo_stack.pop()
            self.process_steps.append(step)
            self.rebuild_process_from_steps(active_index=len(self.process_steps), update_view=True)
            return

        if not self.redo_stack:
            return
        if hasattr(self, "reset_visualisation_for_process_change"):
            self.reset_visualisation_for_process_change()
        state = self.redo_stack.pop()
        self.history.append(state)
        self.history_list.addItem(state["title"])
        self.history_list.setCurrentRow(len(self.history) - 1)
        self._restore_xsec_snapshot(state)
        self.update_plot(state["title"])
