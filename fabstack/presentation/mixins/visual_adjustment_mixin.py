"""Display-only visual override and adjustment behavior."""

import io
import logging
import math
import os
import tempfile

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)
try:
    from PIL import Image
except ImportError:
    Image = None

from fabstack.config.visualization import (
    EXPORT_PREVIEW_SUBPLOTS,
    FEATURE_EMPHASIS_DEFAULT,
    FEATURE_EMPHASIS_MAX,
    FEATURE_EMPHASIS_MIN,
    PREVIEW_EXPORT_AS_MAIN_VIEW,
    VISUAL_SCALE_MAX,
    VISUAL_SCALE_MIN,
)
from fabstack.materials.chemistry import chem_subs_unicode, legend_label
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
from fabstack.resources import LOGO_PATH

try:
    from docx import Document
    from docx.shared import Inches
except ImportError:
    Document = None

LOGGER = logging.getLogger("fabstack.presentation.mixins.plotting_export_mixin")


class VisualAdjustmentMixin:
    def _visual_thickness_overrides(self):
        overrides = getattr(self, "visual_thickness_overrides", {}) or {}
        if not isinstance(overrides, dict):
            return {}
        cleaned = {}
        for key, value in overrides.items():
            try:
                target = int(value)
            except (TypeError, ValueError):
                continue
            if target > 0:
                cleaned[str(key)] = int(target)
        return cleaned

    def _visual_thickness_step_overrides(self):
        overrides = getattr(self, "visual_thickness_step_overrides", {}) or {}
        if not isinstance(overrides, dict):
            return {}
        cleaned = {}
        for idx, values in overrides.items():
            if not isinstance(values, dict):
                continue
            step_values = {}
            for key, value in values.items():
                try:
                    target = int(value)
                except (TypeError, ValueError):
                    continue
                if target > 0:
                    step_values[str(key)] = int(target)
            if step_values:
                cleaned[str(idx)] = step_values
        return cleaned

    def _visual_step_override_values(self, history_index=None):
        try:
            idx = int(history_index)
        except (TypeError, ValueError):
            return {}
        if idx < 0:
            return {}
        return self._visual_thickness_step_overrides().get(str(idx), {})

    def _visual_override_value_for_key(self, key: str, history_index=None):
        if not self._visual_overrides_enabled():
            return None
        key = str(key or "")
        if not key:
            return None
        step_value = self._visual_step_override_values(history_index).get(key)
        if step_value is not None:
            return int(step_value)
        global_value = self._visual_thickness_overrides().get(key)
        if global_value is not None:
            return int(global_value)
        return None

    def _display_height_for_current_view(self, *, ignore_visual_overrides: bool = False) -> int:
        if getattr(self, "xsec", None) is None:
            return 0
        grid = self.xsec.grid
        history_index = self._active_history_index_for_view()
        row_heights = self._row_heights_for_plot_grid(grid, history_index)
        saved_info = getattr(self, "_last_display_remap_info", None)
        saved_overrides = getattr(self, "visual_thickness_overrides", {})
        saved_step_overrides = getattr(self, "visual_thickness_step_overrides", {})
        try:
            if ignore_visual_overrides:
                self.visual_thickness_overrides = {}
                self.visual_thickness_step_overrides = {}
            display_grid = self._display_grid_auto(
                grid,
                row_heights_um=row_heights,
                history_index=history_index,
            )
            return int(np.asarray(display_grid).shape[0])
        finally:
            if ignore_visual_overrides:
                self.visual_thickness_overrides = saved_overrides
                self.visual_thickness_step_overrides = saved_step_overrides
            if saved_info is not None:
                self._last_display_remap_info = saved_info

    def _visual_override_canvas_height_scale(self) -> float:
        if getattr(self, "xsec", None) is None:
            return 1.0

        height = 0
        cached = getattr(self, "_last_main_display_grid", None)
        try:
            if cached is not None:
                cached = np.asarray(cached)
                if cached.ndim == 2:
                    height = int(cached.shape[0])
        except Exception:
            height = 0

        if height <= 0:
            try:
                height = int(self._display_height_for_current_view())
            except Exception:
                height = 0
        if height <= 0:
            return 1.0

        baseline = float(DYNAMIC_TARGET_STACK_PX + DISPLAY_TOP_HEADROOM_PX)
        scale = float(height) / max(1.0, baseline)
        return max(1.0, min(float(MAX_AUTO_CANVAS_HEIGHT_SCALE), scale))

    def _display_photoresist_indices(self):
        names = getattr(self, "material_names", []) or []
        indices = set()
        for idx, name in enumerate(names):
            low = str(name or "").lower()
            if "photoresist" in low or "su-8" in low or low.startswith("pr_"):
                indices.add(int(idx))
        return indices

    def _active_history_index_for_view(self) -> int:
        if not getattr(self, "history", None):
            return -1
        row = -1
        if hasattr(self, "history_list") and self.history_list is not None:
            try:
                row = int(self.history_list.currentRow())
            except Exception:
                row = -1
        if 0 <= row < len(self.history):
            return row
        return len(self.history) - 1

    def _normalised_visualisation_mode(self) -> str:
        mode = str(getattr(self, "visualisation_mode", "dynamic") or "dynamic").strip().lower()
        return mode if mode in {"dynamic", "pending", "finalised"} else "dynamic"

    def _visual_overrides_enabled(self) -> bool:
        return self._normalised_visualisation_mode() in {"pending", "finalised"}

    def _display_segment_target_height(
        self,
        height_um: float,
        *,
        material_idx=None,
        override_key=None,
        history_index=None,
        process_profile=None,
        is_substrate_core=False,
        structural_body_id=None,
        structural_nominal_um=None,
        minimum_substrate_removal_rows=0,
        preferred_substrate_removal_rows=None,
    ) -> int:
        try:
            height_um = float(height_um)
        except (TypeError, ValueError):
            return 0
        if not np.isfinite(height_um) or height_um <= 1e-12:
            return 0

        if material_idx is not None:
            key = override_key or self._display_material_count_profile_key(material_idx, height_um)
            override_value = self._visual_override_value_for_key(key, history_index=history_index)
            if override_value is not None:
                return max(1, int(override_value))

        if bool(is_substrate_core):
            return self._display_substrate_core_target_height(
                height_um,
                process_profile,
                structural_body_id=structural_body_id,
                structural_nominal_um=structural_nominal_um,
                minimum_removed_rows=minimum_substrate_removal_rows,
                preferred_removed_rows=preferred_substrate_removal_rows,
            )

        if material_idx is not None and int(material_idx) == int(getattr(self, "VOID_INDEX", 0)):
            return self._display_profile_feature_rows(height_um, process_profile)

        return self._display_profile_feature_rows(height_um, process_profile)

    def _display_physical_target_height(self, total_um: float) -> int:
        total_um = max(0.0, float(total_um))
        if total_um <= 0.0:
            return 0
        return self._display_segment_target_height(total_um)

    def _format_visual_thickness_um(self, value) -> str:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return str(value)
        text = f"{value:.9f}".rstrip("0").rstrip(".")
        return text if text else "0"

    def _clear_visual_thickness_highlight(self, redraw: bool = False):
        artists = list(getattr(self, "visual_thickness_highlight_artists", []) or [])
        for artist in artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.visual_thickness_highlight_artists = []
        if redraw and hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.draw_idle()

    def _display_row_from_plot_y(self, ydata, display_height: int):
        try:
            ydata = float(ydata)
        except (TypeError, ValueError):
            return None
        if display_height <= 0:
            return None
        scale = self._visual_render_scale_factor()
        bottom_y = float(display_height) - 0.5
        unscaled_y = bottom_y - (bottom_y - ydata) / max(scale, 1e-12)
        row = int(np.floor(unscaled_y + 0.5))
        if row < 0 or row >= display_height:
            return None
        return row

    def _visual_layer_pick_from_event(self, event):
        """
        Resolve a canvas click to the exact physical layer segment under it.

        `column_segments` (built alongside the fill in
        `_render_material_columns_from_targets`/`_display_grid_row_bands`) is
        the same data the base render used to place pixels, so this lookup
        can't disagree with what's on screen the way the old dual-path
        (direct column run vs. a separately-computed "band" fallback) could.
        """
        if event is None or event.inaxes != getattr(self.canvas, "ax", None):
            return None
        if event.xdata is None or event.ydata is None:
            return None

        display_grid = getattr(self, "_last_main_display_grid", None)

        if display_grid is None:
            return None
        display_grid = np.asarray(display_grid)
        if display_grid.ndim != 2 or display_grid.size == 0:
            return None

        col = int(np.floor(float(event.xdata) + 0.5))
        row = self._display_row_from_plot_y(event.ydata, int(display_grid.shape[0]))
        if row is None or col < 0 or col >= display_grid.shape[1]:
            return None

        material_idx = int(display_grid[row, col])

        history_index = getattr(self, "_last_main_display_history_index", None)
        try:
            history_index = int(history_index)
        except (TypeError, ValueError):
            history_index = self._active_history_index_for_view()

        info = getattr(self, "_last_main_display_remap_info", {}) or {}
        columns = info.get("column_segments") or []
        if col >= len(columns):
            return None

        segment = None
        for seg in columns[col]:
            if int(seg.get("display_start", -1)) <= row < int(seg.get("display_end", -1)):
                segment = seg
                break
        if segment is None or int(segment.get("material", self.VOID_INDEX)) != material_idx:
            return None

        height_um = max(0.0, float(segment.get("y1_um", 0.0)) - float(segment.get("y0_um", 0.0)))
        if height_um <= 1e-12:
            return None

        key = self._display_material_count_profile_key(material_idx, height_um)
        if material_idx == self.VOID_INDEX:
            material_name = "Etched void"
        else:
            material_name = (
                self.material_names[material_idx]
                if 0 <= material_idx < len(getattr(self, "material_names", []))
                else f"Material {material_idx}"
            )
        return {
            "key": key,
            "material_idx": material_idx,
            "material_name": material_name,
            "height_um": height_um,
            "current_px": max(1, int(segment.get("display_end", 0)) - int(segment.get("display_start", 0))),
            "display_start": int(segment.get("display_start", 0)),
            "display_end": int(segment.get("display_end", 0)),
            "history_index": history_index,
            "display_grid": display_grid,
        }

    def _highlight_visual_layer_pick(self, pick):
        self._clear_visual_thickness_highlight(redraw=False)
        if not pick or not hasattr(self, "canvas") or self.canvas is None:
            return

        display_grid = np.asarray(pick.get("display_grid"))
        if display_grid.ndim != 2 or display_grid.size == 0:
            return
        material_idx = int(pick.get("material_idx", self.VOID_INDEX))
        start = max(0, min(int(pick.get("display_start", 0)), display_grid.shape[0]))
        end = max(start, min(int(pick.get("display_end", start)), display_grid.shape[0]))
        if end <= start:
            return

        mask = np.any(display_grid[start:end, :] == material_idx, axis=0)
        cols = np.where(mask)[0]
        if cols.size == 0:
            return

        h = int(display_grid.shape[0])
        scale = self._visual_scale_factor_clamped()
        y_top = self._plot_y_for_visual_scale(float(start) - 0.5, h, scale)
        y_bottom = self._plot_y_for_visual_scale(float(end) - 0.5, h, scale)
        rect_y = min(y_top, y_bottom)
        rect_h = abs(y_bottom - y_top)
        if rect_h <= 0.0:
            rect_h = 1e-6

        segments = []
        seg_start = int(cols[0])
        prev = int(cols[0])
        for col in cols[1:]:
            col = int(col)
            if col == prev + 1:
                prev = col
                continue
            segments.append((seg_start, prev))
            seg_start = prev = col
        segments.append((seg_start, prev))

        artists = []
        for seg_start, seg_end in segments:
            rect = mpatches.Rectangle(
                (float(seg_start) - 0.5, rect_y),
                float(seg_end - seg_start + 1),
                rect_h,
                facecolor="#facc15",
                edgecolor="#b45309",
                linewidth=1.6,
                alpha=0.28,
                zorder=9,
            )
            rect.set_gid("visual_thickness_highlight")
            self.canvas.ax.add_patch(rect)
            artists.append(rect)
        self.visual_thickness_highlight_artists = artists
        self.canvas.draw_idle()

    def _set_process_visual_override(self, key: str, target_px: int) -> None:
        """
        Set a process-wide override for `key`: it applies at every step
        where this exact (material, thickness) combination occurs, not just
        the step where it was set - this is what makes a chosen visual
        thickness stay constant as the process grows, instead of vanishing
        the moment history moves past whichever step originally set it. Any
        step-scoped override for the same key is purged, since step-scoped
        always wins the lookup and would otherwise silently shadow this one
        at that one step.
        """
        key = str(key or "")
        if not key:
            return
        target_px = max(1, int(target_px))

        overrides = self._visual_thickness_overrides()
        overrides[key] = target_px
        self.visual_thickness_overrides = overrides

        step_overrides = self._visual_thickness_step_overrides()
        for step_key in list(step_overrides.keys()):
            values = dict(step_overrides.get(step_key, {}) or {})
            if key in values:
                values.pop(key, None)
            if values:
                step_overrides[step_key] = values
            else:
                step_overrides.pop(step_key, None)
        self.visual_thickness_step_overrides = step_overrides

    def _set_visual_thickness_override_from_pick(self, pick, target_px: int, scope: str):
        key = str((pick or {}).get("key", "") or "")
        if not key:
            return False
        target_px = max(1, int(target_px))

        if scope == "process":
            self._set_process_visual_override(key, target_px)
        else:
            try:
                history_index = int((pick or {}).get("history_index", self._active_history_index_for_view()))
            except (TypeError, ValueError):
                history_index = self._active_history_index_for_view()
            if history_index < 0:
                return False
            step_overrides = self._visual_thickness_step_overrides()
            values = dict(step_overrides.get(str(history_index), {}) or {})
            values[key] = target_px
            step_overrides[str(history_index)] = values
            self.visual_thickness_step_overrides = step_overrides

        self.visual_adjustment_notice_dismissed = False
        self.update_plot(getattr(self, "current_title", "") or "")
        if hasattr(self, "_snapshot_active_tab_state"):
            self._snapshot_active_tab_state()
        return True

    def _clear_visual_thickness_override_from_pick(self, pick) -> bool:
        """
        Remove both a process-wide and a step-scoped override for this
        pick's key, reverting it to the automatic dynamic scale. "Default"
        clears whichever scope is actually set rather than asking the user
        to know which one applied - the dialog itself never distinguishes.
        """
        key = str((pick or {}).get("key", "") or "")
        if not key:
            return False
        changed = False

        overrides = self._visual_thickness_overrides()
        if key in overrides:
            overrides.pop(key, None)
            self.visual_thickness_overrides = overrides
            changed = True

        try:
            history_index = int((pick or {}).get("history_index", self._active_history_index_for_view()))
        except (TypeError, ValueError):
            history_index = self._active_history_index_for_view()
        if history_index >= 0:
            step_overrides = self._visual_thickness_step_overrides()
            step_key = str(history_index)
            values = dict(step_overrides.get(step_key, {}) or {})
            if key in values:
                values.pop(key, None)
                changed = True
                if values:
                    step_overrides[step_key] = values
                else:
                    step_overrides.pop(step_key, None)
                self.visual_thickness_step_overrides = step_overrides

        if changed:
            self.visual_adjustment_notice_dismissed = False
            self.update_plot(getattr(self, "current_title", "") or "")
            if hasattr(self, "_snapshot_active_tab_state"):
                self._snapshot_active_tab_state()
        return changed

    def _show_visual_thickness_pick_dialog(self, pick):
        original_mode = self._normalised_visualisation_mode()
        original_process_overrides = dict(self._visual_thickness_overrides())
        original_step_overrides = dict(self._visual_thickness_step_overrides())

        # Manual edits retain the existing saved-data format. Only the selected
        # feature receives a display override.
        if original_mode != "finalised":
            if original_mode == "dynamic":
                self.visual_thickness_overrides = {}
                self.visual_thickness_step_overrides = {}
            self._set_visualisation_mode("finalised")

        dlg = QDialog(self)
        dlg.setWindowTitle("Adjust display height")
        layout = QVBoxLayout(dlg)

        material = str(pick.get("material_name", "Layer"))
        height_um = self._format_visual_thickness_um(pick.get("height_um", 0.0))
        current_px = max(1, int(pick.get("current_px", 1)))
        label = QLabel(f"{material}\nPhysical thickness: {height_um} µm")
        label.setWordWrap(True)
        layout.addWidget(label)

        spin = QSpinBox(dlg)
        spin.setRange(1, 1000)
        spin.setSingleStep(1)
        spin.setValue(current_px)
        display_form = QFormLayout()
        display_form.addRow("Display height:", spin)
        layout.addLayout(display_form)

        button_row = QHBoxLayout()
        apply_button = QPushButton("Save", dlg)
        default_button = QPushButton("Use automatic", dlg)

        cancel_button = QPushButton("Cancel", dlg)
        button_row.addWidget(apply_button)
        button_row.addWidget(default_button)
        button_row.addWidget(cancel_button)
        layout.addLayout(button_row)

        key = str((pick or {}).get("key", "") or "")
        keep_live_value = {"value": False}

        def restore_original():
            self.visual_thickness_overrides = dict(original_process_overrides)
            self.visual_thickness_step_overrides = dict(original_step_overrides)
            self.visual_adjustment_notice_dismissed = False
            self._set_visualisation_mode(original_mode)
            self.update_plot(getattr(self, "current_title", "") or "")

        def preview_live(value):
            if not key:
                return
            self._set_process_visual_override(key, int(value))
            self.visual_adjustment_notice_dismissed = False
            self.update_plot(getattr(self, "current_title", "") or "")

        def apply():
            keep_live_value["value"] = True
            preview_live(int(spin.value()))
            if hasattr(self, "_snapshot_active_tab_state"):
                self._snapshot_active_tab_state()
            if hasattr(self, "statusBar"):
                self.statusBar().showMessage("Display height updated.", 4000)
            dlg.accept()

        def reset_default():
            keep_live_value["value"] = True
            if original_mode == "dynamic":
                self.visual_thickness_overrides = {}
                self.visual_thickness_step_overrides = {}
                self.visual_adjustment_notice_dismissed = False
                self._set_visualisation_mode("dynamic")
                self.update_plot(getattr(self, "current_title", "") or "")
                cleared = True
            else:
                cleared = self._clear_visual_thickness_override_from_pick(pick)
            if hasattr(self, "statusBar"):
                msg = (
                    "Automatic display height restored."
                    if cleared
                    else "This feature already uses automatic display height."
                )
                self.statusBar().showMessage(msg, 4000)
            dlg.accept()

        def finish(result):
            if result != QDialog.Accepted and not keep_live_value["value"]:
                restore_original()
            self._clear_visual_thickness_highlight(redraw=True)

        spin.valueChanged.connect(preview_live)
        apply_button.clicked.connect(lambda _checked=False: apply())
        default_button.clicked.connect(lambda _checked=False: reset_default())
        cancel_button.clicked.connect(dlg.reject)
        dlg.finished.connect(finish)
        dlg.exec_()

    def _handle_visual_thickness_canvas_pick(self, event):
        pick = self._visual_layer_pick_from_event(event)
        if pick is None:
            if hasattr(self, "statusBar"):
                self.statusBar().showMessage(
                    "Select a layer or etched area to adjust its display height.",
                    5000,
                )
            return
        self.visual_thickness_pick_mode = False
        self._highlight_visual_layer_pick(pick)
        self._show_visual_thickness_pick_dialog(pick)

    def open_visual_thickness_dialog(self):
        if getattr(self, "xsec", None) is None:
            QMessageBox.information(
                self,
                "Nothing to adjust",
                "Create or open a process first.",
            )
            return
        self.visual_thickness_pick_mode = True
        self._clear_visual_thickness_highlight(redraw=True)
        if hasattr(self, "statusBar"):
            self.statusBar().showMessage(
                "Select a layer or etched area to adjust its display height.",
                8000,
            )
        return

    def _set_visualisation_mode(self, mode: str) -> None:
        mode = str(mode or "dynamic").strip().lower()
        self.visualisation_mode = mode if mode in {"dynamic", "pending", "finalised"} else "dynamic"
        if hasattr(self, "_update_visualisation_controls_state"):
            self._update_visualisation_controls_state()

    def _update_visualisation_controls_state(self):
        finalise_btn = getattr(self, "finalise_visualisation_button", None)
        adjust_btn = getattr(self, "adjust_visual_thickness_button", None)
        accept_btn = getattr(self, "accept_visualisation_button", None)
        reject_btn = getattr(self, "reject_visualisation_button", None)

        if finalise_btn is not None:
            finalise_btn.setVisible(False)
        if adjust_btn is not None:
            adjust_btn.setVisible(True)
            adjust_btn.setEnabled(getattr(self, "xsec", None) is not None)
        if accept_btn is not None:
            accept_btn.setVisible(False)
        if reject_btn is not None:
            reject_btn.setVisible(False)

    def _collect_dynamic_visual_overrides(self):
        saved_mode = getattr(self, "visualisation_mode", "dynamic")
        saved_overrides = getattr(self, "visual_thickness_overrides", {})
        saved_step_overrides = getattr(self, "visual_thickness_step_overrides", {})
        saved_info = getattr(self, "_last_display_remap_info", None)
        try:
            self.visualisation_mode = "dynamic"
            self.visual_thickness_overrides = {}
            self.visual_thickness_step_overrides = {}
            profile = self._process_display_adjustment_profile()
            overrides = {}
            for source in self._display_adjustment_sources():
                if len(source) >= 3:
                    history_index, grid, row_heights = source[:3]
                    structural_roles = source[3] if len(source) >= 4 else None
                else:
                    history_index, grid, row_heights = None, source[0], source[1]
                    structural_roles = None
                self._display_grid_auto(
                    np.asarray(grid),
                    row_heights_um=row_heights,
                    history_index=history_index,
                    process_profile=profile,
                    structural_role_grid=structural_roles,
                )
                info = getattr(self, "_last_display_remap_info", {}) or {}
                for column_segments in info.get("column_segments", []) or []:
                    for segment in column_segments or []:
                        material_idx = int(segment.get("material", self.VOID_INDEX))
                        height_um = max(
                            0.0,
                            float(segment.get("y1_um", 0.0)) - float(segment.get("y0_um", 0.0)),
                        )
                        display_px = int(segment.get("display_end", 0)) - int(segment.get("display_start", 0))
                        if height_um <= 1e-12 or display_px <= 0:
                            continue
                        key = self._display_material_count_profile_key(material_idx, height_um)
                        overrides[key] = max(int(display_px), int(overrides.get(key, 0)))
            return overrides
        finally:
            self.visualisation_mode = saved_mode
            self.visual_thickness_overrides = saved_overrides
            self.visual_thickness_step_overrides = saved_step_overrides
            if saved_info is not None:
                self._last_display_remap_info = saved_info

    def finalise_visualisation(self):
        if getattr(self, "xsec", None) is None:
            QMessageBox.information(
                self,
                "Nothing to adjust",
                "Create or open a process first.",
            )
            return
        overrides = self._collect_dynamic_visual_overrides()
        if not overrides:
            QMessageBox.information(
                self,
                "Nothing to adjust",
                "The process has no visible layers.",
            )
            return
        self.visual_thickness_overrides = overrides
        self.visual_thickness_step_overrides = {}
        self.visual_adjustment_notice_dismissed = False
        self._set_visualisation_mode("pending")
        self.update_plot(getattr(self, "current_title", "") or "")
        if hasattr(self, "_snapshot_active_tab_state"):
            self._snapshot_active_tab_state()
        if hasattr(self, "statusBar"):
            self.statusBar().showMessage("Display heights are ready to adjust.", 5000)

    def accept_visualisation(self):
        if self._normalised_visualisation_mode() != "pending":
            return
        self._set_visualisation_mode("finalised")
        self.visual_adjustment_notice_dismissed = False
        self.update_plot(getattr(self, "current_title", "") or "")
        if hasattr(self, "_snapshot_active_tab_state"):
            self._snapshot_active_tab_state()
        if hasattr(self, "statusBar"):
            self.statusBar().showMessage("Display adjustments saved.", 4000)

    def reject_visualisation(self):
        if self._normalised_visualisation_mode() not in {"pending", "finalised"}:
            return
        self.visual_thickness_overrides = {}
        self.visual_thickness_step_overrides = {}
        self.visual_adjustment_notice_dismissed = False
        self._set_visualisation_mode("dynamic")
        self.update_plot(getattr(self, "current_title", "") or "")
        if hasattr(self, "_snapshot_active_tab_state"):
            self._snapshot_active_tab_state()
        if hasattr(self, "statusBar"):
            self.statusBar().showMessage("Automatic display heights restored.", 4000)

    def reset_visualisation_for_process_change(self):
        if self._normalised_visualisation_mode() == "dynamic":
            return
        self.visual_thickness_overrides = {}
        self.visual_thickness_step_overrides = {}
        self.visual_adjustment_notice_dismissed = False
        self._set_visualisation_mode("dynamic")

    # -- extracted methods --
