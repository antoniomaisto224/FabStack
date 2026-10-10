"""Canvas construction, interaction, and plotting behavior."""

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


class CanvasRenderingMixin:
    def _include_export_logo(self) -> bool:
        return bool(getattr(self, "include_logo_in_exports", True)) and os.path.exists(LOGO_PATH)

    def _add_export_logo_to_axis(self, ax):
        if ax is None or not self._include_export_logo():
            return None
        try:
            logo = plt.imread(LOGO_PATH)
            width = int(getattr(logo, "shape", [0, 0])[1])
            if width <= 0:
                return None
            target_width_px = 145
            image_box = OffsetImage(logo, zoom=target_width_px / float(width))
            logo_artist = AnnotationBbox(
                image_box,
                (0.985, 0.965),
                xycoords=ax.transAxes,
                box_alignment=(1.0, 1.0),
                frameon=False,
                pad=0.0,
                zorder=20,
            )
            ax.add_artist(logo_artist)
            return logo_artist
        except Exception:
            return None

    def _scroll_figure_view_to_bottom(self):
        if not hasattr(self, "figure_scroll") or self.figure_scroll is None:
            return
        bar = self.figure_scroll.verticalScrollBar()
        if bar is not None:
            bar.setValue(bar.maximum())

    def _preview_main_subplots_adjust(self):
        """
        Keep preview margins visually stable when global vertical scale changes.
        Without this, fixed fractional margins become large in pixels and make
        the stack appear to move upward as the figure gets taller.
        """
        cfg = dict(EXPORT_PREVIEW_SUBPLOTS)
        # Only compensate margins for scales > 1.0. For scales < 1.0 keep
        # baseline margins to avoid downward/upward drift.
        scale = self._visual_render_scale_factor()
        if scale <= 1.0:
            return cfg

        base_bottom = float(cfg.get("bottom", 0.05))
        base_top_gap = 1.0 - float(cfg.get("top", 0.95))
        cfg["bottom"] = max(0.003, base_bottom / scale)
        cfg["top"] = min(0.997, 1.0 - (base_top_gap / scale))

        if cfg["top"] <= cfg["bottom"]:
            cfg["bottom"] = 0.02
            cfg["top"] = 0.98
        return cfg

    def _row_heights_for_plot_grid(self, grid, history_index=-1):
        if 0 <= int(history_index) < len(getattr(self, "history", [])):
            heights = self.history[int(history_index)].get("row_heights_um")
            if heights is not None and len(heights) == np.asarray(grid).shape[0]:
                return np.asarray(heights, dtype=float)
        if getattr(self, "xsec", None) is not None and hasattr(self.xsec, "copy_row_heights"):
            heights = self.xsec.copy_row_heights()
            if len(heights) == np.asarray(grid).shape[0]:
                return heights
        return None

    def _structural_roles_for_plot_grid(self, grid, history_index=-1):
        try:
            idx = int(history_index)
        except (TypeError, ValueError):
            idx = -1
        if 0 <= idx < len(getattr(self, "history", [])):
            roles = self.history[idx].get("structural_role_grid")
            if roles is not None and np.asarray(roles).shape == np.asarray(grid).shape:
                return np.asarray(roles, dtype=np.int16)
        if getattr(self, "xsec", None) is not None and hasattr(self.xsec, "copy_structural_roles"):
            roles = self.xsec.copy_structural_roles()
            if np.asarray(roles).shape == np.asarray(grid).shape:
                return np.asarray(roles, dtype=np.int16)
        return None

    def _planarizer_roles_for_plot_grid(self, grid, history_index=-1):
        try:
            idx = int(history_index)
        except (TypeError, ValueError):
            idx = -1
        if 0 <= idx < len(getattr(self, "history", [])):
            roles = self.history[idx].get("planarizer_role_grid")
            if roles is not None and np.asarray(roles).shape == np.asarray(grid).shape:
                return np.asarray(roles, dtype=np.int32)
        if getattr(self, "xsec", None) is not None and hasattr(
            self.xsec,
            "copy_planarizer_roles",
        ):
            roles = self.xsec.copy_planarizer_roles()
            if np.asarray(roles).shape == np.asarray(grid).shape:
                return np.asarray(roles, dtype=np.int32)
        return None

    def _display_contiguous_column_segments(self, cols):
        cols = sorted({int(col) for col in cols})
        if not cols:
            return []
        segments = []
        start = prev = cols[0]
        for col in cols[1:]:
            if col == prev + 1:
                prev = col
                continue
            segments.append((start, prev))
            start = prev = col
        segments.append((start, prev))
        return segments

    def _open_etch_void_intervals(self, src, row_heights, feature):
        src = np.asarray(src)
        if src.ndim != 2 or src.size == 0:
            return []
        try:
            x_start = int(feature.get("x_start"))
            x_end = int(feature.get("x_end"))
            surface_y = float(feature.get("surface_y_um"))
            bottom_y = float(feature.get("bottom_y_um"))
        except (TypeError, ValueError, AttributeError):
            return []
        if x_end < x_start or bottom_y <= surface_y + 1e-12:
            return []

        x_start = max(0, min(src.shape[1] - 1, x_start))
        x_end = max(0, min(src.shape[1] - 1, x_end))
        if x_end < x_start:
            return []

        heights = self._display_row_heights_array(src, row_heights)
        row_edges = self._display_row_edges_um(heights)
        overlap_rows = np.where(
            (row_edges[:-1] < bottom_y - 1e-12)
            & (row_edges[1:] > surface_y + 1e-12)
        )[0]
        if overlap_rows.size == 0:
            return []

        grouped = {}
        for col in range(x_start, x_end + 1):
            intervals = []
            for row in overlap_rows:
                if int(src[int(row), int(col)]) != self.VOID_INDEX:
                    continue
                y0 = max(surface_y, float(row_edges[int(row)]))
                y1 = min(bottom_y, float(row_edges[int(row) + 1]))
                if y1 <= y0 + 1e-12:
                    continue
                if intervals and y0 <= intervals[-1][1] + 1e-12:
                    intervals[-1] = (intervals[-1][0], max(intervals[-1][1], y1))
                else:
                    intervals.append((y0, y1))

            for y0, y1 in intervals:
                key = (round(float(y0), 12), round(float(y1), 12))
                grouped.setdefault(key, []).append(int(col))

        result = []
        for (y0, y1), cols_for_interval in grouped.items():
            for seg_start, seg_end in self._display_contiguous_column_segments(cols_for_interval):
                result.append((int(seg_start), int(seg_end), float(y0), float(y1)))
        return result

    def _etch_fill_intervals(self, src, row_heights, feature):
        src = np.asarray(src)
        if src.ndim != 2 or src.size == 0:
            return []
        try:
            x_start = int(feature.get("x_start"))
            x_end = int(feature.get("x_end"))
            surface_y = float(feature.get("surface_y_um"))
            bottom_y = float(feature.get("bottom_y_um"))
        except (TypeError, ValueError, AttributeError):
            return []
        if x_end < x_start or bottom_y <= surface_y + 1e-12:
            return []

        x_start = max(0, min(src.shape[1] - 1, x_start))
        x_end = max(0, min(src.shape[1] - 1, x_end))
        if x_end < x_start:
            return []

        heights = self._display_row_heights_array(src, row_heights)
        row_edges = self._display_row_edges_um(heights)
        overlap_rows = np.where(
            (row_edges[:-1] < bottom_y - 1e-12)
            & (row_edges[1:] > surface_y + 1e-12)
        )[0]
        if overlap_rows.size == 0:
            return []

        grouped = {}
        for col in range(x_start, x_end + 1):
            current = None
            intervals = []
            for row in overlap_rows:
                material_idx = int(src[int(row), int(col)])
                if material_idx == self.VOID_INDEX:
                    if current is not None:
                        intervals.append(current)
                        current = None
                    continue
                y0 = max(surface_y, float(row_edges[int(row)]))
                y1 = min(bottom_y, float(row_edges[int(row) + 1]))
                if y1 <= y0 + 1e-12:
                    continue
                if (
                    current is not None
                    and current[0] == material_idx
                    and y0 <= current[2] + 1e-12
                ):
                    current = (current[0], current[1], max(current[2], y1))
                else:
                    if current is not None:
                        intervals.append(current)
                    current = (material_idx, y0, y1)
            if current is not None:
                intervals.append(current)

            for material_idx, y0, y1 in intervals:
                key = (int(material_idx), round(float(y0), 12), round(float(y1), 12))
                grouped.setdefault(key, []).append(int(col))

        result = []
        for (material_idx, y0, y1), cols_for_interval in grouped.items():
            for seg_start, seg_end in self._display_contiguous_column_segments(cols_for_interval):
                result.append((int(seg_start), int(seg_end), int(material_idx), float(y0), float(y1)))
        return result

    def _display_grid_row_bands(
        self,
        grid,
        row_heights_um=None,
        target_height: int = 0,
        history_index=None,
        process_profile=None,
        visual_geometry=None,
        structural_role_grid=None,
        planarizer_role_grid=None,
    ):
        """
        Render the physical grid to a display-pixel array.

        Every column stacks its own material segments via
        `_render_material_columns_from_targets`, whose per-segment target
        row counts are a pure function of that segment's own physical
        thickness (see `_display_material_column_segments_with_targets`) -
        so identical physical situations always render identically, and
        only genuinely different topography (e.g. inside vs. outside an
        etched gap) produces a different pixel count. The placed-segment
        metadata this stashes on `_last_display_remap_info` is the single
        source of truth later reused by both the isotropic-etch-front
        overlay and the "click a layer" pick logic, so neither can disagree
        with what's actually on screen.
        """
        src = np.asarray(grid)
        self._last_display_remap_info = {
            "display_height": 0,
            "column_segments": [],
        }
        if src.ndim != 2 or src.size == 0:
            return src.copy()

        band_data = self._display_active_band_data(
            src,
            row_heights_um,
            structural_role_grid=structural_role_grid,
            planarizer_role_grid=planarizer_role_grid,
        )
        if band_data is None:
            return src.copy()

        top = int(band_data["top"])
        active = band_data["active"]
        active_heights = band_data["active_heights"]
        active_structural_roles = band_data.get("active_structural_roles")
        row_edges = self._display_row_edges_um(self._display_row_heights_array(src, row_heights_um))

        _columns, column_base_height = self._display_material_column_segments_with_targets(
            active,
            active_heights,
            history_index=history_index,
            process_profile=process_profile,
            active_structural_roles=active_structural_roles,
            active_planarizer_roles=band_data.get("active_planarizer_roles"),
        )
        column_base_height = max(1, int(column_base_height or 0))

        minimum_total = column_base_height + int(DISPLAY_TOP_HEADROOM_PX)
        final_height = max(minimum_total, int(target_height or 0))
        result, _rendered_column_height, placed_columns = self._render_material_columns_from_targets(
            active,
            active_heights,
            int(final_height),
            history_index=history_index,
            process_profile=process_profile,
            active_structural_roles=active_structural_roles,
            active_planarizer_roles=band_data.get("active_planarizer_roles"),
        )

        active_top_um = float(row_edges[top])
        column_segments = []
        for col_segments in placed_columns:
            entries = []
            for segment in col_segments:
                material_idx = int(segment.get("material", self.VOID_INDEX))
                y0_um = active_top_um + float(segment.get("y0_local_um", 0.0))
                y1_um = active_top_um + float(segment.get("y1_local_um", 0.0))
                entries.append({
                    "material": material_idx,
                    "is_substrate_core": bool(segment.get("is_substrate_core", False)),
                    "is_structural_body": bool(segment.get("is_structural_body", False)),
                    "structural_body_id": int(segment.get("structural_body_id", 0)),
                    "planarizer_role": int(segment.get("planarizer_role", 0)),
                    "in_substrate_recess": bool(segment.get("in_substrate_recess", False)),
                    "is_planarizer_overburden": bool(
                        segment.get("is_planarizer_overburden", False)
                    ),
                    "is_planarizer_fill": bool(segment.get("is_planarizer_fill", False)),
                    "y0_um": y0_um,
                    "y1_um": y1_um,
                    "display_start": int(segment.get("display_start", 0)),
                    "display_end": int(segment.get("display_end", 0)),
                })
            column_segments.append(entries)

        self._last_display_remap_info = {
            "display_height": int(final_height),
            "column_segments": column_segments,
        }
        return result

    def _display_grid_auto(
        self,
        grid,
        row_heights_um=None,
        *,
        apply_process_profile=True,
        process_profile=None,
        history_index=None,
        visual_geometry=None,
        structural_role_grid=None,
        planarizer_role_grid=None,
    ):
        """
        Return an adaptive display-only copy of the physical process grid.

        The process model keeps exact physical row heights. Rendering maps those
        physical heights to readable display rows so 1 nm films and 525 µm
        substrates can be inspected in the same figure.
        """
        if apply_process_profile:
            profile = process_profile or self._process_display_adjustment_profile()
        else:
            profile = dict(process_profile or self._display_substrate_profile())
            profile["target_height"] = 0

        if structural_role_grid is None:
            structural_role_grid = self._structural_roles_for_plot_grid(grid, history_index)
        if planarizer_role_grid is None:
            planarizer_role_grid = self._planarizer_roles_for_plot_grid(grid, history_index)

        return self._display_grid_row_bands(
            grid,
            row_heights_um=row_heights_um,
            target_height=int((profile or {}).get("target_height", 0) or 0),
            history_index=history_index,
            process_profile=profile,
            visual_geometry=visual_geometry,
            structural_role_grid=structural_role_grid,
            planarizer_role_grid=planarizer_role_grid,
        )

    def _visual_geometry_for_plot_grid(self, grid, history_index=-1):
        try:
            idx = int(history_index)
        except (TypeError, ValueError):
            idx = -1
        history = getattr(self, "history", []) or []
        if 0 <= idx < len(history):
            geometry = history[idx].get("visual_geometry", [])
            return list(geometry) if isinstance(geometry, list) else []
        if grid is not None and history:
            try:
                src = np.asarray(grid)
            except Exception:
                src = None
            if src is not None and src.ndim == 2:
                for state in history:
                    if not isinstance(state, dict) or "grid" not in state:
                        continue
                    candidate = state.get("grid")
                    if candidate is grid:
                        geometry = state.get("visual_geometry", [])
                        return list(geometry) if isinstance(geometry, list) else []
                    try:
                        candidate_arr = np.asarray(candidate)
                    except Exception:
                        continue
                    if candidate_arr.shape == src.shape and np.array_equal(candidate_arr, src):
                        geometry = state.get("visual_geometry", [])
                        return list(geometry) if isinstance(geometry, list) else []
        if getattr(self, "xsec", None) is not None and hasattr(self.xsec, "copy_visual_geometry"):
            return self.xsec.copy_visual_geometry()
        return []

    def _display_y_for_physical_y_in_column(self, col: int, y_um: float):
        """
        Map a physical y (um) to a display row, using the exact segment
        placement that column actually rendered with (`column_segments`,
        built alongside the fill in `_display_grid_row_bands`). This is the
        same lookup used by the click-to-pick logic, so overlays drawn
        through it can't drift from the fill underneath them the way the
        old band-derived interpolation could.
        """
        info = getattr(self, "_last_display_remap_info", {}) or {}
        columns = info.get("column_segments") or []
        if not columns:
            return None
        col = max(0, min(len(columns) - 1, int(col)))
        segments = columns[col]
        if not segments:
            return None

        y_um = float(y_um)
        for segment in segments:
            y0 = float(segment["y0_um"])
            y1 = float(segment["y1_um"])
            if y1 <= y0:
                continue
            if y0 - 1e-9 <= y_um <= y1 + 1e-9:
                frac = max(0.0, min(1.0, (y_um - y0) / (y1 - y0)))
                start = float(segment["display_start"])
                end = float(segment["display_end"])
                return start + frac * (end - start)

        if y_um < float(segments[0]["y0_um"]):
            return float(segments[0]["display_start"])
        return float(segments[-1]["display_end"])

    def _draw_visual_geometry_overlays(self, ax, geometry, display_height: int, scale=None):
        if ax is None or display_height <= 0 or not geometry:
            return
        if scale is None:
            scale = self._visual_scale_factor_clamped()

        for item in geometry:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "isotropic_etch_front":
                continue
            polylines = item.get("polylines", [])
            if not isinstance(polylines, list):
                continue
            for polyline in polylines:
                xs = []
                ys = []
                for point in polyline or []:
                    try:
                        x_value, y_value = point
                    except (TypeError, ValueError):
                        continue
                    col = int(round(float(x_value)))
                    display_edge = self._display_y_for_physical_y_in_column(col, float(y_value))
                    if display_edge is None:
                        continue
                    xs.append(float(x_value))
                    ys.append(self._plot_y_for_visual_scale(display_edge, display_height, scale))
                if len(xs) >= 2:
                    ax.plot(
                        xs,
                        ys,
                        color="#111827",
                        linewidth=0.8,
                        alpha=0.68,
                        solid_capstyle="round",
                        zorder=6,
                    )

    # ---------- Drag texts ----------
    def on_canvas_mouse_press(self, event):
        if event.button != 1:
            return
        self.dragging_litho_window = False
        self.canvas.draw()
        renderer = self.canvas.get_renderer()

        if (
            self.text_artist is not None
            and self.text_artist.get_window_extent(renderer=renderer).contains(event.x, event.y)
        ):
            self.dragging_title = True
            return

        if (
            self.substrate_text_artist is not None
            and self.substrate_text_artist.get_window_extent(renderer=renderer).contains(event.x, event.y)
        ):
            self.dragging_substrate = True
            return

        if (
            getattr(self, "material_legend_artist", None) is not None
            and self.material_legend_artist.get_window_extent(renderer=renderer).contains(event.x, event.y)
        ):
            inv = self.canvas.ax.transAxes.inverted()
            x_axes, y_axes = inv.transform((event.x, event.y))
            legend_x, legend_y = getattr(self, "material_legend_pos", (0.8, 0.0))
            self.material_legend_drag_offset = (x_axes - legend_x, y_axes - legend_y)
            self.dragging_material_legend = True
            return

        for idx, artist in self.layer_text_artists.items():
            if artist is not None and artist.get_window_extent(renderer=renderer).contains(event.x, event.y):
                self.dragging_layer_idx = idx
                return

        if bool(getattr(self, "visual_thickness_pick_mode", False)):
            self._handle_visual_thickness_canvas_pick(event)
            return

        # Lithography convenience: drag directly on canvas to define windows.
        if self._is_lithography_selected():
            frac = self._event_to_window_fraction(event)
            if frac is None:
                return
            self.dragging_litho_window = True
            self.litho_drag_start_frac = frac
            self.litho_drag_current_frac = frac
            self._clear_litho_drag_preview()
            if self.xsec is not None:
                draw_grid = self._display_grid_auto(
                    self.xsec.grid,
                    row_heights_um=self._row_heights_for_plot_grid(self.xsec.grid),
                    history_index=self._active_history_index_for_view(),
                )
                y0, hh = self._scaled_stack_vertical_span(draw_grid)
                self.litho_drag_patch = mpatches.Rectangle(
                    (frac * self.xsec.width - 0.5, y0),
                    1e-6,
                    hh,
                    facecolor="#6f95bf",
                    edgecolor="#355e8a",
                    linewidth=1.1,
                    linestyle="--",
                    alpha=0.22,
                    zorder=8,
                )
                self.litho_drag_patch.set_gid("litho_drag_preview")
                self.canvas.ax.add_patch(self.litho_drag_patch)
                self.canvas.draw_idle()
            return

    def on_canvas_mouse_move(self, event):
        if self.dragging_litho_window:
            frac = self._event_to_window_fraction(event)
            if frac is None:
                return
            self.litho_drag_current_frac = frac
            if self.litho_drag_patch is not None and self.xsec is not None:
                x0f = min(self.litho_drag_start_frac, self.litho_drag_current_frac)
                x1f = max(self.litho_drag_start_frac, self.litho_drag_current_frac)
                self.litho_drag_patch.set_x(x0f * self.xsec.width - 0.5)
                self.litho_drag_patch.set_width(max((x1f - x0f) * self.xsec.width, 1e-6))
                self.canvas.draw_idle()
            return

        if event.x is None or event.y is None:
            return
        if self.dragging_title and self.text_artist is not None:
            inv = self.canvas.ax.transAxes.inverted()
            self.text_pos = inv.transform((event.x, event.y))
            self.text_artist.set_position(self.text_pos)
            self.canvas.draw_idle()
        elif getattr(self, "dragging_material_legend", False) and getattr(self, "material_legend_artist", None) is not None:
            inv = self.canvas.ax.transAxes.inverted()
            x_axes, y_axes = inv.transform((event.x, event.y))
            offset_x, offset_y = getattr(self, "material_legend_drag_offset", (0.0, 0.0))
            self.material_legend_pos = (x_axes - offset_x, y_axes - offset_y)
            self.material_legend_artist.set_bbox_to_anchor(
                self.material_legend_pos,

                transform=self.canvas.ax.transAxes,
            )
            self.canvas.draw_idle()
        elif self.dragging_substrate and self.substrate_text_artist is not None:
            inv = self.canvas.ax.transData.inverted()
            self.substrate_text_pos = tuple(inv.transform((event.x, event.y)))
            self.substrate_text_artist.set_position(self.substrate_text_pos)
            self.canvas.draw_idle()
        elif self.dragging_layer_idx is not None:
            inv = self.canvas.ax.transData.inverted()
            x_data, y_data = inv.transform((event.x, event.y))
            self.layer_text_positions[self.dragging_layer_idx] = (x_data, y_data)
            artist = self.layer_text_artists.get(self.dragging_layer_idx)
            if artist is not None:
                artist.set_position((x_data, y_data))
                self.canvas.draw_idle()

    def on_canvas_mouse_release(self, event):
        if self.dragging_litho_window:
            end_frac = self._event_to_window_fraction(event)
            if end_frac is not None:
                self.litho_drag_current_frac = end_frac
            s = self.litho_drag_start_frac
            e = self.litho_drag_current_frac
            self.dragging_litho_window = False
            self.litho_drag_start_frac = None
            self.litho_drag_current_frac = None
            self._clear_litho_drag_preview()

            if s is not None and e is not None and self.xsec is not None:
                min_span = 1.0 / float(max(1, self.xsec.width))
                if abs(e - s) >= min_span:
                    self.add_litho_window_range(min(s, e), max(s, e), show_warning=False, refresh_plot=True)

        self.dragging_title = False
        self.dragging_substrate = False
        self.dragging_material_legend = False
        self.material_legend_drag_offset = (0.0, 0.0)
        self.dragging_layer_idx = None

    # ---------- Plotting ----------
    def plot_grid(
        self,
        ax,
        grid,
        title=None,
        main_axis=False,
        history_index=-1,
        include_export_logo=False,
        row_heights_um=None,
        visual_geometry=None,
        process_profile=None,
    ):
        """
        Draw the current grid on the given axes.

        Parameters
        ----------
        ax : matplotlib axes
        grid : 2D numpy array
        title : str or None
        main_axis : bool
            If True, use draggable step title; otherwise fixed title.
        """
        if row_heights_um is None:
            row_heights_um = self._row_heights_for_plot_grid(grid, history_index)
        if visual_geometry is None:
            visual_geometry = self._visual_geometry_for_plot_grid(grid, history_index)

        draw_grid = self._display_grid_auto(
            grid,
            row_heights_um=row_heights_um,
            history_index=history_index,
            visual_geometry=visual_geometry,
            process_profile=process_profile,
        )
        if main_axis:
            self._last_main_display_grid = np.asarray(draw_grid).copy()
            self._last_main_display_history_index = history_index
            self._last_main_display_remap_info = dict(getattr(self, "_last_display_remap_info", {}) or {})
            self._last_main_source_grid = np.asarray(grid).copy()
            self._last_main_source_row_heights = self._display_row_heights_array(
                np.asarray(grid),
                row_heights_um,
            )
        h, w = draw_grid.shape
        scale = self._visual_scale_factor_clamped()
        render_scale = self._visual_render_scale_factor(scale)
        image_bottom = float(h) - 0.5
        image_top = self._plot_y_for_visual_scale(-0.5, h, scale)
        cmap = mcolors.ListedColormap(self.material_colors)
        ax.imshow(
            draw_grid,
            origin="upper",
            cmap=cmap,
            interpolation="none",
            aspect="auto",
            vmin=0,
            vmax=len(self.material_colors) - 1,
            extent=(-0.5, float(w) - 0.5, image_bottom, image_top),
        )
        self._draw_visual_geometry_overlays(
            ax,
            visual_geometry,
            int(h),
            scale,
        )
        ax.axis("off")

        if main_axis:
            feature_factor = self._feature_scale_factor_clamped()
            feature_name = (
                self._feature_emphasis_name(feature_factor)
                if hasattr(self, "_feature_emphasis_name")
                else "Balanced" if abs(feature_factor - 0.5) <= 1e-9
                else "Compact" if abs(feature_factor - 0.25) <= 1e-9
                else "Expanded" if abs(feature_factor - 1.0) <= 1e-9
                else "Custom"
            )
            ax.text(
                0.012,
                0.985,
                f"Display detail: {feature_name} · physical dimensions unchanged",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.5,
                color="#64748b",
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "facecolor": "#f8fafc",
                    "edgecolor": "#cbd5e1",
                    "alpha": 0.9,
                    "linewidth": 0.7,
                },
                zorder=12,
            )

        # Lithography visual aid: show only explicit window list (no stale fallback).
        if main_axis and self._is_lithography_selected():
            y0, hh = self._scaled_stack_vertical_span(draw_grid, scale)
            preview_windows = self.effective_lithography_windows()
            for s, e in preview_windows:
                s = max(0.0, min(1.0, float(s)))
                e = max(0.0, min(1.0, float(e)))
                if e <= s:
                    continue
                x0 = s * w - 0.5
                ww = (e - s) * w
                rect = mpatches.Rectangle(
                    (x0, y0),
                    ww,
                    hh,
                    facecolor="#8eacc8",
                    edgecolor="#3d607f",
                    linewidth=1.0,
                    linestyle="--",
                    alpha=0.16,
                    zorder=4,
                )
                rect.set_gid("litho_preview")
                ax.add_patch(rect)

        # Title
        if title:
            title_kwargs = dict(
                fontsize=self.text_fontsize,
                va="top",
                ha="center",
                color=self._normalize_color_hex(getattr(self, "title_color", "#000000")),
                fontfamily=self._matplotlib_font_family(
                    getattr(self, "title_font_family", self.font().family() or "Sans Serif")
                ),
            )
            if main_axis:
                x, y = self.text_pos
                self.text_artist = ax.text(
                    x, y, title, transform=ax.transAxes,
                    **title_kwargs,
                )
            else:
                ax.text(
                    0.5, -0.08, title, transform=ax.transAxes,
                    **title_kwargs,
                )

        # Legend for used materials
        used_indices = sorted(
            int(i) for i in np.unique(draw_grid)
            if int(i) < len(self.material_names)
        )
        handles = []
        for idx in used_indices:
            name = self.material_names[idx]
            if name == "Void":
                continue

            lower_name = name.lower()
            if "baked" in lower_name or "anneal" in lower_name:
                disp_name = name
            else:
                disp_name = legend_label(name)

            color = (
                self.material_colors[idx]
                if idx < len(self.material_colors)
                else "#cccccc"
            )
            patch = mpatches.Patch(
                color=color,
                label=chem_subs_unicode(disp_name)
            )
            handles.append(patch)

        if handles:
            legend_pos = (
                getattr(self, "material_legend_pos", (0.8, 0.0))
                if main_axis
                else (0.8, 0.0)
            )
            legend_obj = ax.legend(
                handles=handles,
                loc="lower left",
                bbox_to_anchor=legend_pos,
                borderaxespad=0.5,
                fontsize=self.material_fontsize,
            )
            if legend_obj is not None:
                if main_axis:
                    self.material_legend_artist = legend_obj
                mat_color = self._normalize_color_hex(getattr(self, "material_color", "#000000"))
                mat_family = self._matplotlib_font_family(
                    getattr(self, "material_font_family", self.font().family() or "Sans Serif")
                )
                for txt in legend_obj.get_texts():
                    txt.set_color(mat_color)
                    txt.set_fontfamily(mat_family)

        # Extra x-room for legend
        ax.set_xlim(-0.5, w + 140)
        axis_top = image_top if render_scale > 1.0 else -0.5
        ax.set_ylim(image_bottom, axis_top)
        if include_export_logo:
            self._add_export_logo_to_axis(ax)

    def update_plot(self, title: str, *, auto_scroll_to_bottom: bool = True):
        self.current_title = title
        self.canvas.ax.clear()
        self.visual_thickness_highlight_artists = []
        self.litho_drag_patch = None
        self.text_artist = None
        self.substrate_text_artist = None
        self.material_legend_artist = None
        visual_adjustment_notice_active = False

        if self.xsec is not None:
            # Main canvas should not show step title text.
            hist_idx = self._active_history_index_for_view()
            process_profile = self._process_display_adjustment_profile()
            plot_grid = self.xsec.grid
            plot_row_heights = self._row_heights_for_plot_grid(plot_grid, hist_idx)
            plot_geometry = None
            history = getattr(self, "history", []) or []
            if 0 <= int(hist_idx) < len(history):
                state = history[int(hist_idx)]
                if isinstance(state, dict) and state.get("grid") is not None:
                    plot_grid = state.get("grid")
                    plot_row_heights = state.get("row_heights_um")
                    plot_geometry = state.get("visual_geometry", [])
            self.plot_grid(
                self.canvas.ax,
                plot_grid,
                None,
                main_axis=True,
                history_index=hist_idx,
                row_heights_um=plot_row_heights,
                visual_geometry=plot_geometry,
                process_profile=process_profile,
            )
            visual_adjustment_notice_active = bool(
                self._visual_overrides_enabled()
                and (
                    self._visual_thickness_overrides()
                    or self._visual_thickness_step_overrides()
                )
            )

            if not PREVIEW_EXPORT_AS_MAIN_VIEW:
                display_grid = self._display_grid_auto(
                    plot_grid,
                    row_heights_um=plot_row_heights,
                    history_index=hist_idx,
                    visual_geometry=plot_geometry,
                    process_profile=process_profile,
                )
                h, w = display_grid.shape
                pad_x = 260
                pad_y = 200

                need_w = w + pad_x
                # Grow canvas only above 1.0; below 1.0 the stack is scaled
                # inside the normal canvas by plot coordinates.
                scale = self._visual_render_scale_factor()
                need_h = int((h + pad_y) * max(1.0, scale))
                self.canvas.setMinimumSize(need_w, need_h)

        if PREVIEW_EXPORT_AS_MAIN_VIEW:
            self.canvas.fig.subplots_adjust(**self._preview_main_subplots_adjust())
            self._apply_export_preview_canvas_aspect()
        else:
            self.canvas.fig.subplots_adjust(bottom=0.18)

        if hasattr(self, "_set_visual_adjustment_notice_visible"):
            self._set_visual_adjustment_notice_visible(visual_adjustment_notice_active)

        self.canvas.draw_idle()

        if hasattr(self, "_sync_canvas_overlay_positions"):
            self._sync_canvas_overlay_positions()
        elif hasattr(self, "overlay_logo_label"):
            self.overlay_logo_label.raise_()
            self._position_fixed_overlay_logo()

        if auto_scroll_to_bottom:
            # Apply now + deferred passes to keep the stack anchored at bottom
            # after canvas size changes (e.g. global visual scale edits).
            self._scroll_figure_view_to_bottom()
            QTimer.singleShot(0, self._scroll_figure_view_to_bottom)
            QTimer.singleShot(30, self._scroll_figure_view_to_bottom)

    # -- extracted methods --
