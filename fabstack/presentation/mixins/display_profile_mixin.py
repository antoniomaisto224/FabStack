"""Display and scene-profile construction behavior."""

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
from fabstack.rendering.scene_constraints import build_scene_constraint_profile
from fabstack.rendering.segment_targets import (
    build_material_column_segments_with_targets,
)
from fabstack.resources import LOGO_PATH

try:
    from docx import Document
    from docx.shared import Inches
except ImportError:
    Document = None

LOGGER = logging.getLogger("fabstack.presentation.mixins.plotting_export_mixin")


class DisplayProfileMixin:
    def _feature_scale_factor_clamped(self):
        return max(
            FEATURE_EMPHASIS_MIN,
            min(
                FEATURE_EMPHASIS_MAX,
                float(getattr(self, "feature_scale_factor", FEATURE_EMPHASIS_DEFAULT)),
            ),
        )

    def _visual_scale_factor_clamped(self):
        return max(
            VISUAL_SCALE_MIN,
            min(VISUAL_SCALE_MAX, float(getattr(self, "visual_scale_factor", 1.0))),
        )

    def _visual_render_scale_factor(self, scale=None):
        if scale is None:
            scale = self._visual_scale_factor_clamped()
        return max(0.01, float(scale) * float(VISUAL_RENDER_SCALE_BASE))

    def _plot_y_for_visual_scale(self, y, display_height, scale=None):
        if scale is None:
            scale = self._visual_scale_factor_clamped()
        scale = self._visual_render_scale_factor(scale)
        bottom_y = float(display_height) - 0.5
        return bottom_y - (bottom_y - float(y)) * float(scale)

    def _scaled_stack_vertical_span(self, grid, scale=None):
        y0, height = self._active_stack_vertical_span(grid)
        if scale is None:
            scale = self._visual_scale_factor_clamped()
        render_scale = self._visual_render_scale_factor(scale)
        return self._plot_y_for_visual_scale(y0, int(grid.shape[0]), scale), float(height) * render_scale

    def _display_adjustment_sources(self):
        sources = []
        for idx, state in enumerate(getattr(self, "history", []) or []):
            if not isinstance(state, dict) or "grid" not in state:
                continue
            sources.append((
                idx,
                state.get("grid"),
                state.get("row_heights_um"),
                state.get("structural_role_grid"),
                state.get("structural_bodies"),
                state.get("planarizer_role_grid"),
                state.get("planarizer_roles"),
            ))

        if sources:
            return sources

        if getattr(self, "xsec", None) is not None:
            heights = (
                self.xsec.copy_row_heights()
                if hasattr(self.xsec, "copy_row_heights")
                else None
            )
            roles = (
                self.xsec.copy_structural_roles()
                if hasattr(self.xsec, "copy_structural_roles")
                else None
            )
            bodies = (
                self.xsec.copy_structural_bodies()
                if hasattr(self.xsec, "copy_structural_bodies")
                else None
            )
            planarizer_roles = (
                self.xsec.copy_planarizer_roles()
                if hasattr(self.xsec, "copy_planarizer_roles")
                else None
            )
            planarizer_metadata = (
                self.xsec.copy_planarizer_metadata()
                if hasattr(self.xsec, "copy_planarizer_metadata")
                else None
            )
            sources.append((
                self._active_history_index_for_view(),
                self.xsec.grid,
                heights,
                roles,
                bodies,
                planarizer_roles,
                planarizer_metadata,
            ))
        return sources

    def _display_row_heights_array(self, src, row_heights_um):
        if row_heights_um is None or len(row_heights_um) != src.shape[0]:
            return np.ones(src.shape[0], dtype=float)
        return np.asarray(row_heights_um, dtype=float)

    def _display_row_edges_um(self, row_heights):
        return np.concatenate(([0.0], np.cumsum(np.asarray(row_heights, dtype=float))))

    def _display_substrate_material_indices(self):
        """Return structural-substrate material ids from model state first."""
        indices = set()
        names = list(getattr(self, "material_names", []) or [])
        xsec = getattr(self, "xsec", None)
        try:
            model_index = int(getattr(xsec, "substrate_index", -1))
        except (TypeError, ValueError):
            model_index = -1
        if 0 <= model_index < len(names):
            indices.add(model_index)

        combo = getattr(self, "substrate_combo", None)
        try:
            selected = str(combo.currentText() or "").strip() if combo is not None else ""
        except Exception:
            selected = ""
        if selected:
            try:
                indices.add(int(names.index(selected)))
            except ValueError:
                pass

        indices.discard(int(getattr(self, "VOID_INDEX", 0)))
        return indices

    def _display_nominal_substrate_um(self, sources=None, substrate_indices=None):
        xsec = getattr(self, "xsec", None)
        try:
            nominal = float(getattr(xsec, "nominal_substrate_thickness_um", 0.0) or 0.0)
        except (TypeError, ValueError):
            nominal = 0.0
        if np.isfinite(nominal) and nominal > 1e-12:
            return nominal

        substrate_indices = {int(idx) for idx in (substrate_indices or [])}
        if not substrate_indices:
            return 0.0

        # Compatibility fallback for older in-memory CrossSection objects:
        # derive the largest continuous substrate run found in history.
        largest = 0.0
        for source in sources or []:
            try:
                grid = np.asarray(source[1] if len(source) >= 3 else source[0])
                heights = self._display_row_heights_array(
                    grid,
                    source[2] if len(source) >= 3 else source[1],
                )
            except Exception:
                continue
            if grid.ndim != 2 or grid.size == 0:
                continue
            for col in range(grid.shape[1]):
                row = 0
                while row < grid.shape[0]:
                    material_idx = int(grid[row, col])
                    end = row + 1
                    while end < grid.shape[0] and int(grid[end, col]) == material_idx:
                        end += 1
                    if material_idx in substrate_indices:
                        largest = max(largest, float(np.sum(heights[row:end])))
                    row = end
        return float(largest)

    def _display_structural_body_nominals(self, sources=None):
        """Collect the authoritative nominal thickness of every wafer body."""
        nominals = {}

        def collect(raw_bodies):
            if not isinstance(raw_bodies, dict):
                return
            for raw_id, metadata in raw_bodies.items():
                if not isinstance(metadata, dict):
                    continue
                try:
                    body_id = int(metadata.get("body_id", raw_id))
                    nominal_um = float(metadata.get("nominal_thickness_um", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                if body_id <= 0 or not np.isfinite(nominal_um) or nominal_um <= 1e-12:
                    continue
                nominals[body_id] = max(float(nominals.get(body_id, 0.0)), nominal_um)

        xsec = getattr(self, "xsec", None)
        if xsec is not None:
            if hasattr(xsec, "copy_structural_bodies"):
                try:
                    collect(xsec.copy_structural_bodies())
                except Exception:
                    pass
            else:
                collect(getattr(xsec, "structural_bodies", None))
        for source in sources or []:
            if len(source) >= 5:
                collect(source[4])
        return nominals

    def _display_substrate_profile(self, sources=None):
        indices = self._display_substrate_material_indices()
        body_nominals = self._display_structural_body_nominals(sources)
        nominal = float(body_nominals.get(1, 0.0) or 0.0)
        if nominal <= 1e-12:
            nominal = self._display_nominal_substrate_um(sources, indices)
            if nominal > 1e-12:
                body_nominals.setdefault(1, float(nominal))
        return {
            "substrate_material_indices": sorted(int(idx) for idx in indices),
            "nominal_substrate_um": float(max(0.0, nominal)),
            "structural_body_nominals_um": {
                int(body_id): float(value)
                for body_id, value in body_nominals.items()
            },
        }

    def _display_base_feature_rows(self, height_um: float) -> int:
        """Feature-emphasis target, preserving the 1 nm readability floor."""
        raw_rows = float(physical_span_to_raw_visual_px(float(height_um)))
        if raw_rows <= 0.0:
            return 0
        floor_rows = float(MIN_LAYER_PX)
        emphasized = floor_rows + self._feature_scale_factor_clamped() * max(
            0.0,
            raw_rows - floor_rows,
        )
        return _visual_height_to_display_rows(emphasized)

    def _display_structural_body_rows(self, height_um: float) -> int:
        """Compact linear body scale, independent of feature emphasis."""
        try:
            height_um = float(height_um)
        except (TypeError, ValueError):
            return 0
        if not np.isfinite(height_um) or height_um <= 1e-12:
            return 0
        return max(
            int(MIN_SUBSTRATE_CORE_PX),
            int(math.ceil(height_um * float(SUBSTRATE_ROWS_PER_UM) - 1e-9)),
        )

    def _display_partial_core_reserve_rows(self, nominal_um: float, depth_um: float) -> int:
        """Readable reserve that transitions to proportional protection for deep etches."""
        nominal_um = max(0.0, float(nominal_um))
        depth_um = max(0.0, float(depth_um))
        remaining_um = max(0.0, nominal_um - depth_um)
        if nominal_um <= 1e-12 or remaining_um <= 1e-12:
            return 0
        etch_fraction = max(0.0, min(1.0, depth_um / nominal_um))
        transition = max(0.0, min(1.0, (etch_fraction - 0.10) / 0.30))
        smooth_weight = transition * transition * (3.0 - 2.0 * transition)
        proportional = int(math.ceil(
            remaining_um * float(SUBSTRATE_ROWS_PER_UM) - 1e-9
        ))
        weighted = int(math.ceil(float(proportional) * smooth_weight - 1e-9))
        return max(int(MIN_SUBSTRATE_CORE_PX), weighted)

    @staticmethod
    def _display_physical_order_margin(larger_um: float, smaller_um: float) -> int:
        larger_um = max(0.0, float(larger_um))
        smaller_um = max(0.0, float(smaller_um))
        tolerance = max(1e-12, max(larger_um, smaller_um) * 1e-9)
        if larger_um <= smaller_um + tolerance:
            return 0
        if smaller_um <= tolerance:
            return int(ORDER_MARGIN_PX)
        ratio = larger_um / smaller_um
        return int(ORDER_MARGIN_PX) if ratio >= float(ORDER_TIE_RATIO) else 1

    def _display_profile_feature_rows(self, height_um: float, profile=None) -> int:
        key = self._display_feature_profile_key(height_um)
        resolved = (profile or {}).get("resolved_feature_rows", {}) or {}
        try:
            value = int(resolved.get(key, 0) or 0)
        except (TypeError, ValueError):
            value = 0
        return value if value > 0 else self._display_base_feature_rows(height_um)

    def _display_raw_material_columns(
        self,
        active,
        active_heights,
        process_profile=None,
        active_structural_roles=None,
        active_planarizer_roles=None,
    ):
        """Extract physical column runs and mark the structural substrate run."""
        active = np.asarray(active)
        active_heights = np.asarray(active_heights, dtype=float)
        if active.ndim != 2 or active.size == 0:
            return []
        if len(active_heights) != active.shape[0]:
            active_heights = np.ones(active.shape[0], dtype=float)
        roles = np.asarray(active_structural_roles) if active_structural_roles is not None else None
        if roles is None or roles.shape != active.shape:
            roles = np.zeros_like(active, dtype=np.int16)
        planarizer_roles = (
            np.asarray(active_planarizer_roles)
            if active_planarizer_roles is not None
            else None
        )
        if planarizer_roles is None or planarizer_roles.shape != active.shape:
            planarizer_roles = np.zeros_like(active, dtype=np.int32)

        substrate_indices = {
            int(idx) for idx in (process_profile or {}).get("substrate_material_indices", [])
        }
        body_nominals = {
            int(body_id): float(value)
            for body_id, value in (
                (process_profile or {}).get("structural_body_nominals_um", {}) or {}
            ).items()
        }
        columns = []
        for col in range(active.shape[1]):
            segments = []
            row = 0
            cursor_um = 0.0
            while row < active.shape[0]:
                material_idx = int(active[row, col])
                body_id = int(roles[row, col])
                planarizer_role = int(planarizer_roles[row, col])
                end = row + 1
                while (
                    end < active.shape[0]
                    and int(active[end, col]) == material_idx
                    and int(roles[end, col]) == body_id
                    and int(planarizer_roles[end, col]) == planarizer_role
                ):
                    end += 1
                height_um = float(np.sum(active_heights[row:end]))
                if height_um > 1e-12:
                    segments.append({
                        "material": material_idx,
                        "height_um": height_um,
                        "source_segment_id": f"{col}:{row}",
                        "source_height_um": height_um,
                        "target_rows": 0,
                        "y0_local_um": cursor_um,
                        "y1_local_um": cursor_um + height_um,
                        "is_substrate_core": False,
                        "is_structural_body": bool(body_id > 0),
                        "structural_body_id": int(body_id),
                        "structural_nominal_um": float(body_nominals.get(body_id, 0.0)),
                        "planarizer_role": int(planarizer_role),
                        "is_planarizer_overburden": bool(planarizer_role > 0),
                        "is_planarizer_fill": bool(planarizer_role < 0),
                    })
                    cursor_um += height_um
                row = end

            role_core_candidates = [
                idx for idx, segment in enumerate(segments)
                if int(segment.get("structural_body_id", 0)) > 0
            ]
            core_candidates = role_core_candidates or [
                idx for idx, segment in enumerate(segments)
                if int(segment.get("material", self.VOID_INDEX)) in substrate_indices
                and not int(segment.get("structural_body_id", 0))
            ]
            if core_candidates:
                # The first structural body encountered from the process side
                # is the currently exposed wafer. This naturally switches to
                # a bonded substrate, and switches back after a flip.
                core_idx = min(core_candidates)
                segments[core_idx]["is_substrate_core"] = True

                # The original substrate surface is the reference plane for
                # the etched recess. Split merged material/void runs at this
                # plane so the single displayed recess budget can be shared by
                # trench solids and the actually remaining void.
                try:
                    nominal_um = float(segments[core_idx].get("structural_nominal_um", 0.0) or 0.0)
                    if nominal_um <= 1e-12:
                        nominal_um = float(
                            (process_profile or {}).get("nominal_substrate_um", 0.0) or 0.0
                        )
                except (TypeError, ValueError):
                    nominal_um = 0.0
                core_segment = segments[core_idx]
                reference_y = float(core_segment.get("y1_local_um", 0.0)) - nominal_um
                core_top_y = float(core_segment.get("y0_local_um", 0.0))
                if nominal_um > 1e-12 and core_top_y > reference_y + 1e-12:
                    split_segments = []
                    for segment in segments:
                        y0 = float(segment.get("y0_local_um", 0.0))
                        y1 = float(segment.get("y1_local_um", 0.0))
                        cuts = [y0]
                        if y0 < reference_y - 1e-12 and y1 > reference_y + 1e-12:
                            cuts.append(reference_y)
                        cuts.append(y1)
                        for part_y0, part_y1 in zip(cuts, cuts[1:]):
                            if part_y1 <= part_y0 + 1e-12:
                                continue
                            part = dict(segment)
                            part["y0_local_um"] = float(part_y0)
                            part["y1_local_um"] = float(part_y1)
                            part["height_um"] = float(part_y1 - part_y0)
                            part["in_substrate_recess"] = bool(
                                part_y0 >= reference_y - 1e-9
                                and part_y1 <= core_top_y + 1e-9
                                and not bool(part.get("is_substrate_core", False))
                            )
                            split_segments.append(part)
                    segments = split_segments
                else:
                    for segment in segments:
                        segment["in_substrate_recess"] = False
            columns.append(segments)
        return columns

    def _display_contiguous_solid_stack_keys(self, segments):
        """Return maximal solid stacks, split at voids and the substrate core."""
        stacks = []
        current = []
        current_source_id = None
        for segment in segments or []:
            material_idx = int(segment.get("material", self.VOID_INDEX))
            separator = (
                material_idx == int(self.VOID_INDEX)
                or bool(segment.get("is_substrate_core", False))
                or bool(segment.get("is_structural_body", False))
            )
            if separator:
                if current:
                    stacks.append(tuple(current))
                    current = []
                current_source_id = None
                continue
            source_id = str(segment.get("source_segment_id", id(segment)))
            if source_id == current_source_id:
                continue
            current.append(self._display_feature_profile_key(
                segment.get("source_height_um", segment.get("height_um", 0.0))
            ))
            current_source_id = source_id
        if current:
            stacks.append(tuple(current))
        return stacks

    def _display_scene_constraint_profile(self, sources, profile):
        return build_scene_constraint_profile(self, sources, profile)

    def _display_substrate_core_target_height(
        self,
        height_um: float,
        profile=None,
        *,
        structural_body_id=None,
        structural_nominal_um=None,
        minimum_removed_rows: int = 0,
        preferred_removed_rows=None,
    ) -> int:
        """Render the remaining structural body from nominal body minus etch."""
        height_um = max(0.0, float(height_um))
        if height_um <= 1e-12:
            return 0
        try:
            body_id = int(structural_body_id or 1)
        except (TypeError, ValueError):
            body_id = 1
        try:
            nominal_um = float(structural_nominal_um or 0.0)
        except (TypeError, ValueError):
            nominal_um = 0.0
        if nominal_um <= 1e-12:
            try:
                nominal_um = float(
                    ((profile or {}).get("structural_body_nominals_um", {}) or {}).get(
                        body_id,
                        0.0,
                    ) or 0.0
                )
            except (TypeError, ValueError):
                nominal_um = 0.0
        if nominal_um <= 1e-12:
            try:
                nominal_um = float((profile or {}).get("nominal_substrate_um", 0.0) or 0.0)
            except (TypeError, ValueError):
                nominal_um = 0.0

        if nominal_um <= 1e-12:
            return self._display_profile_feature_rows(height_um, profile)

        try:
            nominal_rows = int(
                ((profile or {}).get("structural_body_rows", {}) or {}).get(body_id, 0) or 0
            )
        except (TypeError, ValueError):
            nominal_rows = 0
        if nominal_rows <= 0 and body_id == 1:
            try:
                nominal_rows = int((profile or {}).get("nominal_substrate_rows", 0) or 0)
            except (TypeError, ValueError):
                nominal_rows = 0
        if nominal_rows <= 0:
            nominal_rows = self._display_structural_body_rows(nominal_um)
        if height_um >= nominal_um - 1e-12:
            extra_um = max(0.0, height_um - nominal_um)
            extra_rows = (
                self._display_profile_feature_rows(extra_um, profile)
                if extra_um > 1e-12 else 0
            )
            return int(nominal_rows + extra_rows)

        removed_um = max(0.0, nominal_um - height_um)
        removed_rows = self._display_profile_feature_rows(removed_um, profile)
        if preferred_removed_rows is not None:
            removed_rows = max(1, int(preferred_removed_rows))
        else:
            removed_rows = max(int(removed_rows), max(0, int(minimum_removed_rows)))
        return max(
            int(MIN_SUBSTRATE_CORE_PX),
            int(nominal_rows) - min(
                int(nominal_rows) - int(MIN_SUBSTRATE_CORE_PX),
                int(removed_rows),
            ),
        )

    def _apportion_display_rows(self, total_target: int, weights):
        weights = [max(0.0, float(value)) for value in weights]
        positive = [idx for idx, value in enumerate(weights) if value > 1e-12]
        targets = [0] * len(weights)
        if not positive:
            return targets

        total_target = max(1, int(total_target))
        if total_target < len(positive):
            ranked = sorted(positive, key=lambda idx: weights[idx], reverse=True)
            for idx in ranked[:total_target]:
                targets[idx] = 1
            return targets

        total_weight = float(sum(weights[idx] for idx in positive))
        if total_weight <= 1e-12:
            for idx in positive:
                targets[idx] = 1
            return targets

        raw = {
            idx: (float(total_target) * weights[idx] / total_weight)
            for idx in positive
        }
        for idx in positive:
            targets[idx] = max(1, int(math.floor(raw[idx])))

        assigned = int(sum(targets))
        if assigned > total_target:
            ranked = sorted(positive, key=lambda idx: (raw[idx] - math.floor(raw[idx]), weights[idx]))
            for idx in ranked:
                if assigned <= total_target:
                    break
                if targets[idx] > 1:
                    targets[idx] -= 1
                    assigned -= 1
        elif assigned < total_target:
            ranked = sorted(
                positive,
                key=lambda idx: (raw[idx] - math.floor(raw[idx]), weights[idx]),
                reverse=True,
            )
            cursor = 0
            while assigned < total_target and ranked:
                idx = ranked[cursor % len(ranked)]
                targets[idx] += 1
                assigned += 1
                cursor += 1
        return targets

    def _display_material_column_segments_with_targets(
        self,
        active,
        active_heights,
        *,
        history_index=None,
        process_profile=None,
        active_structural_roles=None,
        active_planarizer_roles=None,
    ):
        return build_material_column_segments_with_targets(
            self,
            active,
            active_heights,
            history_index=history_index,
            process_profile=process_profile,
            active_structural_roles=active_structural_roles,
            active_planarizer_roles=active_planarizer_roles,
        )

    def _render_material_columns_from_targets(
        self,
        active,
        active_heights,
        final_height: int,
        *,
        history_index=None,
        process_profile=None,
        active_structural_roles=None,
        active_planarizer_roles=None,
    ):
        """
        Stack each column's material segments bottom-up into a `final_height`
        canvas, using the exact target row counts already computed per
        segment (a pure function of that segment's own physical thickness -
        see `_display_material_column_segments_with_targets`). Also returns,
        per column, the placed segments annotated with their final display
        row range - this is the single source of truth later reused by both
        the isotropic-etch-front overlay and the "click a layer" pick logic,
        so all three always agree on where a given physical layer landed.
        """
        active = np.asarray(active)
        final_height = max(1, int(final_height))
        width = int(active.shape[1]) if active.ndim == 2 and active.shape[1] > 0 else 1
        columns, base_height = self._display_material_column_segments_with_targets(
            active,
            active_heights,
            history_index=history_index,
            process_profile=process_profile,
            active_structural_roles=active_structural_roles,
            active_planarizer_roles=active_planarizer_roles,
        )
        if base_height <= 0:
            return (
                np.full((final_height, width), self.VOID_INDEX, dtype=active.dtype),
                1,
                [[] for _ in range(width)],
            )

        result = np.full((final_height, width), self.VOID_INDEX, dtype=active.dtype)
        placed_columns = []
        for col, segments in enumerate(columns):
            cursor = final_height
            placed = []
            for segment in reversed(segments):
                target = max(0, int(segment.get("target_rows", 0)))
                if target <= 0:
                    continue
                start = max(0, cursor - target)
                if start < cursor:
                    result[start:cursor, col] = int(segment.get("material", self.VOID_INDEX))
                    placed.append({
                        "material": int(segment.get("material", self.VOID_INDEX)),
                        "is_substrate_core": bool(segment.get("is_substrate_core", False)),
                        "is_structural_body": bool(segment.get("is_structural_body", False)),
                        "structural_body_id": int(segment.get("structural_body_id", 0)),
                        "planarizer_role": int(segment.get("planarizer_role", 0)),
                        "in_substrate_recess": bool(segment.get("in_substrate_recess", False)),
                        "is_planarizer_overburden": bool(
                            segment.get("is_planarizer_overburden", False)
                        ),
                        "is_planarizer_fill": bool(segment.get("is_planarizer_fill", False)),
                        "y0_local_um": float(segment.get("y0_local_um", 0.0)),
                        "y1_local_um": float(segment.get("y1_local_um", 0.0)),
                        "display_start": int(start),
                        "display_end": int(cursor),
                    })
                cursor = start
                if cursor <= 0:
                    break
            placed.reverse()
            placed_columns.append(placed)
        return result.astype(active.dtype, copy=False), int(base_height), placed_columns

    def _display_active_band_data(
        self,
        src,
        row_heights_um=None,
        *,
        structural_role_grid=None,
        planarizer_role_grid=None,
    ):
        src = np.asarray(src)
        if src.ndim != 2 or src.size == 0:
            return None

        row_heights = self._display_row_heights_array(src, row_heights_um)
        row_edges = self._display_row_edges_um(row_heights)
        non_void_rows = np.where(np.any(src != self.VOID_INDEX, axis=1))[0]
        if non_void_rows.size == 0:
            return None

        top = int(non_void_rows.min())
        bottom = int(non_void_rows.max())
        active = src[top:bottom + 1, :]
        active_heights = row_heights[top:bottom + 1]
        roles = np.asarray(structural_role_grid) if structural_role_grid is not None else None
        active_roles = (
            roles[top:bottom + 1, :]
            if roles is not None and roles.shape == src.shape
            else np.zeros_like(active, dtype=np.int16)
        )
        planarizer_roles = (
            np.asarray(planarizer_role_grid)
            if planarizer_role_grid is not None
            else None
        )
        active_planarizer_roles = (
            planarizer_roles[top:bottom + 1, :]
            if planarizer_roles is not None and planarizer_roles.shape == src.shape
            else np.zeros_like(active, dtype=np.int32)
        )

        return {
            "row_heights": row_heights,
            "row_edges": row_edges,
            "top": top,
            "bottom": bottom,
            "active": active,
            "active_heights": active_heights,
            "active_structural_roles": active_roles,
            "active_planarizer_roles": active_planarizer_roles,
        }

    def _process_display_adjustment_profile(self, *, ignore_visual_overrides: bool = False):
        """
        Compute substrate identity/nominal body and the shared canvas height.

        Substrate-core rows need the nominal body thickness so an etched core
        is rendered as ``nominal body - removed depth``.  The final target
        height is shared across history so scrubbing does not resize the view.
        """
        sources = self._display_adjustment_sources()
        profile = self._display_substrate_profile(sources)
        if not sources:
            profile["target_height"] = 0
            return profile

        profile.update(self._display_scene_constraint_profile(sources, profile))
        if (
            not bool(profile.get("constraint_converged", False))
            or list(profile.get("constraint_violations", []) or [])
        ):
            raise RuntimeError(
                "Adaptive display constraints did not converge to a physically ordered scene"
            )

        saved_overrides = None
        saved_step_overrides = None
        if ignore_visual_overrides:
            saved_overrides = getattr(self, "visual_thickness_overrides", {})
            saved_step_overrides = getattr(self, "visual_thickness_step_overrides", {})
            self.visual_thickness_overrides = {}
            self.visual_thickness_step_overrides = {}
        try:
            target_height = 0
            for source in sources:
                if len(source) >= 3:
                    source_index, grid, row_heights = source[:3]
                    structural_roles = source[3] if len(source) >= 4 else None
                    planarizer_roles = source[5] if len(source) >= 6 else None
                else:
                    source_index, grid, row_heights = None, source[0], source[1]
                    structural_roles = None
                    planarizer_roles = None
                try:
                    src = np.asarray(grid)
                    band_data = self._display_active_band_data(
                        src,
                        row_heights,
                        structural_role_grid=structural_roles,
                        planarizer_role_grid=planarizer_roles,
                    )
                    if band_data is None:
                        continue
                    _columns, column_base_height = self._display_material_column_segments_with_targets(
                        band_data["active"],
                        band_data["active_heights"],
                        history_index=source_index,
                        process_profile=profile,
                        active_structural_roles=band_data.get("active_structural_roles"),
                        active_planarizer_roles=band_data.get("active_planarizer_roles"),
                    )
                    state_height = max(0, int(column_base_height or 0))
                    if state_height > 0:
                        target_height = max(target_height, state_height + int(DISPLAY_TOP_HEADROOM_PX))
                except Exception:
                    continue
            profile["target_height"] = int(target_height)
            return profile
        finally:
            if ignore_visual_overrides:
                self.visual_thickness_overrides = saved_overrides
                self.visual_thickness_step_overrides = saved_step_overrides

    def _display_feature_profile_key(self, height_um: float) -> str:
        try:
            height = max(0.0, float(height_um))
        except (TypeError, ValueError):
            height = 0.0
        return f"{height:.9f}"

    def _display_material_count_profile_key(self, material_idx, thickness_um: float) -> str:
        return f"{int(material_idx)}:{self._display_feature_profile_key(thickness_um)}"

    def _current_material_column_total_um(self, material_idx, col: int = 0) -> float:
        """
        Sum of this material's (or VOID_INDEX's) row heights in one column of
        the current xsec grid, within the same active band used at render
        time - the exact per-column total
        `_display_material_column_segments_with_targets` keys its override
        lookup on. Used to set a VT override at operation-creation time so it
        matches what the renderer will actually look up, without waiting for
        a canvas click.
        """
        if getattr(self, "xsec", None) is None:
            return 0.0
        grid = np.asarray(self.xsec.grid)
        if grid.ndim != 2 or grid.size == 0 or not (0 <= col < grid.shape[1]):
            return 0.0
        row_heights = (
            self.xsec.copy_row_heights() if hasattr(self.xsec, "copy_row_heights") else None
        )
        band_data = self._display_active_band_data(grid, row_heights)
        if band_data is None:
            return 0.0
        active = band_data["active"]
        active_heights = band_data["active_heights"]
        return float(np.sum(active_heights[active[:, col] == int(material_idx)]))

    # -- extracted methods --
