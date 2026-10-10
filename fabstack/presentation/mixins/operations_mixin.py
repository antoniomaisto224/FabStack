"""Operation workflow and operation-panel UI behavior for MainWindow."""

import copy
import logging
import os

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from fabstack.materials.chemistry import chem_normalize_ascii, chem_subs_unicode
from fabstack.operations.execution import (
    execute_ui_bake_operation,
    execute_ui_external_bond_operation,
    execute_ui_material_operation,
    execute_ui_simple_bond_operation,
)
from fabstack.persistence.process_file import load_process_file
from fabstack.terminology import operation_label


LOGGER = logging.getLogger(__name__)


class OperationsMixin:
    # ---------- Windows ----------
    def open_windowing_dialog(self):
        if getattr(self, "window_dialog", None) is not None:
            self.window_dialog.show()
            self.window_dialog.raise_()
            self.window_dialog.activateWindow()
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Lithography windows")
        dlg.setModal(False)
        self.window_dialog = dlg

        layout = QVBoxLayout(dlg)
        form = QFormLayout()
        layout.addLayout(form)

        form.addRow("Start (0–1):", self.window_start_spin)
        form.addRow("End (0–1):", self.window_end_spin)

        self.central_width_spin = QDoubleSpinBox()
        self.central_width_spin.setRange(0.05, 1.0)
        self.central_width_spin.setSingleStep(0.05)
        self.central_width_spin.setValue(0.30)
        form.addRow("Centred width (0–1):", self.central_width_spin)
        layout.addWidget(QLabel("Drag across the stack to draw a window."))

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add range")
        add_btn.clicked.connect(self.add_litho_window)
        btn_row.addWidget(add_btn)

        clear_btn = QPushButton("Clear all")
        clear_btn.clicked.connect(self.clear_litho_windows)
        btn_row.addWidget(clear_btn)

        add_central_btn = QPushButton("Add centred window")

        def _add_central_window():
            width = float(self.central_width_spin.value())
            width = max(0.0, min(1.0, width))
            if width <= 0.0:
                QMessageBox.warning(self, "Check window width", "Enter a width greater than 0.")
                return
            start = max(0.0, 0.5 - width / 2.0)
            end = min(1.0, 0.5 + width / 2.0)
            if end <= start:
                QMessageBox.warning(self, "Check window width", "Keep the window within the stack.")
                return
            self.window_start_spin.setValue(start)
            self.window_end_spin.setValue(end)
            self.add_litho_window()

        add_central_btn.clicked.connect(_add_central_window)
        btn_row.addWidget(add_central_btn)

        draw_btn = QPushButton("Draw on canvas")
        draw_btn.setToolTip("Drag across the stack to add a window")
        draw_btn.clicked.connect(self.start_litho_window_drawing)
        btn_row.addWidget(draw_btn)

        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        layout.addWidget(QLabel("Windows:"))
        layout.addWidget(self.window_list)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.close)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

        dlg.show()

    def start_litho_window_drawing(self):
        if getattr(self, "xsec", None) is None:
            QMessageBox.warning(
                self,
                "Create or open a process first",
                "A process is required to draw a lithography window.",
            )
            return

        if hasattr(self, "op_combo"):
            idx = self.op_combo.findText("Lithography", Qt.MatchFixedString)
            if idx >= 0 and self.op_combo.currentIndex() != idx:
                self.op_combo.setCurrentIndex(idx)
                self.update_op_controls()

        self.dragging_litho_window = False
        self.litho_drag_start_frac = None
        self.litho_drag_current_frac = None
        self._clear_litho_drag_preview()

        if hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.setFocus()
            self.canvas.draw_idle()
        QTimer.singleShot(0, self._activate_canvas_for_litho_drawing)
        if hasattr(self, "statusBar"):
            self.statusBar().showMessage("Drag across the stack to draw a lithography window.", 6000)

    def _activate_canvas_for_litho_drawing(self):
        for dialog_name in ("window_dialog", "llm_chat_dialog"):
            dialog = getattr(self, dialog_name, None)
            if dialog is not None and dialog.isVisible():
                dialog.lower()

        self.show()
        self.raise_()
        self.activateWindow()

        if hasattr(self, "figure_scroll") and self.figure_scroll is not None:
            self.figure_scroll.setFocus(Qt.OtherFocusReason)
        if hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.setFocus(Qt.OtherFocusReason)
            self.canvas.draw_idle()

    def open_process_flow_dialog(self):
        # Process flow is now embedded in the left panel.
        if not hasattr(self, "history_list"):
            return
        row = self.history_list.currentRow()
        if row >= 0:
            item = self.history_list.item(row)
            if item is not None:
                self.history_list.scrollToItem(item)
        self.history_list.setFocus()

    # ---------- Bond to external stack ----------
    def _set_bond_source_info(self, info):
        """Store/update the currently selected external process for Bond."""
        self.bond_source_info = info
        if not hasattr(self, "bond_target_file_label"):
            return
        if not info:
            self.bond_target_file_label.setText("No process selected")
            self.bond_target_file_label.setToolTip("Choose a FabStack process to bond")
            return

        path = str(info.get("path", "")).strip()
        base = os.path.basename(path) if path else "Selected process"
        steps = info.get("history_titles", [])
        count = len(steps) if isinstance(steps, list) else 0
        self.bond_target_file_label.setText(f"{base} ({count} steps)")
        self.bond_target_file_label.setToolTip(path or base)

    def clear_bond_source_selection(self):
        self._set_bond_source_info(None)

    def _copy_bond_source_info(self, info):
        if not isinstance(info, dict):
            return None
        copied = {}
        for key, value in info.items():
            if key in {
                "grid",
                "row_heights_um",
                "structural_role_grid",
                "planarizer_role_grid",
            }:
                copied[key] = np.asarray(value).copy()
            elif key in {"structural_bodies", "planarizer_roles"}:
                copied[key] = copy.deepcopy(value) if isinstance(value, dict) else {}
            elif key == "visual_geometry":
                copied[key] = list(value) if isinstance(value, list) else []
            elif isinstance(value, list):
                copied[key] = list(value)
            else:
                copied[key] = value
        return copied

    def _read_process_file_data(self, path: str):
        return load_process_file(path)

    def _build_bond_source_info(self, data: dict, path: str):
        history = data.get("history", [])
        if not isinstance(history, list) or not history:
            raise ValueError("Selected file has no process history.")

        last = history[-1]
        if not isinstance(last, dict) or "grid" not in last:
            raise ValueError("Selected file has no valid final stack.")

        src_grid = np.asarray(last.get("grid"))
        if src_grid.ndim != 2 or src_grid.size == 0:
            raise ValueError("Selected file has an invalid stack grid.")

        src_names = data.get("material_names", [])
        src_colors = data.get("material_colors", [])
        if not isinstance(src_names, list) or not src_names:
            raise ValueError("Selected file has no material list.")
        if not isinstance(src_colors, list):
            src_colors = []
        if len(src_colors) < len(src_names):
            src_colors = src_colors + (["#cccccc"] * (len(src_names) - len(src_colors)))

        src_void = int(data.get("void_index", 0))

        titles = []
        for st in history:
            if isinstance(st, dict):
                t = str(st.get("title", "")).strip()
                titles.append(t if t else "(untitled)")
            else:
                titles.append("(untitled)")

        if np.any(src_grid != src_void):
            max_idx = int(np.max(src_grid))
            if max_idx >= len(src_names):
                raise ValueError("Selected file contains invalid material indices.")

        source_heights = np.asarray(
            last.get("row_heights_um", np.ones(src_grid.shape[0], dtype=float)),
            dtype=float,
        )
        source_roles = np.asarray(
            last.get("structural_role_grid", np.zeros_like(src_grid, dtype=np.int16)),
            dtype=np.int16,
        )
        source_bodies = copy.deepcopy(last.get("structural_bodies", {}) or {})
        source_planarizer_grid = np.asarray(
            last.get("planarizer_role_grid", np.zeros_like(src_grid, dtype=np.int32)),
            dtype=np.int32,
        )
        source_planarizer_roles = copy.deepcopy(last.get("planarizer_roles", {}) or {})
        if source_roles.shape != src_grid.shape:
            source_roles = np.zeros_like(src_grid, dtype=np.int16)
        if source_planarizer_grid.shape != src_grid.shape:
            source_planarizer_grid = np.zeros_like(src_grid, dtype=np.int32)
        if not np.any(source_roles > 0):
            substrate_name = str(data.get("substrate_name", "") or "").strip()
            try:
                substrate_idx = int(src_names.index(substrate_name))
            except ValueError:
                substrate_idx = -1
            if substrate_idx >= 0:
                for col in range(src_grid.shape[1]):
                    rows = np.where(src_grid[:, col] == substrate_idx)[0]
                    if rows.size:
                        source_roles[rows, col] = 1
                source_bodies = {
                    1: {
                        "body_id": 1,
                        "material_index": substrate_idx,
                        "nominal_thickness_um": float(data.get("substrate_thickness", 0.0) or 0.0),
                        "kind": "substrate",
                    }
                }

        return {
            "path": path,
            "void_index": src_void,
            "material_names": [str(x) for x in src_names],
            "material_colors": [str(x) for x in src_colors],
            "history_titles": titles,
            "grid": src_grid.copy(),
            "row_heights_um": source_heights.copy(),
            "visual_geometry": list(last.get("visual_geometry", []) or []),
            "structural_role_grid": source_roles.copy(),
            "structural_bodies": source_bodies,
            "planarizer_role_grid": source_planarizer_grid.copy(),
            "planarizer_roles": source_planarizer_roles,
        }

    def select_bond_target_file(self):
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Choose process to bond",
            "",
            "FabStack process (*.fab);;All Files (*)"
        )
        if not fname:
            return

        if self.current_process_path:
            try:
                if os.path.abspath(fname) == os.path.abspath(self.current_process_path):
                    QMessageBox.warning(
                        self,
                        "Choose another process",
                        "A process cannot be bonded to itself."
                    )
                    return
            except Exception:
                pass

        try:
            data = self._read_process_file_data(fname)
            info = self._build_bond_source_info(data, fname)
        except Exception:
            LOGGER.exception("Could not open bond process %s", fname)
            QMessageBox.warning(
                self,
                "Couldn’t open bond process",
                "Choose another FabStack process file.",
            )
            return

        self._set_bond_source_info(info)

    def _remap_bond_source_grid_to_local(self, source_info: dict):
        """
        Map an external process grid to local material indices.
        Returns (mapped_grid, added_material_names).
        """
        src_grid = np.asarray(source_info.get("grid"))
        src_void = int(source_info.get("void_index", 0))
        src_names = source_info.get("material_names", [])
        src_colors = source_info.get("material_colors", [])

        if src_grid.ndim != 2:
            raise ValueError("Invalid external stack grid.")

        mapped = np.full(src_grid.shape, self.VOID_INDEX, dtype=self.xsec.grid.dtype)
        added_names = []
        index_map = {src_void: self.VOID_INDEX}

        for src_idx in np.unique(src_grid):
            src_idx = int(src_idx)
            if src_idx == src_void:
                index_map[src_idx] = self.VOID_INDEX
                continue
            if src_idx < 0 or src_idx >= len(src_names):
                continue

            src_name = str(src_names[src_idx]).strip()
            if not src_name or src_name == "Void":
                index_map[src_idx] = self.VOID_INDEX
                continue

            dst_idx = self.material_index(src_name)
            if dst_idx < 0:
                src_color = "#cccccc"
                if src_idx < len(src_colors):
                    c = str(src_colors[src_idx]).strip()
                    if c:
                        src_color = c
                if hasattr(self, "_unique_material_color"):
                    src_color = self._unique_material_color(src_color, exclude_name=src_name)
                self.material_names.append(src_name)
                self.material_colors.append(src_color)
                if src_name != "Void":
                    if self._material_combo_index_by_full_name(src_name) < 0:
                        self.material_combo.addItem(src_name)
                    if self.substrate_combo.findText(src_name) < 0:
                        self.substrate_combo.addItem(src_name)
                self.ensure_in_library(src_name)
                dst_idx = len(self.material_names) - 1
                added_names.append(src_name)
            else:
                # If it was soft-deleted, restore it because imported stack uses it.
                if src_name in self.deleted_materials:
                    self.deleted_materials.discard(src_name)
                    if src_name != "Void":
                        if self._material_combo_index_by_full_name(src_name) < 0:
                            self.material_combo.addItem(src_name)
                        if self.substrate_combo.findText(src_name) < 0:
                            self.substrate_combo.addItem(src_name)

            index_map[src_idx] = dst_idx

        for src_idx, dst_idx in index_map.items():
            mapped[src_grid == src_idx] = dst_idx

        return mapped, added_names

    def _apply_bond_with_external_stack(self, bond_material_index: int, bond_thickness_um: float):
        if self.bond_source_info is None:
            raise ValueError("No external process selected.")
        if self.xsec is None:
            raise ValueError("No active stack to bond.")

        ext_grid, added_names = self._remap_bond_source_grid_to_local(self.bond_source_info)
        ext_heights = np.asarray(
            self.bond_source_info.get("row_heights_um", np.ones(ext_grid.shape[0], dtype=float)),
            dtype=float,
        )
        if len(ext_heights) != ext_grid.shape[0]:
            ext_heights = np.ones(ext_grid.shape[0], dtype=float)
        cur_grid = self.xsec.grid
        self.xsec._ensure_structural_roles()
        cur_roles = self.xsec.copy_structural_roles()
        cur_planarizer_roles = self.xsec.copy_planarizer_roles()
        cur_heights = (
            self.xsec.copy_row_heights()
            if hasattr(self.xsec, "copy_row_heights")
            else np.ones(cur_grid.shape[0], dtype=float)
        )

        if ext_grid.shape[1] != cur_grid.shape[1]:
            raise ValueError(
                f"Width mismatch (current: {cur_grid.shape[1]}, selected: {ext_grid.shape[1]}).\n"
                "Use process files with the same wafer width."
            )

        ext_non_void_rows = np.where(np.any(ext_grid != self.VOID_INDEX, axis=1))[0]
        if ext_non_void_rows.size == 0:
            raise ValueError("Selected process has no non-empty stack to bond.")

        # Keep only active stack, then flip so the external top surface faces down.
        ext_top = int(ext_non_void_rows.min())
        ext_bottom = int(ext_non_void_rows.max())
        ext_block = ext_grid[ext_top:ext_bottom + 1, :]
        ext_block = np.flipud(ext_block)
        ext_block_heights = ext_heights[ext_top:ext_bottom + 1][::-1]
        ext_roles = np.asarray(
            self.bond_source_info.get(
                "structural_role_grid",
                np.zeros_like(ext_grid, dtype=np.int16),
            ),
            dtype=np.int16,
        )
        if ext_roles.shape != ext_grid.shape:
            ext_roles = np.zeros_like(ext_grid, dtype=np.int16)
        original_ext_block_roles = np.flipud(
            ext_roles[ext_top:ext_bottom + 1, :]
        ).copy()
        ext_block_roles = np.zeros_like(original_ext_block_roles, dtype=np.int16)
        source_planarizer_grid = np.asarray(
            self.bond_source_info.get(
                "planarizer_role_grid",
                np.zeros_like(ext_grid, dtype=np.int32),
            ),
            dtype=np.int32,
        )
        if source_planarizer_grid.shape != ext_grid.shape:
            source_planarizer_grid = np.zeros_like(ext_grid, dtype=np.int32)
        original_ext_planarizer = np.flipud(
            source_planarizer_grid[ext_top:ext_bottom + 1, :]
        ).copy()
        ext_block_planarizer = np.zeros_like(original_ext_planarizer, dtype=np.int32)

        # Imported body ids are local to the source process. Remap them to
        # fresh ids so both wafers retain independent structural provenance.
        imported_bodies = self.bond_source_info.get("structural_bodies", {}) or {}
        role_id_map = {}
        for old_id in sorted(
            int(value) for value in np.unique(original_ext_block_roles) if int(value) > 0
        ):
            cells = np.where(original_ext_block_roles == old_id)
            if cells[0].size == 0:
                continue
            mapped_material = int(ext_block[cells[0][0], cells[1][0]])
            source_meta = imported_bodies.get(old_id, imported_bodies.get(str(old_id), {}))
            nominal = float((source_meta or {}).get("nominal_thickness_um", 0.0) or 0.0)
            if nominal <= 0.0:
                nominal = float(np.sum(
                    ext_block_heights[np.any(original_ext_block_roles == old_id, axis=1)]
                ))
            new_id = self.xsec._new_structural_body(
                mapped_material,
                nominal,
                kind="bonded_substrate",
            )
            role_id_map[old_id] = new_id
        for old_id, new_id in role_id_map.items():
            ext_block_roles[original_ext_block_roles == old_id] = int(new_id)

        imported_planarizers = self.bond_source_info.get("planarizer_roles", {}) or {}
        planarizer_id_map = {}
        for old_id in sorted({
            abs(int(value))
            for value in np.unique(original_ext_planarizer)
            if int(value) != 0
        }):
            cells = np.where(np.abs(original_ext_planarizer) == int(old_id))
            if cells[0].size == 0:
                continue
            mapped_material = int(ext_block[cells[0][0], cells[1][0]])
            source_meta = imported_planarizers.get(
                old_id,
                imported_planarizers.get(str(old_id), {}),
            )
            requested_um = float(
                (source_meta or {}).get("requested_thickness_um", 0.0) or 0.0
            )
            fill_thicknesses = (source_meta or {}).get(
                "fill_thickness_um_by_column",
                None,
            )
            try:
                source_orientation = (
                    1 if int((source_meta or {}).get("orientation", 1) or 1) >= 0 else -1
                )
            except (TypeError, ValueError):
                source_orientation = 1
            new_id = self.xsec._new_planarizer_role(
                mapped_material,
                requested_um,
                fill_thickness_um_by_column=fill_thicknesses,
                # The imported block is reversed vertically before it is
                # attached to the current device.
                orientation=-source_orientation,
            )
            planarizer_id_map[int(old_id)] = int(new_id)
        for old_id, new_id in planarizer_id_map.items():
            ext_block_planarizer[original_ext_planarizer == old_id] = int(new_id)
            ext_block_planarizer[original_ext_planarizer == -old_id] = -int(new_id)

        cur_non_void_rows = np.where(np.any(cur_grid != self.VOID_INDEX, axis=1))[0]
        cur_top = int(cur_non_void_rows.min()) if cur_non_void_rows.size else int(cur_grid.shape[0])

        parts = []
        role_parts = []
        planarizer_parts = []
        heights = []
        if cur_top > 0:
            parts.append(cur_grid[:cur_top, :])
            role_parts.append(cur_roles[:cur_top, :])
            planarizer_parts.append(cur_planarizer_roles[:cur_top, :])
            heights.append(cur_heights[:cur_top])

        parts.append(ext_block)
        role_parts.append(ext_block_roles)
        planarizer_parts.append(ext_block_planarizer)
        heights.append(ext_block_heights)

        bond_thickness_um = max(0.0, float(bond_thickness_um))
        if bond_thickness_um > 0.0:
            parts.append(
                np.full((1, cur_grid.shape[1]), bond_material_index, dtype=cur_grid.dtype)
            )
            role_parts.append(np.zeros((1, cur_grid.shape[1]), dtype=np.int16))
            planarizer_parts.append(np.zeros((1, cur_grid.shape[1]), dtype=np.int32))
            heights.append(np.asarray([bond_thickness_um], dtype=float))

        parts.append(cur_grid[cur_top:, :])
        role_parts.append(cur_roles[cur_top:, :])
        planarizer_parts.append(cur_planarizer_roles[cur_top:, :])
        heights.append(cur_heights[cur_top:])

        merged = np.vstack(parts)
        merged_roles = np.vstack(role_parts)
        merged_planarizer = np.vstack(planarizer_parts)
        merged_heights = np.concatenate(heights)
        self.xsec.grid = merged
        self.xsec.structural_role_grid = merged_roles
        self.xsec.planarizer_role_grid = merged_planarizer
        self.xsec.row_heights_um = merged_heights
        if hasattr(self.xsec, "visual_geometry"):
            self.xsec.visual_geometry = []
        self.xsec.height, self.xsec.width = merged.shape
        if hasattr(self.xsec, "_merge_adjacent_equal_rows"):
            self.xsec._merge_adjacent_equal_rows()

        src_path = str(self.bond_source_info.get("path", "")).strip()
        src_name = os.path.basename(src_path) if src_path else "external process"
        step_metadata = {
            "bond_source": {
                "path": src_path,
                "history_titles": list(self.bond_source_info.get("history_titles", [])),
            }
        }
        return src_name, step_metadata, added_names

    def _is_lithography_selected(self) -> bool:
        return hasattr(self, "op_combo") and self.op_combo.currentText() == "Lithography"

    def _event_to_window_fraction(self, event):
        """
        Convert a canvas mouse event into a normalized horizontal fraction [0, 1].
        Returns None if event is outside the active plotting axes.
        """
        if self.xsec is None:
            return None
        if event is None or event.inaxes != self.canvas.ax or event.xdata is None:
            return None
        width = max(1, int(getattr(self.xsec, "width", 1)))
        x = float(event.xdata)
        # Matplotlib renders cell edges from -0.5 through width - 0.5.
        if x < -0.5 or x > (width - 0.5):
            return None
        frac = (x + 0.5) / float(width)
        return max(0.0, min(1.0, frac))

    def _active_stack_vertical_span(self, grid):
        """
        Return (y0, height) for the non-void stack region in plot coordinates.
        """
        h = int(grid.shape[0])
        non_void_rows = np.where(np.any(grid != self.VOID_INDEX, axis=1))[0]
        if non_void_rows.size == 0:
            return -0.5, float(h)
        top = int(non_void_rows.min())
        bottom = int(non_void_rows.max())
        y0 = float(top) - 0.5
        height = float(bottom - top + 1)
        return y0, height

    def _clear_litho_drag_preview(self):
        if self.litho_drag_patch is None:
            return
        try:
            self.litho_drag_patch.remove()
        except Exception:
            pass
        self.litho_drag_patch = None

    def _clear_litho_preview_artists(self, redraw: bool = False):
        """
        Remove any lithography preview rectangles currently drawn on the canvas.
        """
        self._clear_litho_drag_preview()
        if not hasattr(self, "canvas") or self.canvas is None:
            return
        if not hasattr(self.canvas, "fig") or self.canvas.fig is None:
            return

        # Remove tagged preview patches from every axis in the figure to avoid
        # stale rectangles if axes/layout are rebuilt.
        for ax in list(self.canvas.fig.axes):
            to_remove = []
            for patch in list(ax.patches):
                gid = patch.get_gid() if hasattr(patch, "get_gid") else None
                if gid in ("litho_preview", "litho_drag_preview"):
                    to_remove.append(patch)
            for patch in to_remove:
                try:
                    patch.remove()
                except Exception:
                    pass
        if redraw:
            self.canvas.draw_idle()

    def add_litho_window_range(self, s: float, e: float, *, show_warning: bool = True, refresh_plot: bool = False):
        s = max(0.0, min(1.0, float(s)))
        e = max(0.0, min(1.0, float(e)))
        if e <= s:
            if show_warning:
                QMessageBox.warning(self, "Check window range", "End must be greater than start.")
            return False

        self.litho_windows.append((s, e))
        self.window_list.addItem(f"{s:.2f} – {e:.2f}")
        self.window_start_spin.setValue(s)
        self.window_end_spin.setValue(e)

        if hasattr(self, "window_count_label"):
            self.window_count_label.setText(f"Windows: {len(self.litho_windows)}")

        if refresh_plot:
            self.update_plot(self.current_title or "")
        return True

    def add_litho_window(self):
        s = float(self.window_start_spin.value())
        e = float(self.window_end_spin.value())
        self.add_litho_window_range(s, e, show_warning=True, refresh_plot=True)

    def clear_litho_windows(self, _checked: bool = False, *, refresh_plot: bool = True):
        self.litho_windows.clear()
        self.window_list.clear()
        self.dragging_litho_window = False
        self.litho_drag_start_frac = None
        self.litho_drag_current_frac = None
        self._clear_litho_preview_artists(redraw=False)
        if hasattr(self, "window_count_label"):
            self.window_count_label.setText("Windows: 0")
        if refresh_plot and self.xsec is not None:
            self.update_plot(self.current_title or "")

    def effective_lithography_windows(self):
        """Return the explicit windows shared by preview and execution."""
        windows = []
        for entry in getattr(self, "litho_windows", []) or []:
            try:
                start, end = entry
                start = max(0.0, min(1.0, float(start)))
                end = max(0.0, min(1.0, float(end)))
            except (TypeError, ValueError):
                continue
            if end > start:
                windows.append((start, end))
        return windows

    def get_geometry_etch_mask(self, etch_index: int):
        grid = self.xsec.grid
        h, w = grid.shape
        top = np.full(w, self.VOID_INDEX, dtype=int)
        for x in range(w):
            col = grid[:, x]
            nz = np.where(col != self.VOID_INDEX)[0]
            if nz.size:
                top[x] = col[nz.min()]
        non_void_top = top[top != self.VOID_INDEX]
        if non_void_top.size == 0:
            return np.ones(w, dtype=bool)

        has_etch_top = np.any(top == etch_index)
        has_other_top = np.any((top != etch_index) & (top != self.VOID_INDEX))

        if has_etch_top and has_other_top:
            return (top == etch_index)

        return np.ones(w, dtype=bool)

    # ---------- helpers ----------
    def format_temp(self):
        t = self.temp_spin.value()
        return f"{t:.1f} °C" if abs(t) > 0.0 else ""

    def format_exposure(self):
        e = self.exposure_spin.value()
        return f"{e:.1f} s" if e > 0.0 else ""

    def format_um_value(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return str(value)
        text = f"{value:.9f}".rstrip("0").rstrip(".")
        return text if text else "0"

    def _darker_color(self, hex_color: str, factor: float = 0.6) -> str:
        if not hex_color or not hex_color.startswith("#") or len(hex_color) != 7:
            return hex_color
        try:
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
        except ValueError:
            return hex_color
        r = max(0, min(255, int(r * factor)))
        g = max(0, min(255, int(g * factor)))
        b = max(0, min(255, int(b * factor)))
        return "#{:02x}{:02x}{:02x}".format(r, g, b)

    def _bake_material(self, idx: int) -> int:
        if idx <= 0 or idx >= len(self.material_names):
            return idx

        original_name = self.material_names[idx]
        if original_name.endswith("baked/annealed"):
            return idx

        baked_name = f"{original_name} baked/annealed"
        if baked_name in self.material_names:
            return self.material_names.index(baked_name)

        base_color = self.material_colors[idx]
        darker = self._darker_color(base_color, factor=0.76)
        if hasattr(self, "_unique_material_color"):
            darker = self._unique_material_color(darker, exclude_name=baked_name)

        self.material_names.append(baked_name)
        self.material_colors.append(darker)
        self.material_combo.addItem(baked_name)
        self.substrate_combo.addItem(baked_name)
        self.refresh_operation_material_combo(preferred_name=baked_name)

        self.refresh_materials_list()
        return len(self.material_names) - 1

    def _lithography_invert_from_text(self, text: str):
        low = str(text or "").strip().lower()
        if "negative" in low:
            return True
        if "positive" in low:
            return False
        return None

    def _default_lithography_invert_for_material(self, material_name: str):
        low = str(material_name or "").strip().lower()
        if not low:
            return None
        if any(term in low for term in ("negative", "su-8", "su8", "hsq", "nlof")):
            return True
        if any(term in low for term in ("positive", "dnq", "novolac")):
            return False
        return None

    def _lithography_method_for_invert(self, invert: bool) -> str:
        return "Negative lithography" if bool(invert) else "Positive lithography"

    def _set_lithography_invert(self, invert: bool, *, sync_method: bool = True):
        if not hasattr(self, "invert_lithography_checkbox"):
            return

        self._syncing_lithography_invert = True
        try:
            self.invert_lithography_checkbox.setChecked(bool(invert))
        finally:
            self._syncing_lithography_invert = False

        if sync_method and hasattr(self, "etchant_combo") and self.op_combo.currentText() == "Lithography":
            self._set_etchant_combo_value(self._lithography_method_for_invert(bool(invert)))

    def _current_lithography_invert(self) -> bool:
        if hasattr(self, "invert_lithography_checkbox"):
            return bool(self.invert_lithography_checkbox.isChecked())

        combo_value = ""
        if hasattr(self, "etchant_combo"):
            combo_value = self.etchant_combo.currentData() or self.etchant_combo.currentText()
        method_invert = self._lithography_invert_from_text(combo_value)
        if method_invert is not None:
            return bool(method_invert)
        material_invert = self._default_lithography_invert_for_material(self.current_operation_material_name())
        if material_invert is not None:
            return bool(material_invert)
        return False

    def _sync_lithography_invert_from_current_method(self):
        if not hasattr(self, "etchant_combo"):
            return
        method = self.etchant_combo.currentData() or self.etchant_combo.currentText()
        invert = self._lithography_invert_from_text(method)
        if invert is not None:
            self._set_lithography_invert(bool(invert), sync_method=False)

    def _sync_lithography_invert_from_current_material(self):
        invert = self._default_lithography_invert_for_material(self.current_operation_material_name())
        if invert is not None:
            self._set_lithography_invert(bool(invert), sync_method=True)

    def on_invert_lithography_toggled(self, checked: bool):
        if getattr(self, "_syncing_lithography_invert", False):
            return
        if self.op_combo.currentText() != "Lithography":
            return
        self._set_lithography_invert(bool(checked), sync_method=True)

    def on_operation_material_changed(self, _text: str = ""):
        if not hasattr(self, "op_combo") or self.op_combo.currentText() != "Lithography":
            return
        self._sync_lithography_invert_from_current_material()

    # ---------- Apply operations ----------
    def apply_operation(self):
        if self.xsec is None:
            if hasattr(self, "statusBar"):
                self.statusBar().showMessage("Create or open a process first.", 3500)
            return
        op = self.op_combo.currentText()
        material_name = self.current_operation_material_name()
        mat_index = self.material_index(material_name)
        value_overrides = getattr(self, "_operation_value_overrides", None)
        using_value_overrides = isinstance(value_overrides, dict)

        def overridden_float(key, fallback):
            if using_value_overrides and key in value_overrides:
                try:
                    return float(value_overrides.get(key))
                except (TypeError, ValueError):
                    return float(fallback)
            return float(fallback)

        thick_um = overridden_float("thickness_um", self.thick_spin.value())
        depth_um = overridden_float("depth_um", self.depth_spin.value())
        title = ""
        step_metadata = None
        step_extra = {"operation": op}
        force_log_no_change = False
        if material_name:
            step_extra["material_name"] = material_name

        if using_value_overrides:
            if "temperature_c" in value_overrides:
                temp_value = overridden_float("temperature_c", 0.0)
                temp_str = f"{temp_value:.1f} °C" if abs(temp_value) > 0.0 else ""
            else:
                temp_str = ""
            if "time_s" in value_overrides:
                time_value = overridden_float("time_s", 0.0)
                expo_str = f"{time_value:.1f} s" if time_value > 0.0 else ""
            else:
                expo_str = ""
        else:
            temp_str = self.format_temp()
            expo_str = self.format_exposure()

        restricted_ops = {
            "Deposit",
            "Spincoating",
            "Bond",
            "Etch",
            "Strip material",
            "Lift-off",
            "Clean / Surface prep",
            "Bake/Annealing",
            "Lithography",
        }
        if op in restricted_ops and self._is_material_restricted_for_operation(op, material_name):
            QMessageBox.warning(
                self,
                "Choose another material",
                f"{material_name} cannot be used for {operation_label(op)}.",
            )
            return

        material_for_recent = None
        grid_before = self.xsec.copy_state()
        row_heights_before = (
            self.xsec.copy_row_heights()
            if hasattr(self.xsec, "copy_row_heights")
            else None
        )

        if op == "Deposit":
            if mat_index < 0:
                QMessageBox.warning(self, "Select a material", "Choose a material for this step.")
                return
            execute_ui_material_operation(
                self,
                op,
                mat_index,
                thickness_um=float(thick_um),
            )
            display_um = thick_um
            base = f"Deposit {material_name} ({self.format_um_value(display_um)} µm)"
            method_parts = []
            dep_method = (self.deposit_method_combo.currentText() or "").strip()
            if dep_method and dep_method != "No compatible methods":
                method_parts.append(dep_method)
            if temp_str:
                method_parts.append(temp_str)
            if expo_str:
                method_parts.append(f"Time {expo_str}")
            if self.include_method_in_step_titles and method_parts:
                title = " — ".join([base] + method_parts)
            else:
                title = base
            material_for_recent = material_name

        elif op == "Spincoating":
            if mat_index < 0:
                QMessageBox.warning(self, "Select a material", "Choose a material for this step.")
                return
            execute_ui_material_operation(
                self,
                op,
                mat_index,
                thickness_um=float(thick_um),
            )
            display_um = thick_um
            base = f"Spin coat {material_name} ({self.format_um_value(display_um)} µm)"
            method_parts = []
            if using_value_overrides:
                rpm_val = overridden_float("temperature_c", 0.0) if "temperature_c" in value_overrides else 0.0
            else:
                rpm_val = self.temp_spin.value()
            if rpm_val > 0:
                method_parts.append(f"{rpm_val:.0f} rpm")
            if expo_str:
                method_parts.append(f"Time {expo_str}")
            if self.include_method_in_step_titles and method_parts:
                title = " — ".join([base] + method_parts)
            else:
                title = base
            material_for_recent = material_name

        elif op == "Bond":
            if mat_index < 0:
                QMessageBox.warning(self, "Select a material", "Choose a material for this step.")
                return
            source_name_for_title = ""
            bond_source_info_for_replay = None
            if self.bond_source_info is not None:
                bond_source_info_for_replay = self._copy_bond_source_info(self.bond_source_info)
                try:
                    (
                        source_name_for_title,
                        step_metadata,
                        added_names,
                    ) = execute_ui_external_bond_operation(
                        self, mat_index, float(thick_um)
                    )
                except Exception:
                    LOGGER.exception("Bond operation failed for %s", material_name)
                    QMessageBox.warning(
                        self,
                        "Bond failed",
                        "Check the selected process and try again.",
                    )
                    return

                if added_names:
                    self.refresh_materials_list()
                    self.refresh_operation_material_combo(preferred_name=material_name)
                    self.update_etchant_options()
                    self.update_deposit_methods_for_material()
            else:
                execute_ui_simple_bond_operation(
                    self,
                    mat_index,
                    float(thick_um),
                )
            if bond_source_info_for_replay is not None:
                step_extra["bond_source_info"] = bond_source_info_for_replay
            display_um = thick_um
            if source_name_for_title:
                base = f"Bond {material_name} ({self.format_um_value(display_um)} µm) to {source_name_for_title}"
            else:
                base = f"Bond {material_name} ({self.format_um_value(display_um)} µm)"
            method_parts = []
            method_chem = chem_normalize_ascii(
                (self.etchant_combo.currentData() or self.etchant_combo.currentText() or "").strip()
            )
            method_fmt = chem_subs_unicode(method_chem) if method_chem else ""
            if method_fmt:
                method_parts.append(method_fmt)
            if temp_str:
                method_parts.append(temp_str)
            if self.include_method_in_step_titles and method_parts:
                title = " — ".join([base] + method_parts)
            else:
                title = base
            material_for_recent = material_name

        elif op == "Etch":
            if mat_index < 0:
                QMessageBox.warning(self, "Select a material", "Choose a material for this step.")
                return
            etch_mode = getattr(self, "current_etch_mode", "anisotropic")
            step_extra["etch_mode"] = etch_mode
            mask = self.get_geometry_etch_mask(mat_index)
            base = f"Etch {material_name} ({self.format_um_value(depth_um)} µm)"
            mode_tag = "isotropic" if etch_mode == "isotropic" else "anisotropic"
            base = f"{base} {mode_tag}"
            method_parts = []
            etch_base = self.etchant_combo.currentData() or self.etchant_combo.currentText()
            etch_base = chem_normalize_ascii((etch_base or "").strip())
            etch_str_fmt = chem_subs_unicode(etch_base) if etch_base else ""
            if etch_str_fmt:
                if self.include_method_in_step_titles:
                    step_extra["method"] = etch_base
                method_parts.append(etch_str_fmt)
            if temp_str:
                method_parts.append(temp_str)
            if expo_str:
                method_parts.append(f"Time {expo_str}")

            if float(depth_um) <= 0.0:
                # Requested etch is below the current simulation resolution.
                # Keep the operation in process flow as an explicit no-op.
                base = f"{base} (no geometry change)"
                force_log_no_change = True
            else:
                execute_ui_material_operation(
                    self,
                    op,
                    mat_index,
                    depth_um=float(depth_um),
                    mask=mask,
                    etch_mode=etch_mode,
                )

            if self.include_method_in_step_titles and method_parts:
                title = " — ".join([base] + method_parts)
            else:
                title = base
            material_for_recent = material_name

        elif op == "Strip material":
            if mat_index < 0:
                QMessageBox.warning(self, "Select a material", "Choose a material for this step.")
                return
            execute_ui_material_operation(self, op, mat_index)
            base = f"Strip {material_name}"
            method_parts = []
            method_chem = chem_normalize_ascii(
                (self.etchant_combo.currentData() or self.etchant_combo.currentText() or "").strip()
            )
            method_fmt = chem_subs_unicode(method_chem) if method_chem else ""
            if method_fmt:
                method_parts.append(method_fmt)
            if temp_str:
                method_parts.append(temp_str)
            if expo_str:
                method_parts.append(expo_str)
            if self.include_method_in_step_titles and method_parts:
                title = " — ".join([base] + method_parts)
            else:
                title = base
            material_for_recent = material_name

        elif op == "Lift-off":
            # Lift-off requires a photoresist material to be selected. We remove
            # the selected photoresist and any film deposited on top of it.
            if mat_index < 0 or not self.is_photoresist_name(material_name):
                QMessageBox.warning(
                    self,
                    "Select a photoresist",
                    "Choose a material from the Photoresists category.",
                )
                return
            # Perform the lift-off on the cross-section
            try:
                execute_ui_material_operation(self, op, mat_index)
            except Exception:
                LOGGER.exception("Lift-off failed for %s", material_name)
                QMessageBox.warning(
                    self,
                    "Lift-off failed",
                    "Check the selected material and method, then try again.",
                )
                return

            base = f"Lift-off {material_name}"
            method_parts = []

            # Use the etchant selected to indicate the solvent/stripper used for lift-off
            method_chem = chem_normalize_ascii(
                (self.etchant_combo.currentData() or self.etchant_combo.currentText() or "").strip()
            )
            method_fmt = chem_subs_unicode(method_chem) if method_chem else ""
            if method_fmt:
                method_parts.append(method_fmt)
            if temp_str:
                method_parts.append(temp_str)
            if expo_str:
                method_parts.append(expo_str)

            if self.include_method_in_step_titles and method_parts:
                title = " — ".join([base] + method_parts)
            else:
                title = base

            # Lift-off does not add a material back to the stack, so we do not mark any material as recent
            material_for_recent = None

        elif op == "Clean / Surface prep":
            execute_ui_material_operation(self, op, mat_index)
            method_chem = chem_normalize_ascii(
                (self.etchant_combo.currentData() or self.etchant_combo.currentText() or "").strip()
            )
            method_fmt = chem_subs_unicode(method_chem) if method_chem else ""
            target = material_name if mat_index >= 0 else "wafer"

            base = f"Clean / surface prep {target}"
            method_parts = []
            if method_fmt:
                method_parts.append(method_fmt)
            if temp_str:
                method_parts.append(temp_str)
            if expo_str:
                method_parts.append(f"Time {expo_str}")

            if self.include_method_in_step_titles and method_parts:
                title = " — ".join([base] + method_parts)
            else:
                title = base

            material_for_recent = material_name if mat_index >= 0 else None

        elif op == "Bake/Annealing":
            if mat_index < 0:
                QMessageBox.warning(self, "Select a material", "Choose a material to bake or anneal.")
                return

            baked_idx = execute_ui_bake_operation(self, mat_index)

            if baked_idx != mat_index and self.xsec is not None:
                baked_name = self.material_names[baked_idx]
                target = baked_name
                material_for_recent = baked_name
            else:
                target = self.material_names[mat_index] + " baked"
                material_for_recent = self.material_names[mat_index]

            base = f"Bake / anneal {target}"
            method_parts = []
            bake_method = str(getattr(self, "_assistant_bake_method_override", "") or "").strip()
            if bake_method:
                step_extra["method"] = bake_method
                method_parts.append(bake_method)
            if temp_str:
                method_parts.append(temp_str)
            if expo_str:
                method_parts.append(f"Time {expo_str}")

            if self.include_method_in_step_titles and method_parts:
                title = " — ".join([base] + method_parts)
            else:
                title = base

        elif op == "Lithography":
            if mat_index < 0 or not self.is_photoresist_name(material_name):
                QMessageBox.warning(self, "Select a photoresist", "Choose a material from the Photoresists category.")
                return
            windows = self.effective_lithography_windows()
            if not windows:
                QMessageBox.warning(
                    self,
                    "Add a lithography window",
                    "Add or draw at least one window before applying this step.",
                )
                return
            step_extra["windows"] = [(float(s), float(e)) for s, e in windows]
            step_extra["window_start"] = float(self.window_start_spin.value())
            step_extra["window_end"] = float(self.window_end_spin.value())
            lithography_method = str(getattr(self, "_assistant_lithography_method_override", "") or "").strip()
            if not lithography_method:
                lithography_method = self._lithography_method_for_invert(self._current_lithography_invert())
            lithography_method = chem_normalize_ascii(lithography_method)
            method_invert = self._lithography_invert_from_text(lithography_method)
            invert_lithography = self._current_lithography_invert() if method_invert is None else bool(method_invert)
            self._set_lithography_invert(invert_lithography, sync_method=True)
            step_extra["invert_lithography"] = bool(invert_lithography)
            if not lithography_method:
                lithography_method = chem_normalize_ascii(
                    (self.etchant_combo.currentData() or self.etchant_combo.currentText() or "").strip()
                )
                if lithography_method.lower().startswith("no compatible"):
                    lithography_method = ""
            execute_ui_material_operation(
                self,
                op,
                mat_index,
                windows=windows,
                tone="negative" if invert_lithography else "positive",
            )
            self.clear_litho_windows(refresh_plot=False)
            base = "Lithography and development"
            desc = []
            if lithography_method:
                step_extra["method"] = lithography_method
                desc.append(lithography_method)
            if material_name:
                desc.append(material_name)
            if expo_str:
                desc.append(f"Exposure {expo_str}")
            if temp_str:
                desc.append(temp_str)
            if self.include_method_in_step_titles and desc:
                title = " — ".join([base] + desc)
            else:
                title = base
            material_for_recent = material_name

        geometry_ops = {
            "Deposit",
            "Spincoating",
            "Bond",
            "Etch",
            "Strip material",
            "Lift-off",
            "Bake/Annealing",
            "Lithography",
        }
        row_heights_after = (
            self.xsec.copy_row_heights()
            if hasattr(self.xsec, "copy_row_heights")
            else None
        )
        same_geometry = np.array_equal(grid_before, self.xsec.grid)
        if row_heights_before is not None or row_heights_after is not None:
            same_geometry = (
                same_geometry
                and row_heights_before is not None
                and row_heights_after is not None
                and len(row_heights_before) == len(row_heights_after)
                and np.allclose(row_heights_before, row_heights_after, atol=1e-12, rtol=0.0)
            )
        if op in geometry_ops and same_geometry and not force_log_no_change:
            if hasattr(self, "statusBar"):
                self.statusBar().showMessage(
                    f"{operation_label(op)} made no change to the current stack.",
                    4000,
                )
            return

        if material_for_recent:
            self._add_recent_material(material_for_recent)

        if title:
            if op in {"Deposit", "Spincoating", "Bond"}:
                step_extra["thickness_um"] = float(thick_um)
            elif op == "Etch":
                step_extra["depth_um"] = float(depth_um)

            payload = dict(step_extra)
            if step_metadata:
                payload.update(step_metadata)
            self.record_process_step(title, payload)

    # ---------- Dynamic UI for Operations panel ----------
    def on_op_list_row_changed(self, row: int):
        if row < 0:
            return
        if self.op_combo.currentIndex() != row:
            self.op_combo.setCurrentIndex(row)

    def sync_op_list_with_combo(self, index: int):
        if not hasattr(self, "op_list_widget"):
            return
        block = self.op_list_widget.blockSignals(True)
        try:
            self.op_list_widget.setCurrentRow(index)
        finally:
            self.op_list_widget.blockSignals(block)

    def update_op_controls(self):
        op = self.op_combo.currentText()
        prev_op = self._last_op_name_for_view
        op_changed = (prev_op != op)
        prev_material = self.current_operation_material_name() if hasattr(self, "material_combo") else ""

        # Rebuild selectable material list per operation restrictions.
        self.refresh_operation_material_combo(preferred_name=prev_material)

        self.op_list_widget.setVisible(True)
        self.op_combo.setVisible(False)
        self.material_row_widget.setVisible(False)
        self.dimension_row_widget.setVisible(False)
        if hasattr(self, "bond_target_row_widget"):
            self.bond_target_row_widget.setVisible(False)
        if hasattr(self, "lithography_invert_row_widget"):
            self.lithography_invert_row_widget.setVisible(False)
        self.material_combo.setEnabled(False)

        for label, widget in [
            (self.thickness_label, self.thick_spin),
            (self.etch_depth_label, self.depth_spin),
        ]:
            label.setVisible(False)
            widget.setVisible(False)
            widget.setEnabled(False)

        if hasattr(self, "windowing_group"):
            self.windowing_group.setVisible(op == "Lithography")

        # Default Temperature field semantics for non-deposit operations.
        self.update_deposit_method_ui("")

        if op == "Deposit":
            self.material_row_widget.setVisible(True)
            self.dimension_row_widget.setVisible(True)
            self.material_combo.setEnabled(True)

            self.thickness_label.setVisible(True)
            self.thick_spin.setVisible(True)
            self.thick_spin.setEnabled(True)

            self.update_deposit_methods_for_material()
            self.update_deposit_method_ui(self.deposit_method_combo.currentText())

        elif op == "Spincoating":
            self.material_row_widget.setVisible(True)
            self.dimension_row_widget.setVisible(True)
            self.material_combo.setEnabled(True)

            self.thickness_label.setVisible(True)
            self.thick_spin.setVisible(True)
            self.thick_spin.setEnabled(True)

            # Spincoating uses speed (rpm) instead of temperature in the Method panel.
            self.update_deposit_method_ui("Spincoating")

        elif op == "Bond":
            self.material_row_widget.setVisible(True)
            self.dimension_row_widget.setVisible(True)
            if hasattr(self, "bond_target_row_widget"):
                self.bond_target_row_widget.setVisible(True)
            self.material_combo.setEnabled(True)

            self.thickness_label.setVisible(True)
            self.thick_spin.setVisible(True)
            self.thick_spin.setEnabled(True)

        elif op == "Etch":
            self.material_row_widget.setVisible(True)
            self.dimension_row_widget.setVisible(True)
            self.material_combo.setEnabled(True)

            self.etch_depth_label.setVisible(True)
            self.depth_spin.setVisible(True)
            self.depth_spin.setEnabled(True)

        elif op in ("Strip material", "Lift-off", "Clean / Surface prep", "Bake/Annealing", "Lithography"):
            self.material_row_widget.setVisible(True)
            self.material_combo.setEnabled(True)
            if op == "Lithography" and hasattr(self, "lithography_invert_row_widget"):
                self.lithography_invert_row_widget.setVisible(True)

        self._last_op_name_for_view = op
        if op_changed and self.xsec is not None:
            self.dragging_litho_window = False
            self.litho_drag_start_frac = None
            self.litho_drag_current_frac = None
            self._clear_litho_preview_artists(redraw=False)
            self.update_plot(self.current_title or "")

        self.update_etchant_options()
        self.update_deposit_methods_for_material()
        if op_changed:
            self.load_default_method_for_current_operation(notify_when_missing=False)
        if op == "Lithography":
            self._sync_lithography_invert_from_current_method()
            self._sync_lithography_invert_from_current_material()

    def update_deposit_method_ui(self, method_text: str):
        """Adjust Temperature/RPM field in Deposit depending on the selected method."""
        method = (method_text or "").lower().strip()
        if method.startswith("spin"):
            self.temp_label.setText("Speed (rpm):")
            self.temp_spin.setSuffix(" rpm")
            self.temp_spin.setDecimals(0)
            self.temp_spin.setRange(100.0, 10000.0)
            if self.temp_spin.value() < 100.0 or self.temp_spin.value() > 10000.0:
                self.temp_spin.setValue(3000.0)
        else:
            self.temp_label.setText("Temperature (°C):")
            self.temp_spin.setSuffix(" °C")
            self.temp_spin.setDecimals(1)
            self.temp_spin.setRange(-40.0, 400.0)

    def _update_builtin_parameter_visibility(self):
        """
        Show or hide the built-in Temperature/Time fields in the Method dialog
        based on the current parameter_classes list.
        """
        # If the Method dialog has not been created yet, these widgets are still
        # top-level and toggling their visibility would cause small stray windows
        # to appear (e.g. when loading a file). In that case, do nothing.
        if self.op_params_dialog is None:
            return

        names = {
            (entry.get("name") or "").strip().lower()
            for entry in getattr(self, "parameter_classes", [])
            if entry.get("name")
        }
        show_temp = "temperature" in names
        show_time = "time" in names

        if hasattr(self, "temp_label") and hasattr(self, "temp_spin"):
            self.temp_label.setVisible(show_temp)
            self.temp_spin.setVisible(show_temp)

        if hasattr(self, "time_label") and hasattr(self, "exposure_spin"):
            self.time_label.setVisible(show_time)
            self.exposure_spin.setVisible(show_time)

    # ---------- Operation parameters floating dialog ----------
    def open_operation_parameters_dialog(self):
        op = self.op_combo.currentText()

        if self.op_params_dialog is None:
            dlg = QDialog(self)
            dlg.setWindowTitle("Step settings")
            dlg.setModal(False)
            self.op_params_dialog = dlg

            layout = QVBoxLayout(dlg)
            tabs = QTabWidget()
            self.op_params_tabs = tabs
            layout.addWidget(tabs)

            # Tab 1: parameters for current operation
            params_tab = QWidget()
            params_layout = QFormLayout()
            params_tab.setLayout(params_layout)
            self.op_params_method_tab_index = tabs.addTab(params_tab, "Method")

            # Operation + material info
            self.op_params_operation_label = QLabel("")
            self.op_params_material_label = QLabel("")
            params_layout.addRow("Operation:", self.op_params_operation_label)
            params_layout.addRow("Material:", self.op_params_material_label)

            # Method rows (we reuse the existing combos)
            self.op_params_deposit_label = QLabel("Deposition method:")
            self.op_params_etchant_label = QLabel("Etchant / method:")

            params_layout.addRow(self.op_params_deposit_label, self.deposit_method_combo)
            params_layout.addRow(self.op_params_etchant_label, self.etchant_combo)

            # Etch type (anisotropic / isotropic)
            self.op_params_isotropic_label = QLabel("Etch type:")
            params_layout.addRow(self.op_params_isotropic_label, self.isotropic_etch_checkbox)

            # Custom and default parameters area (Temperature, Time, custom)
            self.custom_params_container = QWidget()
            self.custom_params_layout = QGridLayout()
            self.custom_params_container.setLayout(self.custom_params_layout)
            params_layout.addRow(self.custom_params_container)

            # Method defaults (per operation)
            defaults_widget = QWidget()
            defaults_layout = QHBoxLayout(defaults_widget)
            defaults_layout.setContentsMargins(0, 0, 0, 0)
            defaults_layout.setSpacing(6)
            self.save_method_default_btn = QPushButton("Set default")
            self.load_method_default_btn = QPushButton("Load default")
            defaults_layout.addWidget(self.save_method_default_btn)
            defaults_layout.addWidget(self.load_method_default_btn)
            defaults_layout.addStretch(1)
            params_layout.addRow("Defaults:", defaults_widget)

            # Recipes (named method presets)
            recipe_save_widget = QWidget()
            recipe_save_layout = QHBoxLayout(recipe_save_widget)
            recipe_save_layout.setContentsMargins(0, 0, 0, 0)
            recipe_save_layout.setSpacing(6)
            self.method_recipe_name_edit = QLineEdit()
            self.method_recipe_name_edit.setPlaceholderText("Recipe name")
            self.save_method_recipe_btn = QPushButton("Save")
            recipe_save_layout.addWidget(self.method_recipe_name_edit, 1)
            recipe_save_layout.addWidget(self.save_method_recipe_btn)
            params_layout.addRow("Recipe:", recipe_save_widget)

            recipe_load_widget = QWidget()
            recipe_load_layout = QHBoxLayout(recipe_load_widget)
            recipe_load_layout.setContentsMargins(0, 0, 0, 0)
            recipe_load_layout.setSpacing(6)
            self.method_recipe_combo = QComboBox()
            self.load_method_recipe_btn = QPushButton("Load")
            self.delete_method_recipe_btn = QPushButton("Delete")
            recipe_load_layout.addWidget(self.method_recipe_combo, 1)
            recipe_load_layout.addWidget(self.load_method_recipe_btn)
            recipe_load_layout.addWidget(self.delete_method_recipe_btn)
            params_layout.addRow("Saved recipes:", recipe_load_widget)

            # Tab 2: parameter classes
            classes_tab = QWidget()
            classes_layout = QVBoxLayout()
            classes_tab.setLayout(classes_layout)
            self.op_params_parameters_tab_index = tabs.addTab(classes_tab, "Parameters")
            tabs.currentChanged.connect(self.on_op_params_tab_changed)

            info = QLabel("Add the parameters available for each method.")
            info.setWordWrap(True)
            classes_layout.addWidget(info)

            self.param_classes_list = QListWidget()
            classes_layout.addWidget(self.param_classes_list)

            form = QHBoxLayout()
            self.param_class_name_edit = QLineEdit()
            self.param_class_name_edit.setPlaceholderText("Parameter name")
            self.param_class_unit_edit = QLineEdit()
            self.param_class_unit_edit.setPlaceholderText("Unit (e.g., mbar)")
            add_class_btn = QPushButton("Add")
            remove_class_btn = QPushButton("Remove")
            form.addWidget(self.param_class_name_edit)
            form.addWidget(self.param_class_unit_edit)
            form.addWidget(add_class_btn)
            form.addWidget(remove_class_btn)
            classes_layout.addLayout(form)

            # Persist the current parameter list as the application default.
            self.save_defaults_btn = QPushButton("Save parameter defaults")
            classes_layout.addWidget(self.save_defaults_btn)

            add_class_btn.clicked.connect(self.add_parameter_class)
            remove_class_btn.clicked.connect(self.remove_selected_parameter_class)
            self.save_defaults_btn.clicked.connect(self.save_current_parameter_defaults)
            self.save_method_default_btn.clicked.connect(self.save_default_method_for_current_operation)
            self.load_method_default_btn.clicked.connect(
                lambda: self.load_default_method_for_current_operation(True)
            )
            self.save_method_recipe_btn.clicked.connect(self.save_method_recipe_from_dialog)
            self.load_method_recipe_btn.clicked.connect(self.load_selected_method_recipe)
            self.delete_method_recipe_btn.clicked.connect(self.delete_selected_method_recipe)

            btn_box = QDialogButtonBox(QDialogButtonBox.Close)
            btn_box.rejected.connect(dlg.close)
            layout.addWidget(btn_box)

            # Initialize parameter classes UI for any existing classes
            self.refresh_parameter_classes_ui()
            self._sync_op_params_expert_state()

        # Update dialog contents for current operation/material
        self.op_params_operation_label.setText(operation_label(op))
        self.op_params_material_label.setText(self.current_operation_material_name() or "—")

        # Configure which method row is active
        if op == "Deposit":
            self.op_params_deposit_label.show()
            self.deposit_method_combo.show()
            self.op_params_etchant_label.hide()
            self.etchant_combo.hide()
            self.op_params_isotropic_label.hide()
            self.isotropic_etch_checkbox.hide()
            self.update_deposit_methods_for_material()
            self.update_deposit_method_ui(self.deposit_method_combo.currentText())
        elif op == "Spincoating":
            self.op_params_deposit_label.hide()
            self.deposit_method_combo.hide()
            self.op_params_etchant_label.hide()
            self.etchant_combo.hide()
            self.op_params_isotropic_label.hide()
            self.isotropic_etch_checkbox.hide()
            self.update_deposit_method_ui("Spincoating")
        else:
            self.op_params_deposit_label.hide()
            self.deposit_method_combo.hide()

            if op in ("Etch", "Bond", "Strip material", "Lift-off", "Clean / Surface prep"):
                method_labels = {
                    "Etch": "Etchant:",
                    "Bond": "Bond method:",
                    "Strip material": "Strip method:",
                    "Lift-off": "Lift-off method:",
                    "Clean / Surface prep": "Cleaning method:",
                }
                self.op_params_etchant_label.setText(method_labels[op])
                self.op_params_etchant_label.show()
                self.etchant_combo.show()
                self.update_etchant_options()

                if op == "Etch":
                    self.op_params_isotropic_label.show()
                    self.isotropic_etch_checkbox.show()
                else:
                    self.op_params_isotropic_label.hide()
                    self.isotropic_etch_checkbox.hide()
            elif op == "Lithography":
                self.op_params_etchant_label.setText("Lithography method:")
                self.op_params_etchant_label.show()
                self.etchant_combo.show()
                self.update_etchant_options()
                self.op_params_isotropic_label.hide()
                self.isotropic_etch_checkbox.hide()
            else:
                self.op_params_etchant_label.hide()
                self.etchant_combo.hide()
                self.op_params_isotropic_label.hide()
                self.isotropic_etch_checkbox.hide()

        # Temperature/time availability controlled by parameter classes
        self._update_builtin_parameter_visibility()

        # Ensure parameters grid (default + custom) reflects current parameter classes
        self.refresh_parameter_classes_ui()
        self.refresh_method_recipe_combo()
        self._sync_op_params_expert_state()

        self.op_params_dialog.show()
        self.op_params_dialog.raise_()
        self.op_params_dialog.activateWindow()

    def _sync_op_params_expert_state(self):
        if not hasattr(self, "op_params_tabs"):
            return
        params_idx = int(getattr(self, "op_params_parameters_tab_index", -1))
        if params_idx < 0:
            return
        is_expert = bool(getattr(self, "is_expert_user", False))
        tooltip = "Manage parameter fields" if is_expert else "Requires Expert mode"
        self.op_params_tabs.setTabToolTip(params_idx, tooltip)
        if not is_expert and self.op_params_tabs.currentIndex() == params_idx:
            method_idx = int(getattr(self, "op_params_method_tab_index", 0))
            self._switching_op_params_tab = True
            try:
                self.op_params_tabs.setCurrentIndex(max(0, method_idx))
            finally:
                self._switching_op_params_tab = False

    def on_op_params_tab_changed(self, index: int):
        params_idx = int(getattr(self, "op_params_parameters_tab_index", -1))
        if index != params_idx or getattr(self, "_switching_op_params_tab", False):
            return
        if getattr(self, "is_expert_user", False):
            return
        if self.ensure_expert_user_mode("Parameters"):
            self._sync_op_params_expert_state()
            return
        method_idx = int(getattr(self, "op_params_method_tab_index", 0))
        self._switching_op_params_tab = True
        try:
            self.op_params_tabs.setCurrentIndex(max(0, method_idx))
        finally:
            self._switching_op_params_tab = False

    def add_parameter_class(self):
        if not self.ensure_expert_user_mode("Parameters"):
            return
        name = self.param_class_name_edit.text().strip()
        unit = self.param_class_unit_edit.text().strip()

        if not name:
            QMessageBox.warning(self, "Name required", "Enter a parameter name.")
            return

        entry = {"name": name, "unit": unit}
        self.parameter_classes.append(entry)

        self.refresh_parameter_classes_ui()

        self.param_class_name_edit.clear()
        self.param_class_unit_edit.clear()

    def remove_selected_parameter_class(self):
        """Remove the selected parameter."""
        if not self.ensure_expert_user_mode("Parameters"):
            return
        if not hasattr(self, "param_classes_list"):
            return
        row = self.param_classes_list.currentRow()
        if row < 0 or row >= len(self.parameter_classes):
            QMessageBox.warning(self, "Select a parameter", "Choose the parameter to remove.")
            return

        reply = QMessageBox.question(
            self,
            "Remove parameter?",
            "This parameter will no longer appear in step settings.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self.parameter_classes.pop(row)
        self.refresh_parameter_classes_ui()

    def rebuild_custom_params_widgets(self):
        """
        Rebuild the parameter fields grid in the Method tab.
        Includes both default parameters (Temperature, Time) and custom ones,
        distributed on two columns with label + box together.
        """
        if not hasattr(self, "custom_params_layout"):
            return

        # Clear existing widgets from the grid, but keep the built-in widgets alive
        keep_widgets = {
            getattr(self, "temp_label", None),
            getattr(self, "temp_spin", None),
            getattr(self, "time_label", None),
            getattr(self, "exposure_spin", None),
        }

        while self.custom_params_layout.count():
            item = self.custom_params_layout.takeAt(0)
            w = item.widget()
            if w is not None and w not in keep_widgets:
                w.deleteLater()

        self.custom_param_widgets = {}

        # Build entries from parameter_classes
        entries = []
        for entry in getattr(self, "parameter_classes", []):
            name = entry.get("name", "")
            unit = entry.get("unit", "")
            if not name:
                continue
            entries.append((name, unit))

        if not entries:
            if hasattr(self, "custom_params_container"):
                self.custom_params_container.setVisible(False)
            return

        row = 0
        col_slot = 0  # 0 = left pair (cols 0-1), 1 = right pair (cols 2-3)

        for name, unit in entries:
            lname = name.strip().lower()

            if lname == "temperature":
                label = self.temp_label
                spin = self.temp_spin
            elif lname == "time":
                label = self.time_label
                spin = self.exposure_spin
            else:
                label_text = f"{name} ({unit})" if unit else name
                label = QLabel(label_text)
                spin = QDoubleSpinBox()
                spin.setDecimals(2)
                spin.setRange(-1e9, 1e9)
                if unit:
                    spin.setSuffix(" " + unit)
                self.custom_param_widgets[name] = spin

            # Align label on the right
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            # Compact width for spinbox; for built-ins we just ensure a decent width
            fm = spin.fontMetrics()
            sample_text = "-9999.99"
            if unit:
                sample_text += " " + unit
            width = fm.horizontalAdvance(sample_text) + 16
            spin.setFixedWidth(width)
            spin.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

            base_col = 0 if col_slot == 0 else 2
            self.custom_params_layout.addWidget(label, row, base_col)
            self.custom_params_layout.addWidget(spin, row, base_col + 1)

            # Two parameters per row
            if col_slot == 0:
                col_slot = 1
            else:
                col_slot = 0
                row += 1

        # Two label+spin columns
        self.custom_params_layout.setColumnStretch(0, 1)
        self.custom_params_layout.setColumnStretch(1, 0)
        self.custom_params_layout.setColumnStretch(2, 1)
        self.custom_params_layout.setColumnStretch(3, 0)
        self.custom_params_layout.setHorizontalSpacing(12)
        self.custom_params_layout.setVerticalSpacing(4)

        if hasattr(self, "custom_params_container"):
            self.custom_params_container.setVisible(True)

    def refresh_parameter_classes_ui(self):
        """Refresh the Parameter tab and the parameter grid area."""
        if hasattr(self, "param_classes_list"):
            self.param_classes_list.clear()
            for entry in self.parameter_classes:
                name = entry.get("name", "")
                unit = entry.get("unit", "")
                if not name:
                    continue
                display = f"{name} ({unit})" if unit else name
                self.param_classes_list.addItem(display)

        self.rebuild_custom_params_widgets()
        # Update built-in Temperature/Time visibility when classes change
        self._update_builtin_parameter_visibility()
