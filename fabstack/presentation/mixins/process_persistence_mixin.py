"""Process save/load and lifecycle actions for MainWindow."""

import logging
import os

from PyQt5.QtWidgets import QFileDialog, QMessageBox

from fabstack.config.visualization import (
    FEATURE_EMPHASIS_DEFAULT,
    clamp_feature_emphasis,
    clamp_visual_scale,
)
from fabstack.domain.cross_section import CrossSection
from fabstack.persistence.process_file import save_process_file

LOGGER = logging.getLogger(__name__)


def _clamp_visual_scale(value):
    return clamp_visual_scale(value)


def _clamp_feature_scale(value):
    return clamp_feature_emphasis(value)


class ProcessPersistenceMixin:
    # ---------- Process save/load ----------
    def _make_process_dict(self):
        data = {
            "format_version": 1,
            "material_names": self.material_names,
            "material_colors": self.material_colors,
            "void_index": self.VOID_INDEX,
            "substrate_name": self.substrate_combo.currentText(),
            "substrate_thickness": self.substrate_thick_spin.value(),
            "history": self.history,
            "process_steps": list(getattr(self, "process_steps", [])) if getattr(self, "process_steps_replay_enabled", False) else None,
            "text_pos": self.text_pos,
            "text_fontsize": self.text_fontsize,
            "material_fontsize": self.material_fontsize,
            "title_font_family": self.title_font_family,
            "title_color": self.title_color,
            "material_font_family": self.material_font_family,
            "material_color": self.material_color,
            "material_legend_pos": tuple(getattr(self, "material_legend_pos", (0.8, 0.0))),
            "litho_windows": self.litho_windows,
            "window_start": self.window_start_spin.value(),
            "window_end": self.window_end_spin.value(),
            "deleted_materials": list(self.deleted_materials),
            "recent_materials": self.recent_materials,
            "hidden_material_categories": list(self.hidden_material_categories),
            "parameter_classes": self.parameter_classes,
            "include_method_in_step_titles": self.include_method_in_step_titles,
            "include_logo_in_exports": bool(getattr(self, "include_logo_in_exports", True)),
            # Persist visual scale settings.
            "visual_scale_factor": _clamp_visual_scale(getattr(self, "visual_scale_factor", 1.0)),
            "feature_scale_factor": _clamp_feature_scale(
                getattr(self, "feature_scale_factor", FEATURE_EMPHASIS_DEFAULT)
            ),
            "visualisation_mode": (
                self._normalised_visualisation_mode()
                if hasattr(self, "_normalised_visualisation_mode")
                else "dynamic"
            ),
            "visual_thickness_overrides": (
                dict(getattr(self, "visual_thickness_overrides", {}) or {})
                if (
                    self._normalised_visualisation_mode()
                    if hasattr(self, "_normalised_visualisation_mode")
                    else "dynamic"
                ) in {"pending", "finalised"}
                else {}
            ),
            "visual_thickness_step_overrides": {},
            "llm_chat_messages": (
                self._assistant_chat_for_save()
                if hasattr(self, "_assistant_chat_for_save")
                else []
            ),
        }
        return data

    def _load_process_dict(self, data: dict):
        self.clear_bond_source_selection()
        if hasattr(self, "_set_assistant_chat_messages"):
            self._set_assistant_chat_messages(data.get("llm_chat_messages", []))
        self.material_names = data.get("material_names", ["Void", "Silicon (Si)"])
        self.material_colors = data.get("material_colors", ["#ffffff", "#b0b0b0"])
        self.VOID_INDEX = data.get("void_index", 0)
        self.text_pos = tuple(data.get("text_pos", (0.5, -0.08)))
        self.text_fontsize = data.get("text_fontsize", 10)
        self.material_fontsize = data.get("material_fontsize", 8)
        default_font_family = self.font().family() or "Sans Serif"
        self.title_font_family = str(data.get("title_font_family", default_font_family))
        self.title_color = self._normalize_color_hex(str(data.get("title_color", "#000000")))
        self.material_font_family = str(data.get("material_font_family", default_font_family))
        self.material_color = self._normalize_color_hex(str(data.get("material_color", "#000000")))
        legend_pos = data.get("material_legend_pos", (0.8, 0.0))
        try:
            self.material_legend_pos = (float(legend_pos[0]), float(legend_pos[1]))
        except (TypeError, ValueError, IndexError):
            self.material_legend_pos = (0.8, 0.0)
        self.material_legend_artist = None
        self.dragging_material_legend = False
        self.material_legend_drag_offset = (0.0, 0.0)

        # Restore visual scale.
        self.visual_scale_factor = _clamp_visual_scale(data.get("visual_scale_factor", 1.0))
        raw_mode = data.get("visualisation_mode", None)
        mode = str(raw_mode or "dynamic").strip().lower()
        mode = mode if mode in {"dynamic", "pending", "finalised"} else "dynamic"
        self.visualisation_mode = mode
        if "feature_scale_factor" in data:
            migrated_feature_scale = data.get("feature_scale_factor", FEATURE_EMPHASIS_DEFAULT)
        else:
            # Old frozen files must reproduce their prior 100% adaptive map;
            # old dynamic files adopt the new balanced default.
            migrated_feature_scale = 1.0 if mode in {"pending", "finalised"} else FEATURE_EMPHASIS_DEFAULT
        self.feature_scale_factor = _clamp_feature_scale(migrated_feature_scale)
        if hasattr(self, "_sync_feature_scale_controls_from_state"):
            self._sync_feature_scale_controls_from_state()
        if raw_mode is None or mode == "dynamic":
            self.visual_thickness_overrides = {}
        else:
            visual_overrides = data.get("visual_thickness_overrides", {})
            self.visual_thickness_overrides = (
                dict(visual_overrides)
                if isinstance(visual_overrides, dict)
                else {}
            )
        self.visual_thickness_step_overrides = {}

        self.deleted_materials = set(data.get("deleted_materials", []))
        self.recent_materials = data.get("recent_materials", [])
        self.hidden_material_categories = set(data.get("hidden_material_categories", []))
        if hasattr(self, "_ensure_unique_material_colors"):
            self._ensure_unique_material_colors()

        pc = data.get("parameter_classes", None)
        if isinstance(pc, list):
            self.parameter_classes = pc
        else:
            self.parameter_classes = []
        # Backwards compatibility: if no parameter classes were stored, seed defaults.
        if not self.parameter_classes:
            self.parameter_classes = self.load_default_parameter_classes()

        # Restore the project-level step-title preference.
        self.include_method_in_step_titles = data.get(
            "include_method_in_step_titles", True
        )
        self.include_logo_in_exports = bool(data.get("include_logo_in_exports", True))

        self.text_size_spin.setValue(self.text_fontsize)
        self.material_font_size_spin.setValue(self.material_fontsize)
        self._sync_text_options_widgets_from_state()

        if hasattr(self, "_begin_substrate_control_update"):
            self._begin_substrate_control_update()
        try:
            self.substrate_combo.clear()
            for name in self.material_names:
                if name == "Void":
                    continue
                if name in self.deleted_materials:
                    continue
                self.substrate_combo.addItem(name)

            substrate_name = data.get("substrate_name", "Silicon (Si)")
            idx = self.substrate_combo.findText(substrate_name)
            if idx < 0 and self.substrate_combo.count() > 0:
                idx = 0
                substrate_name = self.substrate_combo.itemText(0)
            self.substrate_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self.substrate_thick_spin.setValue(data.get("substrate_thickness", 200))
        finally:
            if hasattr(self, "_end_substrate_control_update"):
                self._end_substrate_control_update()

        self.refresh_operation_material_combo()

        loaded_history = data.get("history", [])
        steps = data.get("process_steps", None)
        if isinstance(steps, list):
            self.process_steps_replay_enabled = True
            self.process_steps = [dict(step) for step in steps]
            self.history = []
        else:
            self.process_steps_replay_enabled = False
            self.process_steps = []
            self.history = loaded_history if isinstance(loaded_history, list) else []
        self.redo_stack = []
        self.history_list.clear()
        for st in self.history:
            self.history_list.addItem(st.get("title", ""))

        self.litho_windows = data.get("litho_windows", [])
        self.window_list.clear()
        for s, e in self.litho_windows:
            self.window_list.addItem(f"{s:.2f} – {e:.2f}")
        if hasattr(self, "window_count_label"):
            self.window_count_label.setText(f"Windows: {len(self.litho_windows)}")

        self.window_start_spin.setValue(data.get("window_start", 0.35))
        self.window_end_spin.setValue(data.get("window_end", 0.65))

        # Restore parameter classes list and parameter grid UI in dialog if it already exists
        self.refresh_parameter_classes_ui()

        # Keep Edit menu checkbox in sync with loaded setting
        if hasattr(self, "act_toggle_method_in_titles"):
            self.act_toggle_method_in_titles.setChecked(
                self.include_method_in_step_titles
            )
        if hasattr(self, "act_include_logo_in_exports"):
            self.act_include_logo_in_exports.setChecked(
                self.include_logo_in_exports
            )

        if self.process_steps_replay_enabled:
            self.rebuild_process_from_steps(update_view=True)
        elif self.history:
            last = self.history[-1]
            grid = last["grid"]
            h, w = grid.shape
            substrate_index = self.material_index(substrate_name)
            if substrate_index < 0:
                substrate_index = 1
            sub_thick_px = self.um_to_process_px(self.substrate_thick_spin.value(), mode="ceil", minimum=1)

            self.xsec = CrossSection(
                width_px=w,
                height_px=h,
                substrate_index=substrate_index,
                void_index=self.VOID_INDEX,
                substrate_thickness_px=sub_thick_px,
                substrate_thickness_um=float(self.substrate_thick_spin.value()),
                top_headroom_um=max(
                    0.0,
                    float(self.process_grid_height_um_for_substrate(self.substrate_thick_spin.value()))
                    - float(self.substrate_thick_spin.value()),
                ),
            )
            if hasattr(self, "_restore_xsec_snapshot"):
                self._restore_xsec_snapshot(last)
            else:
                self.xsec.restore_state(
                    grid,
                    last.get("row_heights_um"),
                    last.get("visual_geometry"),
                    last.get("structural_role_grid"),
                    last.get("structural_bodies"),
                    last.get("planarizer_role_grid"),
                    last.get("planarizer_roles"),
                )
            self.update_plot(last["title"])
            self.history_list.setCurrentRow(len(self.history) - 1)
        else:
            self.reset_wafer()

        self.refresh_materials_list()
        self.refresh_recent_materials_list()
        self.update_etchant_options()
        self.update_deposit_methods_for_material()
        # Ensure etch mode is consistent with the current etchant
        self.on_etchant_changed()
        if hasattr(self, "_sync_canvas_scale_overlay_from_state"):
            self._sync_canvas_scale_overlay_from_state()
        if hasattr(self, "_update_visualisation_controls_state"):
            self._update_visualisation_controls_state()
        # Re-apply expert gating after loading
        self.update_expert_user_controls()

    def _save_process_to_path(self, path: str):
        data = self._make_process_dict()
        try:
            save_process_file(path, data)
            self.current_process_path = path
            if 0 <= self.active_project_tab_index < len(self.project_sessions):
                session = self.project_sessions[self.active_project_tab_index]
                live_data = self._deepcopy_process_data(data)
                session["data"] = live_data
                session["path"] = path
                session["name"] = os.path.basename(path)
                session["saved_hash"] = self._process_data_hash(live_data)
                session["dirty"] = False
                session["unsaved_prompt_shown"] = False
                self._update_project_tab_ui(self.active_project_tab_index)
            self._refresh_window_title_from_active_tab()
            self._hide_unsaved_save_prompt()
            if hasattr(self, "statusBar"):
                self.statusBar().showMessage(
                    f"Saved {os.path.basename(path)}.",
                    3500,
                )
            return True
        except Exception:
            LOGGER.exception("Could not save process to %s", path)
            QMessageBox.warning(
                self,
                "Couldn’t save process",
                "Check that the destination is writable and has enough free space, then try again.",
            )
            return False

    def _maybe_save_before_discard(self) -> bool:
        """
        Ask the user whether to save before discarding the current project.

        Returns True if it is safe to continue (discard or successfully saved),
        False if the user cancels.
        """
        self._snapshot_active_tab_state()
        is_dirty = False
        if 0 <= self.active_project_tab_index < len(self.project_sessions):
            is_dirty = bool(self.project_sessions[self.active_project_tab_index].get("dirty", False))
        else:
            is_dirty = len(self.history) > 1
        if not is_dirty:
            return True

        tab_name = ""
        idx = self.project_tabs.currentIndex() if hasattr(self, "project_tabs") else -1
        if 0 <= idx < self.project_tabs.count():
            tab_name = self.project_tabs.tabText(idx)

        prompt = "Save changes before closing?"
        if tab_name:
            prompt = f"Save changes to “{tab_name}” before closing?"

        reply = QMessageBox.question(
            self,
            "Save changes?",
            prompt,
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )

        if reply == QMessageBox.Save:
            # Existing projects save in place; untitled projects open Save As.
            ok = self.save_process()
            return ok
        elif reply == QMessageBox.Discard:
            return True
        else:
            return False

    # ---------- Public process actions ----------
    def new_process(self):
        self._open_session_in_new_tab(self._default_new_process_data(), path=None)

    def save_process(self):
        if not self.history:
            QMessageBox.information(self, "Nothing to save", "Add a process step first.")
            return False
        if not self.current_process_path:
            return self.save_process_as()
        return self._save_process_to_path(self.current_process_path)

    def save_process_as(self) -> bool:
        if not self.history:
            QMessageBox.information(self, "Nothing to save", "Add a process step first.")
            return False
        fname, _ = QFileDialog.getSaveFileName(
            self,
            "Save process as",
            "",
            "FabStack process (*.fab);;All Files (*)"
        )
        if not fname:
            return False
        if not fname.lower().endswith(".fab"):
            fname += ".fab"
        return self._save_process_to_path(fname)

    def load_process(self):
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Open process",
            "",
            "FabStack process (*.fab);;All Files (*)"
        )
        if not fname:
            return
        try:
            data = self._read_process_file_data(fname)
            self._open_session_in_new_tab(data, path=fname)
        except Exception:
            LOGGER.exception("Could not open process file %s", fname)
            QMessageBox.warning(
                self,
                "Couldn’t open process",
                "The file is unavailable, damaged, or not a supported FabStack process. Choose another file.",
            )

    # ---------- Close event: ask to save before closing ----------
    def closeEvent(self, event):
        if self._maybe_save_all_tabs_before_close():
            event.accept()
        else:
            event.ignore()
