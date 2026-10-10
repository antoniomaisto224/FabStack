"""Materials and library behavior for MainWindow."""

import json
import logging
import os
import shutil

import numpy as np
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QBrush, QColor, QFont, QPalette
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QMessageBox,
    QVBoxLayout,
)
from PyQt5.QtGui import QDesktopServices

from fabstack.materials.chemistry import chem_normalize_ascii, chem_subs_unicode
from fabstack.materials.schema import validate_material_library
from fabstack.presentation.dialogs.material_library import LibraryManagerDialog
from fabstack.terminology import operation_label

LOGGER = logging.getLogger(__name__)


class MaterialsLibraryMixin:
    def _confirm_material_action(
        self,
        title: str,
        message: str,
        action_label: str,
        *,
        default_to_action: bool = False,
    ) -> bool:
        dialog = QMessageBox(
            QMessageBox.Question,
            title,
            message,
            QMessageBox.Yes | QMessageBox.Cancel,
            self,
        )
        action_button = dialog.button(QMessageBox.Yes)
        action_button.setText(action_label)
        cancel_button = dialog.button(QMessageBox.Cancel)
        dialog.setDefaultButton(action_button if default_to_action else cancel_button)
        return dialog.exec_() == QMessageBox.Yes

    # ---------- Materials list (current file) ----------
    def refresh_materials_list(self):
        if not hasattr(self, "materials_list_widget"):
            return
        self.materials_list_widget.clear()

        for name in self.material_names:
            if name == "Void":
                continue
            if name in self.deleted_materials:
                continue
            self.materials_list_widget.addItem(name)

    # ---------- Recent materials list ----------
    def refresh_recent_materials_list(self):
        if not hasattr(self, "recent_materials_list_widget"):
            return
        self.recent_materials_list_widget.clear()
        for name in self.recent_materials:
            if name == "Void":
                continue
            if name in self.deleted_materials:
                continue
            self.recent_materials_list_widget.addItem(name)

    def on_recent_material_clicked(self, item):
        name = item.text()
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

    def _material_color_hex(self, value: str, fallback: str = "#cccccc") -> str:
        color = QColor(str(value or ""))
        if not color.isValid():
            color = QColor(str(fallback or "#cccccc"))
        if not color.isValid():
            color = QColor("#cccccc")
        return color.name()

    def _is_disallowed_material_color(self, color) -> bool:
        if color is None or not color.isValid():
            return False
        r, g, b, _ = color.getRgb()
        return (r, g, b) == (255, 255, 255)

    def _used_material_color_hexes(self, exclude_name: str = ""):
        used = set()
        exclude = str(exclude_name or "").strip()
        for idx, name in enumerate(getattr(self, "material_names", [])):
            name = str(name or "").strip()
            if not name or name == "Void" or name == exclude:
                continue
            if idx >= len(getattr(self, "material_colors", [])):
                continue
            used.add(self._material_color_hex(self.material_colors[idx]))
        return used

    def _is_material_color_in_use(self, color_hex: str, exclude_name: str = "") -> bool:
        return self._material_color_hex(color_hex) in self._used_material_color_hexes(exclude_name=exclude_name)

    def _pick_unique_material_color(self, preferred_hex: str, used_colors: set) -> str:
        used = {self._material_color_hex(c) for c in (used_colors or set())}
        preferred = self._material_color_hex(preferred_hex)
        if preferred != "#ffffff" and preferred not in used:
            return preferred

        palette = [
            "#d4af37", "#5b8def", "#e06c75", "#56b6c2", "#98c379",
            "#c678dd", "#e5c07b", "#61afef", "#d19a66", "#4db6ac",
            "#f06292", "#7986cb", "#aed581", "#ff8a65", "#9575cd",
            "#4fc3f7", "#ba8f5d", "#7cb342", "#ec407a", "#26a69a",
            "#8d6e63", "#9ccc65", "#ffb74d", "#64b5f6", "#a1887f",
        ]
        for candidate in palette:
            color = self._material_color_hex(candidate)
            if color != "#ffffff" and color not in used:
                return color

        for i in range(1440):
            hue = (29 + i * 137) % 360
            saturation = 150 + ((i // 360) % 4) * 25
            value = 180 + ((i // 90) % 3) * 25
            color = QColor.fromHsv(hue, min(saturation, 245), min(value, 245)).name()
            if color != "#ffffff" and color not in used:
                return color
        return "#cccccc"

    def _unique_material_color(self, preferred_hex: str, exclude_name: str = "") -> str:
        return self._pick_unique_material_color(
            preferred_hex,
            self._used_material_color_hexes(exclude_name=exclude_name),
        )

    def _ensure_unique_material_colors(self):
        names = list(getattr(self, "material_names", []))
        current = list(getattr(self, "material_colors", []))
        used = set()
        colors = []
        for idx, name in enumerate(names):
            name = str(name or "").strip()
            preferred = current[idx] if idx < len(current) else "#cccccc"
            if name == "Void":
                color = "#ffffff"
            else:
                color = self._pick_unique_material_color(preferred, used)
            colors.append(color)
            used.add(color)
        self.material_colors = colors

    def _prompt_material_color(self, exclude_name: str = ""):
        """
        Prompt for a material color, rejecting pure white and duplicate active material colors.
        Returns QColor or None if user cancels.
        """
        while True:
            color = QColorDialog.getColor()
            if not color.isValid():
                return None
            if self._is_disallowed_material_color(color):
                QMessageBox.warning(
                    self,
                    "Choose another colour",
                    "White represents empty space."
                )
                continue
            if self._is_material_color_in_use(color.name(), exclude_name=exclude_name):
                QMessageBox.warning(
                    self,
                    "Choose another colour",
                    "This colour is already used by another material."
                )
                continue
            return color

    def on_material_context_menu(self, pos):
        if not hasattr(self, "materials_list_widget"):
            return
        item = self.materials_list_widget.itemAt(pos)
        if item is None:
            return
        self.materials_list_widget.setCurrentItem(item)

        menu = QMenu(self)
        act_modify = menu.addAction("Change colour…")
        act_remove = menu.addAction("Remove material")
        chosen = menu.exec_(self.materials_list_widget.mapToGlobal(pos))
        if chosen == act_modify:
            self.modify_material(item.text())
        elif chosen == act_remove:
            self.delete_material(item.text())

    def on_delete_material_button_clicked(self):
        # Prefer removing the currently selected material in the list.
        if hasattr(self, "materials_list_widget"):
            item = self.materials_list_widget.currentItem()
            if item is not None:
                self.delete_material(item.text())
                return
        # Fallback: keep existing selection dialog behavior.
        self.delete_material()

    def _add_recent_material(self, name: str):
        if not name or name == "Void":
            return
        if not hasattr(self, "recent_materials"):
            self.recent_materials = []
        if name in self.recent_materials:
            self.recent_materials.remove(name)
        self.recent_materials.insert(0, name)
        if len(self.recent_materials) > 10:
            self.recent_materials = self.recent_materials[:10]
        self.refresh_recent_materials_list()

    # ---------- Helpers for etchant/method combo ----------
    def _etchants_from_library_or_default(self):
        from_lib = [chem_normalize_ascii(e) for e in self.material_library_by_cat.get("Etchants", [])]
        if from_lib:
            return from_lib
        return [
            "KOH", "TMAH", "EDP", "HNA",
            "HF", "Buffered HF (BOE)", "Hot Phosphoric Acid",
            "Aqua Regia", "KI/I2", "Chromium Etchant (CR-7)",
            "Gold Etchant (KI/I2 or Aqua Regia)",
            "Aluminum Etchant (PAN)", "Copper Etchant (FeCl3)",
            "Nickel Etchant", "Titanium Etchant", "Piranha",
            "Acetone", "NMP", "Remover PG",
            "O2 Plasma", "Ar Plasma",
            "RIE CF4", "RIE CHF3", "RIE SF6", "RIE Cl2",
            "ICP-RIE (SF6 / O2)", "DRIE (Bosch process)", "Cl2/BCl3 Plasma"
        ]

    def _populate_etchant_combo(self, names):
        self.etchant_combo.clear()
        for base in names:
            base_ascii = chem_normalize_ascii(base)
            pretty = chem_subs_unicode(base_ascii)
            self.etchant_combo.addItem(pretty, base_ascii)

    def _lithography_methods(self):
        return [
            "Positive lithography",
            "Negative lithography",
        ]

    def _default_deposit_methods(self):
        return [
            "Thermal deposition",
            "Chemical vapor deposition",
            "Sputtering",
            "Atomic layer deposition",
        ]

    # ---------- material -> allowed etchants ----------
    def compatible_etchants(self, material_name: str):
        if not material_name:
            return []

        if material_name in self.compat_etch_by_material:
            return self.compat_etch_by_material[material_name][:]

        for cat, items in self.material_library_by_cat.items():
            if material_name in items and cat in self.compat_etch_by_category:
                return self.compat_etch_by_category[cat][:]

        name = material_name.lower()

        def has(*keys): return any(k.lower() in name for k in keys)

        if has("polysilicon") or (name == "silicon (si)") or has("silicon (si)"):
            return ["KOH", "TMAH", "EDP", "HNA", "RIE SF6", "ICP-RIE (SF6 / O2)", "DRIE (Bosch process)"]
        if has("silicon dioxide", "sio2", "glass", "fused silica", "quartz"):
            return ["HF", "Buffered HF (BOE)", "RIE CHF3", "RIE CF4"]
        if has("silicon nitride", "si3n4"):
            return ["Hot Phosphoric Acid", "RIE CHF3", "RIE CF4"]
        if has("sapphire"):
            return ["Cl2/BCl3 Plasma", "RIE Cl2"]
        if has("gold"):
            return ["Gold Etchant (KI/I2 or Aqua Regia)", "Aqua Regia"]
        if has("chromium"):
            return ["Chromium Etchant (CR-7)"]
        if has("aluminum"):
            return ["Aluminum Etchant (PAN)", "RIE Cl2", "Cl2/BCl3 Plasma"]
        if has("copper"):
            return ["Copper Etchant (FeCl3)"]
        if has("nickel"):
            return ["Nickel Etchant"]
        if has("titanium"):
            return ["Titanium Etchant", "RIE Cl2"]
        if has("platinum"):
            return ["Aqua Regia"]
        if has("tungsten"):
            return ["RIE SF6", "RIE Cl2"]
        if material_name in self.material_library_by_cat.get("Photoresists", []):
            return ["Acetone", "NMP", "Remover PG", "O2 Plasma", "Ar Plasma"]
        if material_name in self.material_library_by_cat.get("Polymers", []):
            return ["Acetone", "NMP", "O2 Plasma", "Ar Plasma"]
        return []

    def compatible_deposit_methods(self, material_name: str):
        """
        Return a list of compatible deposition methods for the given material,
        based on library mappings.
        """
        if not material_name:
            return []

        def _remove_spincoating(methods):
            out = []
            for m in methods:
                txt = str(m).strip()
                if txt.lower() == "spincoating":
                    continue
                out.append(txt)
            return out

        if material_name in self.compat_deposit_by_material:
            methods = self.compat_deposit_by_material[material_name][:]
            return _remove_spincoating(methods)

        for cat, items in self.material_library_by_cat.items():
            if material_name in items and cat in self.compat_deposit_by_category:
                methods = self.compat_deposit_by_category[cat][:]
                return _remove_spincoating(methods)

        return []

    def compatible_liftoff_methods(self, material_name: str):
        """
        Return allowed lift-off methods for the given material.
        Uses material-specific mapping first, then category mapping.
        """
        if not material_name:
            return []

        if material_name in self.compat_liftoff_by_material:
            return self.compat_liftoff_by_material[material_name][:]

        for cat, items in self.material_library_by_cat.items():
            if material_name in items and cat in self.compat_liftoff_by_category:
                return self.compat_liftoff_by_category[cat][:]

        return []

    def _is_material_restricted_for_operation(self, operation: str, material_name: str) -> bool:
        if not operation or not material_name:
            return False
        blocked = self.operation_restrictions_by_operation.get(operation, [])
        return material_name in blocked

    def _operation_material_candidates(self):
        return [
            n for n in self.material_names
            if n not in ("Void",) and n not in self.deleted_materials
        ]

    def _restricted_material_names_for_operation(self, operation: str):
        blocked_raw = self.operation_restrictions_by_operation.get(operation, [])
        if not isinstance(blocked_raw, list):
            return set()
        blocked = set()
        for name in blocked_raw:
            txt = str(name).strip()
            if txt:
                blocked.add(txt)
        return blocked

    def _material_short_name(self, material_name: str) -> str:
        txt = str(material_name or "").strip()
        if not txt.endswith(")") or "(" not in txt:
            return ""
        start = txt.rfind("(")
        short = txt[start + 1:-1].strip()
        return short

    def _operation_material_display_name(self, full_name: str) -> str:
        """
        Visible label for operation material combo:
        use short name only when full name does not fit in current combo width.
        """
        name = str(full_name or "").strip()
        if not name or not hasattr(self, "material_combo"):
            return name

        short = self._material_short_name(name)
        if not short:
            return name

        combo = self.material_combo
        combo_w = combo.width()
        if combo_w <= 0:
            combo_w = combo.minimumWidth()
        if combo_w <= 0:
            combo_w = combo.sizeHint().width()
        if combo_w <= 0:
            combo_w = 145

        # Reserve space for arrow + frame + side padding.
        available = max(40, int(combo_w) - 34)
        if combo.fontMetrics().horizontalAdvance(name) > available:
            return short
        return name

    def current_operation_material_name(self) -> str:
        """Return selected operation material full name, independent of display text."""
        if not hasattr(self, "material_combo"):
            return ""
        data = self.material_combo.currentData()
        if isinstance(data, str):
            txt = data.strip()
            if txt.startswith("__restricted__:"):
                return txt.split(":", 1)[1].strip()
            if txt.startswith("__"):
                return ""
            return txt
        return (self.material_combo.currentText() or "").strip()

    def _material_combo_index_by_full_name(self, name: str) -> int:
        if not hasattr(self, "material_combo"):
            return -1
        target = str(name or "").strip()
        if not target:
            return -1
        combo = self.material_combo
        for i in range(combo.count()):
            data = combo.itemData(i)
            if isinstance(data, str):
                txt = data.strip()
                if txt.startswith("__restricted__:"):
                    txt = txt.split(":", 1)[1].strip()
                elif txt.startswith("__"):
                    continue
                if txt == target:
                    return i
            if combo.itemText(i).strip() == target:
                return i
        return -1

    def _set_combo_item_enabled(self, combo: QComboBox, index: int, enabled: bool):
        model = combo.model()
        if model is None or not hasattr(model, "item"):
            return
        item = model.item(index)
        if item is not None:
            item.setEnabled(enabled)

    def _style_restricted_header_item(self, combo: QComboBox, index: int):
        model = combo.model()
        if model is None or not hasattr(model, "item"):
            return
        item = model.item(index)
        if item is None:
            return
        # Make the header visually secondary (smaller + darker/muted).
        font = QFont(combo.font())
        if font.pointSizeF() > 0:
            font.setPointSizeF(max(7.0, font.pointSizeF() - 1.0))
        item.setFont(font)
        muted = self.palette().color(QPalette.Disabled, QPalette.Text)
        item.setForeground(QBrush(muted))

    def refresh_operation_material_combo(self, preferred_name: str = ""):
        if not hasattr(self, "material_combo"):
            return

        op = self.op_combo.currentText() if hasattr(self, "op_combo") else ""
        all_materials = self._operation_material_candidates()
        blocked = self._restricted_material_names_for_operation(op)

        allowed = [name for name in all_materials if name not in blocked]
        restricted = [name for name in all_materials if name in blocked]

        prev = (preferred_name or self.current_operation_material_name() or "").strip()

        block = self.material_combo.blockSignals(True)
        try:
            self.material_combo.clear()

            for name in allowed:
                self.material_combo.addItem(self._operation_material_display_name(name), name)

            if getattr(self, "show_restricted_materials_in_selector", False) and restricted:
                if self.material_combo.count() > 0:
                    self.material_combo.insertSeparator(self.material_combo.count())

                header_idx = self.material_combo.count()
                self.material_combo.addItem("-----Restricted materials-----", "__restricted_header__")
                self._set_combo_item_enabled(self.material_combo, header_idx, False)
                self._style_restricted_header_item(self.material_combo, header_idx)

                for name in restricted:
                    idx = self.material_combo.count()
                    self.material_combo.addItem(
                        self._operation_material_display_name(name),
                        f"__restricted__:{name}"
                    )
                    self._set_combo_item_enabled(self.material_combo, idx, False)

            if allowed:
                chosen = 0
                if prev:
                    for i, name in enumerate(allowed):
                        if name == prev:
                            chosen = i
                            break
                self.material_combo.setCurrentIndex(chosen)
            else:
                self.material_combo.setCurrentIndex(-1)
        finally:
            self.material_combo.blockSignals(block)

    def on_toggle_show_restricted_materials(self, checked: bool):
        self.show_restricted_materials_in_selector = bool(checked)
        self.refresh_operation_material_combo()
        self.update_etchant_options()
        self.update_deposit_methods_for_material()
        if hasattr(self, "op_params_material_label"):
            self.op_params_material_label.setText(self.current_operation_material_name() or "—")

    def update_etchant_options(self):
        op = self.op_combo.currentText()
        prev = chem_normalize_ascii(
            (self.etchant_combo.currentData() or self.etchant_combo.currentText() or "").strip()
        )
        default_state = self._default_method_state_for_operation(op)
        default_etchant = chem_normalize_ascii(default_state.get("etchant", ""))

        def _select_matching_etchant(*candidates):
            for cand in candidates:
                target = chem_normalize_ascii((cand or "").strip())
                if not target:
                    continue
                for i in range(self.etchant_combo.count()):
                    item_val = self.etchant_combo.itemData(i) or self.etchant_combo.itemText(i)
                    item_ascii = chem_normalize_ascii((item_val or "").strip())
                    if item_ascii == target:
                        self.etchant_combo.setCurrentIndex(i)
                        return True
            return False

        if op == "Clean / Surface prep":
            self._populate_etchant_combo(self._etchants_from_library_or_default())
            if not _select_matching_etchant(prev, default_etchant) and default_etchant:
                self._set_etchant_combo_value(default_etchant)
            self.etchant_combo.setEnabled(True)
            return

        if op == "Etch":
            mat = self.current_operation_material_name()
            allowed = self.compatible_etchants(mat)
            if not allowed:
                self.etchant_combo.clear()
                self.etchant_combo.addItem("No compatible etchants", None)
                self.etchant_combo.setEnabled(False)
            else:
                self._populate_etchant_combo(allowed)
                _select_matching_etchant(prev, default_etchant)
                self.etchant_combo.setEnabled(True)
            return

        if op == "Lift-off":
            mat = self.current_operation_material_name()
            allowed = self.compatible_liftoff_methods(mat)
            if not allowed:
                allowed = self._etchants_from_library_or_default()
            self._populate_etchant_combo(allowed)
            if not _select_matching_etchant(prev, default_etchant) and default_etchant:
                self._set_etchant_combo_value(default_etchant)
            self.etchant_combo.setEnabled(True)
            return

        if op in ("Bond", "Strip material"):
            self._populate_etchant_combo(self._etchants_from_library_or_default())
            if not _select_matching_etchant(prev, default_etchant) and default_etchant:
                self._set_etchant_combo_value(default_etchant)
            self.etchant_combo.setEnabled(True)
            return

        if op == "Lithography":
            self._populate_etchant_combo(self._lithography_methods())
            _select_matching_etchant(prev, default_etchant, "Positive lithography")
            self.etchant_combo.setEnabled(True)
            return

        self._populate_etchant_combo(self._etchants_from_library_or_default())
        _select_matching_etchant(prev, default_etchant)
        self.etchant_combo.setEnabled(False)

    def on_etchant_changed(self):
        """
        Update the current etch mode based on the selected etchant
        and any library mapping (etch_modes_by_etchant).
        """
        base = self.etchant_combo.currentData() or self.etchant_combo.currentText()
        base = chem_normalize_ascii((base or "").strip())
        mode = self.etch_modes_by_etchant.get(base, "").lower()
        self.current_etch_mode = "isotropic" if mode == "isotropic" else "anisotropic"

        # Keep the checkbox in sync if it exists
        if hasattr(self, "isotropic_etch_checkbox") and self.isotropic_etch_checkbox is not None:
            block = self.isotropic_etch_checkbox.blockSignals(True)
            try:
                self.isotropic_etch_checkbox.setChecked(self.current_etch_mode == "isotropic")
            finally:
                self.isotropic_etch_checkbox.blockSignals(block)

        if (
            hasattr(self, "op_combo")
            and self.op_combo.currentText() == "Lithography"
            and hasattr(self, "_sync_lithography_invert_from_current_method")
        ):
            self._sync_lithography_invert_from_current_method()

    def on_isotropic_checkbox_toggled(self, checked: bool):
        """
        When the user toggles the isotropic checkbox in the Method dialog,
        update the current etch mode accordingly.
        """
        self.current_etch_mode = "isotropic" if checked else "anisotropic"

    def update_deposit_methods_for_material(self):
        """
        Re-populate the deposit_method_combo based on the selected material.
        """
        if self.op_combo.currentText() != "Deposit":
            return

        mat = self.current_operation_material_name()
        prev = (self.deposit_method_combo.currentText() or "").strip()
        default_state = self._default_method_state_for_operation("Deposit")
        default_method = str(default_state.get("deposit_method", "")).strip()

        self.deposit_method_combo.blockSignals(True)
        self.deposit_method_combo.clear()

        if not mat:
            self.deposit_method_combo.addItem("No compatible methods")
            self.deposit_method_combo.setEnabled(False)
            self.deposit_method_combo.blockSignals(False)
            return

        methods = self.compatible_deposit_methods(mat)

        if not methods:
            self.deposit_method_combo.addItem("No compatible methods")
            self.deposit_method_combo.setEnabled(False)
            self.deposit_method_combo.blockSignals(False)
            return

        for m in methods:
            self.deposit_method_combo.addItem(m)

        self.deposit_method_combo.setEnabled(True)
        chosen_idx = -1
        for candidate in (prev, default_method):
            if not candidate:
                continue
            for i, method_name in enumerate(methods):
                if method_name.strip().lower() == candidate.strip().lower():
                    chosen_idx = i
                    break
            if chosen_idx >= 0:
                break
        if chosen_idx < 0:
            chosen_idx = 0
        self.deposit_method_combo.setCurrentIndex(chosen_idx)
        self.deposit_method_combo.blockSignals(False)
        self.update_deposit_method_ui(self.deposit_method_combo.currentText())

    # ---------- Library I/O ----------
    def load_material_library(self):
        self.material_library_by_cat = {}
        self.compat_etch_by_material = {}
        self.compat_etch_by_category = {}
        self.compat_deposit_by_material = {}
        self.compat_deposit_by_category = {}
        # reset extended compat maps as well
        self.compat_bond_by_material = {}
        self.compat_bond_by_category = {}
        self.compat_strip_by_material = {}
        self.compat_strip_by_category = {}
        self.compat_liftoff_by_material = {}
        self.compat_liftoff_by_category = {}
        self.compat_clean_by_material = {}
        self.compat_clean_by_category = {}
        self.operation_restrictions_by_operation = {}
        self.etch_modes_by_etchant = {}

        if not os.path.exists(self.library_path):
            self.material_library_by_cat = {"Substrates": ["Silicon (Si)"]}
            self.etch_modes_by_etchant = {}
            self.rebuild_flat_library()
            LOGGER.warning(
                "Material library not found; using in-memory defaults: %s",
                self.library_path,
            )
            return

        try:
            with open(self.library_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            validate_material_library(data)

            m = data.get("materials", {})
            if isinstance(m, dict):
                by_cat = {}
                for cat, lst in m.items():
                    if isinstance(lst, list):
                        by_cat[str(cat)] = [str(x) for x in lst if isinstance(x, str)]
                self.material_library_by_cat = by_cat
            elif isinstance(m, list):
                self.material_library_by_cat = {"General": [str(x) for x in m if isinstance(x, str)]}
            else:
                self.material_library_by_cat = {}

            compat = data.get("compatibility", {})
            if isinstance(compat, dict):
                ebm = compat.get("etchants_by_material", {})
                ebc = compat.get("etchants_by_category", {})
                dbm = compat.get("deposit_methods_by_material", {})
                dbc = compat.get("deposit_methods_by_category", {})
                emodes = compat.get("etch_modes_by_etchant", {})

                if isinstance(ebm, dict):
                    self.compat_etch_by_material = {
                        str(k): [str(vv) for vv in (v if isinstance(v, list) else [])]
                        for k, v in ebm.items()
                    }
                if isinstance(ebc, dict):
                    self.compat_etch_by_category = {
                        str(k): [str(vv) for vv in (v if isinstance(v, list) else [])]
                        for k, v in ebc.items()
                    }

                if isinstance(dbm, dict):
                    self.compat_deposit_by_material = {
                        str(k): [str(vv) for vv in (v if isinstance(v, list) else [])]
                        for k, v in dbm.items()
                    }
                if isinstance(dbc, dict):
                    self.compat_deposit_by_category = {
                        str(k): [str(vv) for vv in (v if isinstance(v, list) else [])]
                        for k, v in dbc.items()
                    }

                if isinstance(emodes, dict):
                    # store keys in normalized ASCII, values as "anisotropic"/"isotropic"
                    self.etch_modes_by_etchant = {
                        chem_normalize_ascii(str(k)): str(v).lower()
                        for k, v in emodes.items()
                    }
                else:
                    self.etch_modes_by_etchant = {}

                # Extended compat maps (if present in JSON; otherwise stay as {})
                bond_mat = compat.get("bond_by_material", {})
                if isinstance(bond_mat, dict):
                    self.compat_bond_by_material = {
                        str(k): [str(vv) for vv in (v if isinstance(v, list) else [])]
                        for k, v in bond_mat.items()
                    }

                bond_cat = compat.get("bond_by_category", {})
                if isinstance(bond_cat, dict):
                    self.compat_bond_by_category = {
                        str(k): [str(vv) for vv in (v if isinstance(v, list) else [])]
                        for k, v in bond_cat.items()
                    }

                strip_mat = compat.get("strip_by_material", {})
                if isinstance(strip_mat, dict):
                    self.compat_strip_by_material = {
                        str(k): [str(vv) for vv in (v if isinstance(v, list) else [])]
                        for k, v in strip_mat.items()
                    }

                strip_cat = compat.get("strip_by_category", {})
                if isinstance(strip_cat, dict):
                    self.compat_strip_by_category = {
                        str(k): [str(vv) for vv in (v if isinstance(v, list) else [])]
                        for k, v in strip_cat.items()
                    }

                liftoff_mat = compat.get("liftoff_by_material", {})
                if isinstance(liftoff_mat, dict):
                    self.compat_liftoff_by_material = {
                        str(k): [str(vv) for vv in (v if isinstance(v, list) else [])]
                        for k, v in liftoff_mat.items()
                    }

                liftoff_cat = compat.get("liftoff_by_category", {})
                if isinstance(liftoff_cat, dict):
                    self.compat_liftoff_by_category = {
                        str(k): [str(vv) for vv in (v if isinstance(v, list) else [])]
                        for k, v in liftoff_cat.items()
                    }

                clean_mat = compat.get("clean_by_material", {})
                if isinstance(clean_mat, dict):
                    self.compat_clean_by_material = {
                        str(k): [str(vv) for vv in (v if isinstance(v, list) else [])]
                        for k, v in clean_mat.items()
                    }

                clean_cat = compat.get("clean_by_category", {})
                if isinstance(clean_cat, dict):
                    self.compat_clean_by_category = {
                        str(k): [str(vv) for vv in (v if isinstance(v, list) else [])]
                        for k, v in clean_cat.items()
                    }

                restrictions = compat.get("restrictions_by_operation", {})
                if isinstance(restrictions, dict):
                    self.operation_restrictions_by_operation = {
                        str(op): [str(vv) for vv in (vals if isinstance(vals, list) else [])]
                        for op, vals in restrictions.items()
                    }

            self.rebuild_flat_library()
            LOGGER.info(
                "Loaded material library from %s: %d categories, %d material etch "
                "rules, %d category etch rules, %d material deposit rules, and "
                "%d category deposit rules",
                self.library_path,
                len(self.material_library_by_cat),
                len(self.compat_etch_by_material),
                len(self.compat_etch_by_category),
                len(self.compat_deposit_by_material),
                len(self.compat_deposit_by_category),
            )

        except Exception as exc:
            self.material_library_by_cat = {}
            self.compat_etch_by_material = {}
            self.compat_etch_by_category = {}
            self.compat_deposit_by_material = {}
            self.compat_deposit_by_category = {}
            self.compat_bond_by_material = {}
            self.compat_bond_by_category = {}
            self.compat_strip_by_material = {}
            self.compat_strip_by_category = {}
            self.compat_liftoff_by_material = {}
            self.compat_liftoff_by_category = {}
            self.compat_clean_by_material = {}
            self.compat_clean_by_category = {}
            self.operation_restrictions_by_operation = {}
            self.etch_modes_by_etchant = {}
            self.rebuild_flat_library()
            QMessageBox.warning(
                self,
                "Couldn’t open material library",
                "Restore a valid material library, or open Library Manager and save "
                "to rebuild it. FabStack started with an empty material library. "
                "Your saved library was not changed."
            )
            LOGGER.warning(
                "Could not read material library %s: %s",
                self.library_path,
                exc,
            )

    def rebuild_flat_library(self):
        seen = set()
        flat = []
        for lst in self.material_library_by_cat.values():
            for n in lst:
                if n not in seen:
                    seen.add(n)
                    flat.append(n)
        self.material_library_flat = flat

    def save_material_library(self):
        os.makedirs(os.path.dirname(self.library_path), exist_ok=True)
        tmp_path = self.library_path + ".tmp"
        bak_path = self.library_path + ".bak"

        data = {
            "materials": self.material_library_by_cat,
            "compatibility": {
                "etchants_by_material": self.compat_etch_by_material,
                "etchants_by_category": self.compat_etch_by_category,
                "deposit_methods_by_material": self.compat_deposit_by_material,
                "deposit_methods_by_category": self.compat_deposit_by_category,
                "etch_modes_by_etchant": self.etch_modes_by_etchant,
                # Extended maps (bond/strip/clean); keys kept generic to avoid
                # making assumptions about specific library_manager naming.
                "bond_by_material": self.compat_bond_by_material,
                "bond_by_category": self.compat_bond_by_category,
                "strip_by_material": self.compat_strip_by_material,
                "strip_by_category": self.compat_strip_by_category,
                "liftoff_by_material": self.compat_liftoff_by_material,
                "liftoff_by_category": self.compat_liftoff_by_category,
                "clean_by_material": self.compat_clean_by_material,
                "clean_by_category": self.compat_clean_by_category,
                "restrictions_by_operation": self.operation_restrictions_by_operation,
            },
        }

        try:
            validate_material_library(data)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            if os.path.exists(self.library_path):
                try:
                    shutil.copy2(self.library_path, bak_path)
                except Exception as exc:
                    LOGGER.warning(
                        "Could not create material-library backup %s: %s",
                        bak_path,
                        exc,
                    )

            os.replace(tmp_path, self.library_path)
            LOGGER.info(
                "Saved material library to %s (backup: %s)",
                self.library_path,
                bak_path,
            )

        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def ensure_in_library(self, name: str):
        if not name or name == "Void":
            return
        for lst in self.material_library_by_cat.values():
            if name in lst:
                self.rebuild_flat_library()
                return
        self.material_library_by_cat.setdefault("Custom", []).append(name)
        self.rebuild_flat_library()
        self.save_material_library()

    def is_photoresist_name(self, name: str) -> bool:
        name = str(name or "").strip()
        if not name:
            return False
        if name in self.material_library_by_cat.get("Photoresists", []):
            return True
        low = name.lower()
        return (
            "photoresist" in low
            or "photo resist" in low
            or low.startswith("pr_")
            or "pmma" in low
        )

    # ---------- Library management dialog ----------
    def open_library_manager(self):
        # Expert-only feature
        if not getattr(self, "is_expert_user", False):
            if not self.ensure_expert_user_mode("Library manager"):
                return

        dlg = LibraryManagerDialog(
            self,
            self.material_library_by_cat,
            self.compat_etch_by_material,
            self.compat_etch_by_category,
            self.compat_deposit_by_material,
            self.compat_deposit_by_category,
            self.compat_bond_by_material,
            self.compat_bond_by_category,
            self.compat_strip_by_material,
            self.compat_strip_by_category,
            self.compat_liftoff_by_material,
            self.compat_liftoff_by_category,
            self.compat_clean_by_material,
            self.compat_clean_by_category,
            self.operation_restrictions_by_operation,
            self.etch_modes_by_etchant,
            self.save_material_library,
        )
        if dlg.exec_():
            self.rebuild_flat_library()
            self.refresh_operation_material_combo()
            self.update_etchant_options()
            self.update_deposit_methods_for_material()

    def edit_hidden_categories(self):
        """Dialog to mark categories hidden from materials selections (but not from methods)."""
        if not self.material_library_by_cat:
            QMessageBox.information(
                self,
                "No material categories",
                "Open Library Manager and add a category first.",
            )
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Hide material categories")
        layout = QVBoxLayout(dlg)

        info = QLabel(
            "Selected categories will be hidden when choosing materials.\n"
            "Their methods will remain available."
        )
        layout.addWidget(info)

        checkboxes = {}
        for cat in sorted(self.material_library_by_cat.keys()):
            cb = QCheckBox(cat)
            cb.setChecked(cat in self.hidden_material_categories)
            layout.addWidget(cb)
            checkboxes[cat] = cb

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Apply")
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec_():
            self.hidden_material_categories = {
                cat for cat, cb in checkboxes.items() if cb.isChecked()
            }

    # ---------- Materials ----------
    def open_materials_library_file(self):
        if not os.path.exists(self.library_path):
            LOGGER.warning("Material library file not found: %s", self.library_path)
            QMessageBox.warning(
                self,
                "Material library not found",
                "Open Library Manager and save it to recreate the library."
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.library_path))

    def add_material(self):
        name, ok = QInputDialog.getText(self, "Add material", "Material name:")
        if not ok or not name.strip():
            return
        name = name.strip()

        if name in self.material_names and name in self.deleted_materials:
            restore = self._confirm_material_action(
                f"Restore {name}?",
                "This material was removed from the project. Add it again?",
                "Restore",
                default_to_action=True,
            )
            if restore:
                self.deleted_materials.discard(name)
                if name != "Void":
                    if self.substrate_combo.findText(name) < 0:
                        self.substrate_combo.addItem(name)
                self.refresh_operation_material_combo(preferred_name=name)
                self.refresh_materials_list()
                self.update_etchant_options()
                self.update_deposit_methods_for_material()
            return

        if name in self.material_names:
            QMessageBox.warning(
                self,
                "Material already exists",
                f"{name} is already in this project.",
            )
            return
        color = self._prompt_material_color()
        if color is None:
            return
        color_hex = color.name()
        self.material_names.append(name)
        self.material_colors.append(color_hex)
        if name != "Void":
            self.substrate_combo.addItem(name)
        self.refresh_operation_material_combo(preferred_name=name)
        self.ensure_in_library(name)
        self.update_plot(self.current_title or "")
        self.update_etchant_options()
        self.update_deposit_methods_for_material()
        self.refresh_materials_list()
        self.statusBar().showMessage(f"Added {name} to process materials.", 3000)

    def add_material_from_library(self):
        cats = sorted(
            [c for c in self.material_library_by_cat.keys()
             if c not in self.hidden_material_categories]
        )
        if not cats:
            QMessageBox.information(
                self,
                "No categories available",
                "Open Library Manager and show at least one material category."
            )
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Add material from library")
        dlg.resize(520, 340)

        layout = QVBoxLayout(dlg)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Choose a material"))
        top_row.addStretch(1)
        mat_search = QLineEdit()
        mat_search.setPlaceholderText("Search materials…")
        mat_search.setClearButtonEnabled(True)
        mat_search.setFixedWidth(220)
        top_row.addWidget(mat_search)
        layout.addLayout(top_row)

        lists_row = QHBoxLayout()
        cat_col = QVBoxLayout()
        mat_col = QVBoxLayout()
        lists_row.addLayout(cat_col, 1)
        lists_row.addLayout(mat_col, 1)
        layout.addLayout(lists_row)

        cat_col.addWidget(QLabel("Categories"))
        categories_list = QListWidget()
        cat_col.addWidget(categories_list, 1)

        mat_col.addWidget(QLabel("Materials"))
        materials_list = QListWidget()
        mat_col.addWidget(materials_list, 1)

        for c in cats:
            categories_list.addItem(c)

        def _refresh_materials_for_selected_category():
            materials_list.clear()
            query = (mat_search.text() or "").strip().lower()
            mats = []

            if query:
                # Search across all visible categories.
                seen = set()
                for cat in cats:
                    for m in self.material_library_by_cat.get(cat, []):
                        if query in m.lower() and m not in seen:
                            seen.add(m)
                            mats.append(m)
                mats.sort()
            else:
                cat_item = categories_list.currentItem()
                if cat_item is None:
                    return
                mats = sorted(self.material_library_by_cat.get(cat_item.text(), []))

            for m in mats:
                materials_list.addItem(m)
            if materials_list.count() > 0:
                materials_list.setCurrentRow(0)

        categories_list.currentRowChanged.connect(lambda _: _refresh_materials_for_selected_category())
        mat_search.textChanged.connect(lambda _: _refresh_materials_for_selected_category())

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Add")
        layout.addWidget(buttons)

        def _accept_if_material_selected():
            if materials_list.currentItem() is None:
                QMessageBox.information(
                    self,
                    "Select a material",
                    "Choose the material to add."
                )
                return
            dlg.accept()

        buttons.accepted.connect(_accept_if_material_selected)
        buttons.rejected.connect(dlg.reject)
        materials_list.itemDoubleClicked.connect(lambda _item: _accept_if_material_selected())

        categories_list.setCurrentRow(0)
        _refresh_materials_for_selected_category()

        if dlg.exec_() != QDialog.Accepted:
            return

        picked_item = materials_list.currentItem()
        if picked_item is None:
            return
        name = picked_item.text()
        if not name:
            return

        if name in self.material_names and name in self.deleted_materials:
            self.deleted_materials.discard(name)
            if name != "Void":
                if self.substrate_combo.findText(name) < 0:
                    self.substrate_combo.addItem(name)
            self.refresh_operation_material_combo(preferred_name=name)
            self.refresh_materials_list()
            self.update_plot(self.current_title or "")
            self.update_etchant_options()
            self.update_deposit_methods_for_material()
            self.statusBar().showMessage(f"Restored {name} to process materials.", 3000)
            return

        if name in self.material_names:
            QMessageBox.warning(
                self,
                "Material already added",
                f"{name} is already in this project.",
            )
            return
        color = self._prompt_material_color()
        if color is None:
            return
        color_hex = color.name()
        self.material_names.append(name)
        self.material_colors.append(color_hex)
        if name != "Void":
            self.substrate_combo.addItem(name)
        self.refresh_operation_material_combo(preferred_name=name)
        self.ensure_in_library(name)
        self.update_plot(self.current_title or "")
        self.update_etchant_options()
        self.update_deposit_methods_for_material()
        self.refresh_materials_list()
        self.statusBar().showMessage(f"Added {name} to process materials.", 3000)

    def material_index(self, name: str) -> int:
        return self.material_names.index(name) if name in self.material_names else -1

    def _remove_name_from_combo(self, combo: QComboBox, name: str):
        target = str(name or "").strip()
        if not target:
            return
        i = 0
        while i < combo.count():
            txt = (combo.itemText(i) or "").strip()
            data = combo.itemData(i)
            data_txt = str(data).strip() if data is not None else ""
            if data_txt.startswith("__restricted__:"):
                data_txt = data_txt.split(":", 1)[1].strip()
            if txt == target or data_txt == target:
                combo.removeItem(i)
                continue
            i += 1

    def _material_used_anywhere(self, idx: int) -> bool:
        if idx < 0:
            return False
        if self.xsec is not None:
            if np.any(self.xsec.grid == idx):
                return True
        for st in self.history:
            g = st.get("grid", None)
            if g is not None and np.any(np.asarray(g) == idx):
                return True
        return False

    def modify_material(self, selected_name: str = None):
        candidates = [n for n in self.material_names if n not in ("Void",) and n not in self.deleted_materials]
        if not candidates:
            QMessageBox.information(
                self,
                "No project materials",
                "Add a material before changing its colour.",
            )
            return

        if selected_name is None:
            name, ok = QInputDialog.getItem(
                self,
                "Change material colour",
                "Material:",
                candidates,
                0,
                False,
            )
            if not ok or not name:
                return
        else:
            name = selected_name
            if name not in candidates:
                return

        idx = self.material_index(name)
        if idx < 0:
            return

        color = self._prompt_material_color(exclude_name=name)
        if color is None:
            return

        color_hex = color.name()
        if idx < len(self.material_colors):
            self.material_colors[idx] = color_hex

        self.update_plot(self.current_title or "")
        self.update_etchant_options()
        self.update_deposit_methods_for_material()
        self.refresh_materials_list()
        self.statusBar().showMessage(f"Updated the colour for {name}.", 3000)

    def delete_material(self, selected_name: str = None):
        candidates = [n for n in self.material_names if n not in ("Void",) and n not in self.deleted_materials]
        if not candidates:
            QMessageBox.information(
                self,
                "No project materials",
                "There are no materials to remove.",
            )
            return

        if selected_name is None:
            name, ok = QInputDialog.getItem(
                self,
                "Remove material",
                "Material:",
                candidates,
                0,
                False,
            )
            if not ok or not name:
                return
        else:
            name = selected_name
            if name not in candidates:
                return

        idx = self.material_index(name)
        if idx < 0:
            return

        if self._material_used_anywhere(idx):
            QMessageBox.warning(
                self,
                f"Can’t remove {name}",
                "This material is used by the current stack or process history. "
                "Remove the steps that use it, then try again."
            )
            return

        remove = self._confirm_material_action(
            f"Remove {name}?",
            "Remove this material from the project?",
            "Remove",
        )
        if not remove:
            return

        self.deleted_materials.add(name)
        self._remove_name_from_combo(self.material_combo, name)
        self._remove_name_from_combo(self.substrate_combo, name)
        self.refresh_operation_material_combo()

        self.refresh_materials_list()
        self.update_plot(self.current_title or "")
        self.update_etchant_options()
        self.update_deposit_methods_for_material()
        self.statusBar().showMessage(f"Removed {name} from process materials.", 3000)
