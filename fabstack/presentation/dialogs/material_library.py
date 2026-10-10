import json
import time

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QComboBox,
    QPushButton, QInputDialog, QMessageBox, QTabWidget, QCheckBox, QLineEdit,
    QTreeWidget, QTreeWidgetItem
)
from PyQt5.QtCore import QSettings, Qt

from fabstack.terminology import operation_label

# Predefined list of deposition methods that can be used in compatibility
DEFAULT_DEPOSITION_METHODS = [
    "Thermal deposition",
    "Chemical vapor deposition",
    "Sputtering",
    "Atomic layer deposition",
]


class LibraryManagerDialog(QDialog):
    """
    Manage:
      - materials_library_by_cat (categories + materials)
      - compat_etch_by_material / compat_etch_by_category
      - compat_deposit_by_material / compat_deposit_by_category
      - compat_bond_by_material / compat_bond_by_category
      - compat_strip_by_material / compat_strip_by_category
      - compat_liftoff_by_material / compat_liftoff_by_category
      - compat_clean_by_material / compat_clean_by_category
      - operation_restrictions_by_operation

    All dicts are modified IN-PLACE (references held by MainWindow).
    """

    def _confirm_delete(self, title, message):
        dialog = QMessageBox(
            QMessageBox.Question,
            title,
            message,
            QMessageBox.Yes | QMessageBox.Cancel,
            self,
        )
        delete_button = dialog.button(QMessageBox.Yes)
        delete_button.setText("Delete")
        cancel_button = dialog.button(QMessageBox.Cancel)
        dialog.setDefaultButton(cancel_button)
        return dialog.exec_() == QMessageBox.Yes

    def __init__(
        self,
        parent,
        library_by_cat,
        compat_etch_by_material,
        compat_etch_by_category,
        compat_deposit_by_material,
        compat_deposit_by_category,
        compat_bond_by_material,
        compat_bond_by_category,
        compat_strip_by_material,
        compat_strip_by_category,
        compat_liftoff_by_material,
        compat_liftoff_by_category,
        compat_clean_by_material,
        compat_clean_by_category,
        operation_restrictions_by_operation,
        etch_modes_by_etchant=None,
        save_callback=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Materials library")
        self.resize(900, 600)

        # Keep explicit reference to parent (MainWindow) so we can call edit_hidden_categories()
        self._parent = parent

        # References to data structures from MainWindow
        self._library = library_by_cat

        self._compat_by_material = compat_etch_by_material
        self._compat_by_category = compat_etch_by_category

        self._compat_deposit_by_material = compat_deposit_by_material
        self._compat_deposit_by_category = compat_deposit_by_category

        self._compat_bond_by_material = compat_bond_by_material
        self._compat_bond_by_category = compat_bond_by_category

        self._compat_strip_by_material = compat_strip_by_material
        self._compat_strip_by_category = compat_strip_by_category

        self._compat_liftoff_by_material = compat_liftoff_by_material
        self._compat_liftoff_by_category = compat_liftoff_by_category

        self._compat_clean_by_material = compat_clean_by_material
        self._compat_clean_by_category = compat_clean_by_category
        self._op_restrictions_by_operation = operation_restrictions_by_operation
        self._etch_modes_by_etchant = etch_modes_by_etchant or {}

        self._save_callback = save_callback

        # QSettings instance for persisting hidden categories
        # (organization / application names can be adapted if you prefer)
        self._settings = QSettings("FabStack", "FabStack")
        self._material_created_at = self._load_material_created_at_map()
        self._materials_sort_mode = str(self._settings.value("materials_sort_mode", "name")).strip().lower()
        if self._materials_sort_mode not in ("name", "date"):
            self._materials_sort_mode = "name"
        self._sync_material_dates_with_library()

        root_layout = QVBoxLayout(self)

        # Top-level tabs
        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs)

        # ------------------------------------------------------------------
        # TAB 1: Materials
        # ------------------------------------------------------------------
        materials_tab = QDialog(self)
        mat_layout = QVBoxLayout(materials_tab)

        # Category selector + "Hide categories" button on the same row
        cat_row = QHBoxLayout()
        cat_label = QLabel("Category:")
        self.cat_combo = QComboBox()
        self.btn_hide_categories = QPushButton("Hide material categories…")

        cat_row.addWidget(cat_label)
        cat_row.addWidget(self.cat_combo)
        cat_row.addStretch(1)  # pushes the button to the right
        cat_row.addWidget(self.btn_hide_categories)

        mat_layout.addLayout(cat_row)

        # Materials list
        self.material_list = QListWidget()
        materials_header_row = QHBoxLayout()
        materials_header_row.addWidget(QLabel("Materials"))
        materials_header_row.addStretch(1)
        self.btn_toggle_material_sort = QPushButton("⇅")
        # Make the sort control more legible (icon + click target).
        self.btn_toggle_material_sort.setFixedSize(48, 28)
        self.btn_toggle_material_sort.setStyleSheet(
            "QPushButton { font-size: 16px; font-weight: 700; padding: 0px 2px; }"
        )
        materials_header_row.addWidget(self.btn_toggle_material_sort)
        mat_layout.addLayout(materials_header_row)
        mat_layout.addWidget(self.material_list)

        # Buttons for categories
        cat_btn_row = QHBoxLayout()
        self.btn_add_cat = QPushButton("Add category")
        self.btn_ren_cat = QPushButton("Rename category")
        self.btn_del_cat = QPushButton("Delete category")
        cat_btn_row.addWidget(self.btn_add_cat)
        cat_btn_row.addWidget(self.btn_ren_cat)
        cat_btn_row.addWidget(self.btn_del_cat)
        mat_layout.addLayout(cat_btn_row)

        # Buttons for materials
        mat_btn_row = QHBoxLayout()
        self.btn_add_mat = QPushButton("Add material")
        self.btn_ren_mat = QPushButton("Rename material")
        self.btn_del_mat = QPushButton("Delete material")
        mat_btn_row.addWidget(self.btn_add_mat)
        mat_btn_row.addWidget(self.btn_ren_mat)
        mat_btn_row.addWidget(self.btn_del_mat)
        mat_layout.addLayout(mat_btn_row)

        self.tabs.addTab(materials_tab, "Materials")

        # ------------------------------------------------------------------
        # TAB 2: Compatibility (Etch / Deposit / Bond / Strip / Lift-off / Clean)
        # ------------------------------------------------------------------
        compat_tab = QDialog(self)
        compat_main_layout = QVBoxLayout(compat_tab)

        compat_search_row = QHBoxLayout()
        compat_search_row.addWidget(QLabel("Target:"))
        self.compat_target_search_edit = QLineEdit()
        self.compat_target_search_edit.setPlaceholderText("Search materials or categories…")
        self.compat_target_search_edit.setClearButtonEnabled(True)
        compat_search_row.addWidget(self.compat_target_search_edit, 1)
        compat_search_row.addWidget(QLabel("Compatible items:"))
        self.compat_entry_search_edit = QLineEdit()
        self.compat_entry_search_edit.setPlaceholderText("Search compatible items…")
        self.compat_entry_search_edit.setClearButtonEnabled(True)
        compat_search_row.addWidget(self.compat_entry_search_edit, 1)
        compat_main_layout.addLayout(compat_search_row)

        compat_body = QHBoxLayout()
        compat_main_layout.addLayout(compat_body, 1)

        self.comp_tabs = QTabWidget()
        compat_body.addWidget(self.comp_tabs, 3)

        compat_side = QVBoxLayout()
        compat_side.addWidget(QLabel("Materials"))
        self.compat_tree_search_edit = QLineEdit()
        self.compat_tree_search_edit.setPlaceholderText("Search materials or categories…")
        self.compat_tree_search_edit.setClearButtonEnabled(True)
        compat_side.addWidget(self.compat_tree_search_edit)
        self.compat_material_tree = QTreeWidget()
        self.compat_material_tree.setHeaderHidden(True)
        self.compat_material_tree.setRootIsDecorated(True)
        compat_side.addWidget(self.compat_material_tree, 1)
        compat_body.addLayout(compat_side, 2)

        # ----- Etch compatibility sub-tab -----
        etch_tab = QDialog(self)
        etch_layout = QVBoxLayout(etch_tab)

        self.comp_scope_combo = QComboBox()
        self.comp_scope_combo.addItems(["By material", "By category"])
        etch_layout.addWidget(QLabel("Scope:"))
        etch_layout.addWidget(self.comp_scope_combo)

        self.comp_target_combo = QComboBox()
        etch_layout.addWidget(QLabel("Target:"))
        etch_layout.addWidget(self.comp_target_combo)

        self.comp_etch_list = QListWidget()
        etch_layout.addWidget(QLabel("Compatible etchants"))
        etch_layout.addWidget(self.comp_etch_list)

        etch_btn_row = QHBoxLayout()
        self.btn_add_etch = QPushButton("Add etchant")
        self.btn_del_etch = QPushButton("Remove etchant")
        etch_btn_row.addWidget(self.btn_add_etch)
        etch_btn_row.addWidget(self.btn_del_etch)
        etch_layout.addLayout(etch_btn_row)

        self.comp_tabs.addTab(etch_tab, "Etch")

        # ----- Deposit compatibility sub-tab -----
        dep_tab = QDialog(self)
        dep_layout = QVBoxLayout(dep_tab)

        self.dep_scope_combo = QComboBox()
        self.dep_scope_combo.addItems(["By material", "By category"])
        dep_layout.addWidget(QLabel("Scope:"))
        dep_layout.addWidget(self.dep_scope_combo)

        self.dep_target_combo = QComboBox()
        dep_layout.addWidget(QLabel("Target:"))
        dep_layout.addWidget(self.dep_target_combo)

        self.dep_method_list = QListWidget()
        dep_layout.addWidget(QLabel("Compatible deposition methods"))
        dep_layout.addWidget(self.dep_method_list)

        dep_btn_row = QHBoxLayout()
        self.btn_add_method = QPushButton("Add method")
        self.btn_del_method = QPushButton("Remove method")
        dep_btn_row.addWidget(self.btn_add_method)
        dep_btn_row.addWidget(self.btn_del_method)
        dep_layout.addLayout(dep_btn_row)

        self.comp_tabs.addTab(dep_tab, "Deposit")

        # ----- Bond compatibility sub-tab -----
        bond_tab = QDialog(self)
        bond_layout = QVBoxLayout(bond_tab)

        self.bond_scope_combo = QComboBox()
        self.bond_scope_combo.addItems(["By material", "By category"])
        bond_layout.addWidget(QLabel("Scope:"))
        bond_layout.addWidget(self.bond_scope_combo)

        self.bond_target_combo = QComboBox()
        bond_layout.addWidget(QLabel("Target:"))
        bond_layout.addWidget(self.bond_target_combo)

        self.bond_method_list = QListWidget()
        bond_layout.addWidget(QLabel("Compatible bond methods"))
        bond_layout.addWidget(self.bond_method_list)

        bond_btn_row = QHBoxLayout()
        self.btn_add_bond = QPushButton("Add bond method")
        self.btn_del_bond = QPushButton("Remove bond method")
        bond_btn_row.addWidget(self.btn_add_bond)
        bond_btn_row.addWidget(self.btn_del_bond)
        bond_layout.addLayout(bond_btn_row)

        self.comp_tabs.addTab(bond_tab, "Bond")

        # ----- Strip compatibility sub-tab -----
        strip_tab = QDialog(self)
        strip_layout = QVBoxLayout(strip_tab)

        self.strip_scope_combo = QComboBox()
        self.strip_scope_combo.addItems(["By material", "By category"])
        strip_layout.addWidget(QLabel("Scope:"))
        strip_layout.addWidget(self.strip_scope_combo)

        self.strip_target_combo = QComboBox()
        strip_layout.addWidget(QLabel("Target:"))
        strip_layout.addWidget(self.strip_target_combo)

        self.strip_method_list = QListWidget()
        strip_layout.addWidget(QLabel("Compatible strip methods"))
        strip_layout.addWidget(self.strip_method_list)

        strip_btn_row = QHBoxLayout()
        self.btn_add_strip = QPushButton("Add strip method")
        self.btn_del_strip = QPushButton("Remove strip method")
        strip_btn_row.addWidget(self.btn_add_strip)
        strip_btn_row.addWidget(self.btn_del_strip)
        strip_layout.addLayout(strip_btn_row)

        self.comp_tabs.addTab(strip_tab, "Strip")

        # ----- Lift-off compatibility sub-tab -----
        liftoff_tab = QDialog(self)
        liftoff_layout = QVBoxLayout(liftoff_tab)

        self.liftoff_scope_combo = QComboBox()
        self.liftoff_scope_combo.addItems(["By material", "By category"])
        liftoff_layout.addWidget(QLabel("Scope:"))
        liftoff_layout.addWidget(self.liftoff_scope_combo)

        self.liftoff_target_combo = QComboBox()
        liftoff_layout.addWidget(QLabel("Target:"))
        liftoff_layout.addWidget(self.liftoff_target_combo)

        self.liftoff_method_list = QListWidget()
        liftoff_layout.addWidget(QLabel("Compatible lift-off methods"))
        liftoff_layout.addWidget(self.liftoff_method_list)

        liftoff_btn_row = QHBoxLayout()
        self.btn_add_liftoff = QPushButton("Add lift-off method")
        self.btn_del_liftoff = QPushButton("Remove lift-off method")
        liftoff_btn_row.addWidget(self.btn_add_liftoff)
        liftoff_btn_row.addWidget(self.btn_del_liftoff)
        liftoff_layout.addLayout(liftoff_btn_row)

        self.comp_tabs.addTab(liftoff_tab, "Lift-off")

        # ----- Clean compatibility sub-tab -----
        clean_tab = QDialog(self)
        clean_layout = QVBoxLayout(clean_tab)

        self.clean_scope_combo = QComboBox()
        self.clean_scope_combo.addItems(["By material", "By category"])
        clean_layout.addWidget(QLabel("Scope:"))
        clean_layout.addWidget(self.clean_scope_combo)

        self.clean_target_combo = QComboBox()
        clean_layout.addWidget(QLabel("Target:"))
        clean_layout.addWidget(self.clean_target_combo)

        self.clean_method_list = QListWidget()
        clean_layout.addWidget(QLabel("Compatible clean / surface prep methods"))
        clean_layout.addWidget(self.clean_method_list)

        clean_btn_row = QHBoxLayout()
        self.btn_add_clean = QPushButton("Add clean method")
        self.btn_del_clean = QPushButton("Remove clean method")
        clean_btn_row.addWidget(self.btn_add_clean)
        clean_btn_row.addWidget(self.btn_del_clean)
        clean_layout.addLayout(clean_btn_row)

        self.comp_tabs.addTab(clean_tab, "Clean")

        self.tabs.addTab(compat_tab, "Compatibility")

        # ------------------------------------------------------------------
        # TAB 3: Restrictions (top-level)
        # ------------------------------------------------------------------
        restr_tab = QDialog(self)
        restr_main_layout = QHBoxLayout(restr_tab)

        restr_left = QVBoxLayout()
        restr_main_layout.addLayout(restr_left, 3)

        self.restrict_op_combo = QComboBox()
        restriction_operations = [
            "Deposit",
            "Spincoating",
            "Bond",
            "Etch",
            "Strip material",
            "Lift-off",
            "Clean / Surface prep",
            "Bake/Annealing",
            "Lithography",
        ]
        for operation in restriction_operations:
            self.restrict_op_combo.addItem(operation_label(operation), operation)
        restr_op_row = QHBoxLayout()
        restr_op_row.addWidget(QLabel("Operation:"))
        restr_op_row.addWidget(self.restrict_op_combo, 1)
        self.restrict_target_search_edit = QLineEdit()
        self.restrict_target_search_edit.setPlaceholderText("Find operation…")
        self.restrict_target_search_edit.setClearButtonEnabled(True)
        restr_op_row.addWidget(self.restrict_target_search_edit, 1)
        restr_left.addLayout(restr_op_row)

        restr_left.addWidget(QLabel("Restrictions"))
        self.restrict_entry_search_edit = QLineEdit()
        self.restrict_entry_search_edit.setPlaceholderText("Search restricted materials…")
        self.restrict_entry_search_edit.setClearButtonEnabled(True)
        restr_left.addWidget(self.restrict_entry_search_edit)

        self.restrict_material_list = QListWidget()
        restr_left.addWidget(QLabel("Restricted materials"))
        restr_left.addWidget(self.restrict_material_list, 1)

        restr_btn_row = QHBoxLayout()
        self.btn_add_restriction = QPushButton("Add restriction")
        self.btn_del_restriction = QPushButton("Remove restriction")
        restr_btn_row.addWidget(self.btn_add_restriction)
        restr_btn_row.addWidget(self.btn_del_restriction)
        restr_left.addLayout(restr_btn_row)

        restr_right = QVBoxLayout()
        restr_main_layout.addLayout(restr_right, 2)
        restr_right.addWidget(QLabel("Materials"))
        self.restrict_tree_search_edit = QLineEdit()
        self.restrict_tree_search_edit.setPlaceholderText("Search materials or categories…")
        self.restrict_tree_search_edit.setClearButtonEnabled(True)
        restr_right.addWidget(self.restrict_tree_search_edit)
        self.restrict_material_tree = QTreeWidget()
        self.restrict_material_tree.setHeaderHidden(True)
        self.restrict_material_tree.setRootIsDecorated(True)
        restr_right.addWidget(self.restrict_material_tree, 1)

        self.tabs.addTab(restr_tab, "Restrictions")

        # ------------------------------------------------------------------
        # Bottom buttons
        # ------------------------------------------------------------------
        bottom_row = QHBoxLayout()
        bottom_row.addStretch(1)
        self.btn_save = QPushButton("Save && close")
        self.btn_cancel = QPushButton("Cancel")
        bottom_row.addWidget(self.btn_save)
        bottom_row.addWidget(self.btn_cancel)
        root_layout.addLayout(bottom_row)

        # ------------------------------------------------------------------
        # Signals
        # ------------------------------------------------------------------
        # Materials tab
        self.cat_combo.currentTextChanged.connect(self._on_category_changed)
        self.btn_add_cat.clicked.connect(self._add_category)
        self.btn_ren_cat.clicked.connect(self._rename_category)
        self.btn_del_cat.clicked.connect(self._delete_category)
        self.btn_add_mat.clicked.connect(self._add_material)
        self.btn_ren_mat.clicked.connect(self._rename_material)
        self.btn_del_mat.clicked.connect(self._delete_material)
        self.btn_toggle_material_sort.clicked.connect(self._toggle_material_sort_mode)

        # Hide categories button
        self.btn_hide_categories.clicked.connect(self._on_hide_categories_clicked)

        # Shared compatibility helpers
        self.comp_tabs.currentChanged.connect(self._on_comp_subtab_changed)
        self.compat_target_search_edit.textChanged.connect(self._on_compat_target_search_changed)
        self.compat_entry_search_edit.textChanged.connect(self._on_compat_entry_search_changed)
        self.compat_tree_search_edit.textChanged.connect(self._refresh_compat_material_tree)
        self.compat_material_tree.itemDoubleClicked.connect(self._on_compat_material_tree_double_clicked)

        # Etch compatibility
        self.comp_scope_combo.currentIndexChanged.connect(self._reload_comp_targets)
        self.comp_target_combo.currentTextChanged.connect(self._reload_comp_etchants)
        self.btn_add_etch.clicked.connect(self._add_etchant)
        self.btn_del_etch.clicked.connect(self._delete_etchant)

        # Deposit compatibility
        self.dep_scope_combo.currentIndexChanged.connect(self._reload_dep_targets)
        self.dep_target_combo.currentTextChanged.connect(self._reload_dep_methods)
        self.btn_add_method.clicked.connect(self._add_dep_method)
        self.btn_del_method.clicked.connect(self._delete_dep_method)

        # Bond compatibility
        self.bond_scope_combo.currentIndexChanged.connect(self._reload_bond_targets)
        self.bond_target_combo.currentTextChanged.connect(self._reload_bond_methods)
        self.btn_add_bond.clicked.connect(self._add_bond_method)
        self.btn_del_bond.clicked.connect(self._delete_bond_method)

        # Strip compatibility
        self.strip_scope_combo.currentIndexChanged.connect(self._reload_strip_targets)
        self.strip_target_combo.currentTextChanged.connect(self._reload_strip_methods)
        self.btn_add_strip.clicked.connect(self._add_strip_method)
        self.btn_del_strip.clicked.connect(self._delete_strip_method)

        # Lift-off compatibility
        self.liftoff_scope_combo.currentIndexChanged.connect(self._reload_liftoff_targets)
        self.liftoff_target_combo.currentTextChanged.connect(self._reload_liftoff_methods)
        self.btn_add_liftoff.clicked.connect(self._add_liftoff_method)
        self.btn_del_liftoff.clicked.connect(self._delete_liftoff_method)

        # Clean compatibility
        self.clean_scope_combo.currentIndexChanged.connect(self._reload_clean_targets)
        self.clean_target_combo.currentTextChanged.connect(self._reload_clean_methods)
        self.btn_add_clean.clicked.connect(self._add_clean_method)
        self.btn_del_clean.clicked.connect(self._delete_clean_method)

        # Operation restrictions
        self.restrict_op_combo.currentIndexChanged.connect(
            lambda _: self._reload_restrictions_for_operation(self._current_restriction_operation())
        )
        self.restrict_op_combo.currentIndexChanged.connect(
            lambda _: self._refresh_restrict_material_tree()
        )
        self.restrict_target_search_edit.textChanged.connect(self._on_restrict_target_search_changed)
        self.restrict_entry_search_edit.textChanged.connect(
            lambda _: self._reload_restrictions_for_operation(self._current_restriction_operation())
        )
        self.restrict_tree_search_edit.textChanged.connect(self._refresh_restrict_material_tree)
        self.restrict_material_tree.itemDoubleClicked.connect(self._on_restrict_material_tree_double_clicked)
        self.btn_add_restriction.clicked.connect(self._add_operation_restriction)
        self.btn_del_restriction.clicked.connect(self._delete_operation_restriction)

        # Bottom
        self.btn_save.clicked.connect(self._on_save)
        self.btn_cancel.clicked.connect(self.reject)

        # Initial population
        self._reload_categories()
        self._update_material_sort_button_tooltip()
        self._on_category_changed(self.cat_combo.currentText())
        self._reload_comp_targets()
        self._reload_comp_etchants(self.comp_target_combo.currentText())
        self._reload_dep_targets()
        self._reload_dep_methods(self.dep_target_combo.currentText())
        self._reload_bond_targets()
        self._reload_bond_methods(self.bond_target_combo.currentText())
        self._reload_strip_targets()
        self._reload_strip_methods(self.strip_target_combo.currentText())
        self._reload_liftoff_targets()
        self._reload_liftoff_methods(self.liftoff_target_combo.currentText())
        self._reload_clean_targets()
        self._reload_clean_methods(self.clean_target_combo.currentText())
        self._on_comp_subtab_changed(self.comp_tabs.currentIndex())
        self._reload_restrictions_for_operation(self._current_restriction_operation())
        self._refresh_restrict_material_tree()

    # ------------------------------------------------------------------
    # Shared search / tree helpers
    # ------------------------------------------------------------------
    def _compat_target_query(self):
        return (self.compat_target_search_edit.text() or "").strip().lower()

    def _compat_entry_query(self):
        return (self.compat_entry_search_edit.text() or "").strip().lower()

    def _restrict_target_query(self):
        return (self.restrict_target_search_edit.text() or "").strip().lower()

    def _restrict_entry_query(self):
        return (self.restrict_entry_search_edit.text() or "").strip().lower()

    def _current_restriction_operation(self):
        """Return the persisted operation key behind the displayed label."""
        return str(self.restrict_op_combo.currentData() or "").strip()

    def _populate_material_tree_widget(self, tree: QTreeWidget, query: str):
        q = (query or "").strip().lower()
        tree.clear()
        for cat in sorted(self._library.keys()):
            mats = sorted(self._library.get(cat, []), key=lambda s: s.lower())
            if not mats and q and q not in cat.lower():
                continue

            if q:
                cat_match = q in cat.lower()
                filtered = mats if cat_match else [m for m in mats if q in m.lower()]
                if not cat_match and not filtered:
                    continue
            else:
                filtered = mats

            cat_item = QTreeWidgetItem([cat])
            cat_item.setData(0, Qt.UserRole, ("category", cat))
            tree.addTopLevelItem(cat_item)
            for mat in filtered:
                child = QTreeWidgetItem([mat])
                child.setData(0, Qt.UserRole, ("material", cat, mat))
                cat_item.addChild(child)
            cat_item.setExpanded(bool(q) or len(filtered) <= 12)

    def _refresh_compat_material_tree(self):
        self._populate_material_tree_widget(
            self.compat_material_tree,
            self.compat_tree_search_edit.text(),
        )

    def _refresh_restrict_material_tree(self):
        self._populate_material_tree_widget(
            self.restrict_material_tree,
            self.restrict_tree_search_edit.text(),
        )

    def _current_compat_context(self):
        tab = self.comp_tabs.tabText(self.comp_tabs.currentIndex())
        if tab == "Etch":
            return {
                "scope_combo": self.comp_scope_combo,
                "target_combo": self.comp_target_combo,
                "list_widget": self.comp_etch_list,
                "by_material": self._compat_by_material,
                "by_category": self._compat_by_category,
                "reload_targets": self._reload_comp_targets,
                "reload_list": self._reload_comp_etchants,
            }
        if tab == "Deposit":
            return {
                "scope_combo": self.dep_scope_combo,
                "target_combo": self.dep_target_combo,
                "list_widget": self.dep_method_list,
                "by_material": self._compat_deposit_by_material,
                "by_category": self._compat_deposit_by_category,
                "reload_targets": self._reload_dep_targets,
                "reload_list": self._reload_dep_methods,
            }
        if tab == "Bond":
            return {
                "scope_combo": self.bond_scope_combo,
                "target_combo": self.bond_target_combo,
                "list_widget": self.bond_method_list,
                "by_material": self._compat_bond_by_material,
                "by_category": self._compat_bond_by_category,
                "reload_targets": self._reload_bond_targets,
                "reload_list": self._reload_bond_methods,
            }
        if tab == "Strip":
            return {
                "scope_combo": self.strip_scope_combo,
                "target_combo": self.strip_target_combo,
                "list_widget": self.strip_method_list,
                "by_material": self._compat_strip_by_material,
                "by_category": self._compat_strip_by_category,
                "reload_targets": self._reload_strip_targets,
                "reload_list": self._reload_strip_methods,
            }
        if tab == "Lift-off":
            return {
                "scope_combo": self.liftoff_scope_combo,
                "target_combo": self.liftoff_target_combo,
                "list_widget": self.liftoff_method_list,
                "by_material": self._compat_liftoff_by_material,
                "by_category": self._compat_liftoff_by_category,
                "reload_targets": self._reload_liftoff_targets,
                "reload_list": self._reload_liftoff_methods,
            }
        return {
            "scope_combo": self.clean_scope_combo,
            "target_combo": self.clean_target_combo,
            "list_widget": self.clean_method_list,
            "by_material": self._compat_clean_by_material,
            "by_category": self._compat_clean_by_category,
            "reload_targets": self._reload_clean_targets,
            "reload_list": self._reload_clean_methods,
        }

    def _target_matches_query(self, value: str, query: str) -> bool:
        if not query:
            return True
        return query in (value or "").lower()

    def _select_target_by_query(self):
        ctx = self._current_compat_context()
        combo = ctx["target_combo"]
        query = self._compat_target_query()
        if not query:
            return
        for i in range(combo.count()):
            txt = combo.itemText(i)
            if self._target_matches_query(txt.lower(), query):
                combo.setCurrentIndex(i)
                return

    def _on_comp_subtab_changed(self, _index):
        self._select_target_by_query()
        ctx = self._current_compat_context()
        ctx["reload_list"](ctx["target_combo"].currentText())
        self._refresh_compat_material_tree()

    def _on_compat_target_search_changed(self, _text):
        ctx = self._current_compat_context()
        ctx["reload_targets"]()
        self._select_target_by_query()
        ctx["reload_list"](ctx["target_combo"].currentText())

    def _on_compat_entry_search_changed(self, _text):
        ctx = self._current_compat_context()
        ctx["reload_list"](ctx["target_combo"].currentText())

    def _on_compat_material_tree_double_clicked(self, item, _column):
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        ctx = self._current_compat_context()
        scope_txt = ctx["scope_combo"].currentText().lower()
        scope = "material" if "material" in scope_txt else "category"
        target = ctx["target_combo"].currentText()
        if not target:
            QMessageBox.warning(
                self,
                "Select a target",
                "Choose a material or category first.",
            )
            return
        lst = ctx["by_material"].setdefault(target, []) if scope == "material" else ctx["by_category"].setdefault(target, [])

        changed = False
        if data[0] == "material":
            mat = data[2]
            if mat not in lst:
                lst.append(mat)
                changed = True
        elif data[0] == "category":
            cat = data[1]
            for mat in sorted(self._library.get(cat, []), key=lambda s: s.lower()):
                if mat not in lst:
                    lst.append(mat)
                    changed = True
        if changed:
            ctx["reload_list"](target)

    def _on_restrict_target_search_changed(self, _text):
        q = self._restrict_target_query()
        if not q:
            return
        for i in range(self.restrict_op_combo.count()):
            txt = self.restrict_op_combo.itemText(i).lower()
            if q in txt:
                self.restrict_op_combo.setCurrentIndex(i)
                return

    def _on_restrict_material_tree_double_clicked(self, item, _column):
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        operation = self._current_restriction_operation()
        if not operation:
            return
        lst = self._op_restrictions_by_operation.setdefault(operation, [])
        changed = False
        if data[0] == "material":
            mat = data[2]
            if mat not in lst:
                lst.append(mat)
                changed = True
        elif data[0] == "category":
            cat = data[1]
            for mat in sorted(self._library.get(cat, []), key=lambda s: s.lower()):
                if mat not in lst:
                    lst.append(mat)
                    changed = True
        if changed:
            self._reload_restrictions_for_operation(operation)

    # ------------------------------------------------------------------
    # Materials tab helpers
    # ------------------------------------------------------------------
    def _material_date_key(self, category: str, material: str) -> str:
        return f"{category}||{material}"

    def _load_material_created_at_map(self):
        raw = self._settings.value("materials_created_at", "{}")
        if isinstance(raw, dict):
            src = raw
        else:
            try:
                src = json.loads(str(raw or "{}"))
            except Exception:
                src = {}
        cleaned = {}
        if isinstance(src, dict):
            for k, v in src.items():
                key = str(k).strip()
                if not key:
                    continue
                try:
                    cleaned[key] = float(v)
                except Exception:
                    continue
        return cleaned

    def _save_material_created_at_map(self):
        try:
            self._settings.setValue(
                "materials_created_at",
                json.dumps(self._material_created_at, ensure_ascii=True, sort_keys=True),
            )
        except Exception:
            pass

    def _sync_material_dates_with_library(self):
        # Remove stale keys that no longer exist in library.
        valid_keys = set()
        for cat, mats in self._library.items():
            for name in mats:
                valid_keys.add(self._material_date_key(cat, name))

        stale = [k for k in self._material_created_at.keys() if k not in valid_keys]
        for k in stale:
            self._material_created_at.pop(k, None)

        # Add missing timestamps while preserving current order per category.
        now = time.time()
        changed = bool(stale)
        for cat, mats in self._library.items():
            base = now - float(len(mats))
            for idx, name in enumerate(mats):
                key = self._material_date_key(cat, name)
                if key not in self._material_created_at:
                    self._material_created_at[key] = base + float(idx)
                    changed = True

        if changed:
            self._save_material_created_at_map()

    def _sorted_material_names(self, cat):
        mats = list(self._library.get(cat, []))
        if self._materials_sort_mode == "date":
            mats.sort(
                key=lambda n: (
                    self._material_created_at.get(self._material_date_key(cat, n), 0.0),
                    n.lower(),
                ),
                reverse=True,
            )
        else:
            mats.sort(key=lambda n: n.lower())
        return mats

    def _update_material_sort_button_tooltip(self):
        mode = "Date added (newest first)" if self._materials_sort_mode == "date" else "Name (A-Z)"
        nxt = "name" if self._materials_sort_mode == "date" else "date"
        self.btn_toggle_material_sort.setToolTip(
            f"Sorted by {mode}. Select to sort by {nxt}."
        )

    def _toggle_material_sort_mode(self):
        self._materials_sort_mode = "date" if self._materials_sort_mode == "name" else "name"
        try:
            self._settings.setValue("materials_sort_mode", self._materials_sort_mode)
        except Exception:
            pass
        self._update_material_sort_button_tooltip()
        self._on_category_changed(self.cat_combo.currentText())

    def _reload_categories(self):
        self.cat_combo.blockSignals(True)
        self.cat_combo.clear()
        cats = sorted(self._library.keys())
        self.cat_combo.addItems(cats)
        self.cat_combo.blockSignals(False)

    def _on_category_changed(self, cat):
        self.material_list.clear()
        if not cat or cat not in self._library:
            return
        for name in self._sorted_material_names(cat):
            self.material_list.addItem(name)

    # ------------------------------------------------------------------
    # Hide categories handler
    # ------------------------------------------------------------------
    def _on_hide_categories_clicked(self):
        """
        Edit hidden material categories directly from this dialog so it works
        even while Library Manager is modal, then sync back to MainWindow.
        """
        parent = self._parent
        if not self._library:
            QMessageBox.information(
                self,
                "No material categories",
                "Add a category first.",
            )
            return

        # Start from parent state when available, else from QSettings.
        hidden_set = set()
        if parent is not None and hasattr(parent, "hidden_material_categories"):
            try:
                hidden_set = set(getattr(parent, "hidden_material_categories") or [])
            except Exception:
                hidden_set = set()
        if not hidden_set:
            try:
                saved = self._settings.value("materials_hidden_categories", [])
                if isinstance(saved, str):
                    hidden_set = {s for s in saved.split("||") if s}
                elif isinstance(saved, list):
                    hidden_set = {str(s) for s in saved}
            except Exception:
                hidden_set = set()

        dlg = QDialog(self)
        dlg.setWindowTitle("Hide material categories")
        layout = QVBoxLayout(dlg)
        info = QLabel(
            "Selected categories will be hidden when choosing materials.\n"
            "Their methods will remain available."
        )
        layout.addWidget(info)

        checkboxes = {}
        for cat in sorted(self._library.keys()):
            cb = QCheckBox(cat)
            cb.setChecked(cat in hidden_set)
            layout.addWidget(cb)
            checkboxes[cat] = cb

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_ok = QPushButton("Apply")
        btn_cancel = QPushButton("Cancel")
        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)

        if dlg.exec_() != QDialog.Accepted:
            return

        hidden_set = {cat for cat, cb in checkboxes.items() if cb.isChecked()}

        if parent is not None and hasattr(parent, "hidden_material_categories"):
            try:
                parent.hidden_material_categories = set(hidden_set)
            except Exception:
                parent.hidden_material_categories = list(hidden_set)

        try:
            self._settings.setValue("materials_hidden_categories", sorted(hidden_set))
        except Exception:
            pass

    def _add_category(self):
        name, ok = QInputDialog.getText(self, "Add category", "Category name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in self._library:
            QMessageBox.warning(
                self,
                "Category already exists",
                f"{name} is already in the library.",
            )
            return
        self._library[name] = []
        self._reload_categories()
        self.cat_combo.setCurrentText(name)
        self._reload_comp_targets()
        self._reload_dep_targets()
        self._reload_bond_targets()
        self._reload_strip_targets()
        self._reload_liftoff_targets()
        self._reload_clean_targets()
        self._refresh_compat_material_tree()
        self._refresh_restrict_material_tree()

    def _rename_category(self):
        old = self.cat_combo.currentText()
        if not old:
            QMessageBox.warning(
                self,
                "Choose a category",
                "Select the category to rename.",
            )
            return
        new, ok = QInputDialog.getText(self, "Rename category", "New name:", text=old)
        if not ok or not new.strip():
            return
        new = new.strip()
        if new == old:
            return
        if new in self._library:
            QMessageBox.warning(
                self,
                "Category already exists",
                f"{new} is already in the library.",
            )
            return

        # Move materials
        moved_materials = list(self._library.get(old, []))
        self._library[new] = self._library.pop(old)

        # Move etch compatibility-by-category if present
        if old in self._compat_by_category:
            self._compat_by_category[new] = self._compat_by_category.pop(old)

        # Move deposit compatibility-by-category if present
        if old in self._compat_deposit_by_category:
            self._compat_deposit_by_category[new] = self._compat_deposit_by_category.pop(old)

        # Move bond compatibility-by-category if present
        if old in self._compat_bond_by_category:
            self._compat_bond_by_category[new] = self._compat_bond_by_category.pop(old)

        # Move strip compatibility-by-category if present
        if old in self._compat_strip_by_category:
            self._compat_strip_by_category[new] = self._compat_strip_by_category.pop(old)

        # Move lift-off compatibility-by-category if present
        if old in self._compat_liftoff_by_category:
            self._compat_liftoff_by_category[new] = self._compat_liftoff_by_category.pop(old)

        # Move clean compatibility-by-category if present
        if old in self._compat_clean_by_category:
            self._compat_clean_by_category[new] = self._compat_clean_by_category.pop(old)

        # Move timestamps for materials in this category
        changed_dates = False
        for mat in moved_materials:
            old_key = self._material_date_key(old, mat)
            new_key = self._material_date_key(new, mat)
            if old_key in self._material_created_at:
                self._material_created_at[new_key] = self._material_created_at.pop(old_key)
                changed_dates = True
        if changed_dates:
            self._save_material_created_at_map()

        self._reload_categories()
        self.cat_combo.setCurrentText(new)
        self._reload_comp_targets()
        self._reload_dep_targets()
        self._reload_bond_targets()
        self._reload_strip_targets()
        self._reload_liftoff_targets()
        self._reload_clean_targets()
        self._refresh_compat_material_tree()
        self._refresh_restrict_material_tree()

    def _delete_category(self):
        cat = self.cat_combo.currentText()
        if not cat:
            QMessageBox.warning(
                self,
                "Choose a category",
                "Select the category to delete.",
            )
            return
        if not self._confirm_delete(
            f"Delete {cat}?",
            "This deletes the category, its materials, and their compatibility "
            "and restriction settings.",
        ):
            return

        # Remove category
        removed_materials = list(self._library.get(cat, []))
        self._library.pop(cat, None)

        # Remove etch/deposit/bond/strip/clean compatibility-by-category entries
        self._compat_by_category.pop(cat, None)
        self._compat_deposit_by_category.pop(cat, None)
        self._compat_bond_by_category.pop(cat, None)
        self._compat_strip_by_category.pop(cat, None)
        self._compat_liftoff_by_category.pop(cat, None)
        self._compat_clean_by_category.pop(cat, None)

        # Remove per-material compatibility/restrictions for deleted materials.
        for mat in removed_materials:
            self._compat_by_material.pop(mat, None)
            self._compat_deposit_by_material.pop(mat, None)
            self._compat_bond_by_material.pop(mat, None)
            self._compat_strip_by_material.pop(mat, None)
            self._compat_liftoff_by_material.pop(mat, None)
            self._compat_clean_by_material.pop(mat, None)
            for mats in self._op_restrictions_by_operation.values():
                if isinstance(mats, list):
                    while mat in mats:
                        mats.remove(mat)

        # Remove timestamps for deleted category/materials
        changed_dates = False
        for mat in removed_materials:
            key = self._material_date_key(cat, mat)
            if key in self._material_created_at:
                self._material_created_at.pop(key, None)
                changed_dates = True
        if changed_dates:
            self._save_material_created_at_map()

        self._reload_categories()
        self._on_category_changed(self.cat_combo.currentText())
        self._reload_comp_targets()
        self._reload_dep_targets()
        self._reload_bond_targets()
        self._reload_strip_targets()
        self._reload_liftoff_targets()
        self._reload_clean_targets()
        self._reload_restrictions_for_operation(self._current_restriction_operation())
        self._refresh_compat_material_tree()
        self._refresh_restrict_material_tree()

    def _add_material(self):
        cat = self.cat_combo.currentText()
        if not cat:
            QMessageBox.warning(
                self,
                "Select a category",
                "Choose or add a category first.",
            )
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Add material")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Material name:"))
        name_edit = QLineEdit()
        name_edit.setPlaceholderText("e.g. Silicon")
        name_edit.setClearButtonEnabled(True)
        layout.addWidget(name_edit)
        layout.addWidget(QLabel("Short name (optional):"))
        short_edit = QLineEdit()
        short_edit.setPlaceholderText("e.g. Si")
        short_edit.setClearButtonEnabled(True)
        layout.addWidget(short_edit)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_ok = QPushButton("Add")
        btn_cancel = QPushButton("Cancel")
        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)
        name_edit.returnPressed.connect(dlg.accept)
        short_edit.returnPressed.connect(dlg.accept)
        name_edit.setFocus()

        if dlg.exec_() != QDialog.Accepted:
            return

        base_name = (name_edit.text() or "").strip()
        if not base_name:
            return
        short_name = (short_edit.text() or "").strip()
        # Accept "Si" or "(Si)" and normalize to "Si".
        if short_name.startswith("(") and short_name.endswith(")"):
            short_name = short_name[1:-1].strip()

        if short_name and base_name.endswith(")") and "(" in base_name:
            prefix, _, tail = base_name.rpartition("(")
            if prefix and tail.endswith(")"):
                cleaned = prefix.strip()
                if cleaned:
                    base_name = cleaned

        name = f"{base_name} ({short_name})" if short_name else base_name
        if name in self._library.get(cat, []):
            QMessageBox.warning(
                self,
                "Material already exists",
                f"{name} is already in {cat}.",
            )
            return
        self._library.setdefault(cat, []).append(name)
        self._material_created_at[self._material_date_key(cat, name)] = time.time()
        self._save_material_created_at_map()
        self._on_category_changed(cat)
        self._reload_comp_targets()
        self._reload_dep_targets()
        self._reload_bond_targets()
        self._reload_strip_targets()
        self._reload_liftoff_targets()
        self._reload_clean_targets()
        self._reload_restrictions_for_operation(self._current_restriction_operation())
        self._refresh_compat_material_tree()
        self._refresh_restrict_material_tree()

    def _rename_material(self):
        cat = self.cat_combo.currentText()
        if not cat:
            QMessageBox.warning(
                self,
                "Choose a category",
                "Select the category that contains the material.",
            )
            return
        item = self.material_list.currentItem()
        if item is None:
            QMessageBox.warning(
                self,
                "Choose a material",
                "Select the material to rename.",
            )
            return
        old = item.text()
        new, ok = QInputDialog.getText(self, "Rename material", "New name:", text=old)
        if not ok or not new.strip():
            return
        new = new.strip()
        if new == old:
            return
        lst = self._library.get(cat, [])
        if new in lst:
            QMessageBox.warning(
                self,
                "Material already exists",
                f"{new} is already in {cat}.",
            )
            return
        try:
            idx = lst.index(old)
        except ValueError:
            return
        lst[idx] = new

        old_key = self._material_date_key(cat, old)
        new_key = self._material_date_key(cat, new)
        if old_key in self._material_created_at:
            self._material_created_at[new_key] = self._material_created_at.pop(old_key)
        else:
            self._material_created_at[new_key] = time.time()
        self._save_material_created_at_map()

        # If material appears in compat_by_material (etch), move mapping
        if old in self._compat_by_material:
            self._compat_by_material[new] = self._compat_by_material.pop(old)

        # Deposit
        if old in self._compat_deposit_by_material:
            self._compat_deposit_by_material[new] = self._compat_deposit_by_material.pop(old)

        # Bond
        if old in self._compat_bond_by_material:
            self._compat_bond_by_material[new] = self._compat_bond_by_material.pop(old)

        # Strip
        if old in self._compat_strip_by_material:
            self._compat_strip_by_material[new] = self._compat_strip_by_material.pop(old)

        # Lift-off
        if old in self._compat_liftoff_by_material:
            self._compat_liftoff_by_material[new] = self._compat_liftoff_by_material.pop(old)

        # Clean
        if old in self._compat_clean_by_material:
            self._compat_clean_by_material[new] = self._compat_clean_by_material.pop(old)

        # Restrictions lists by operation
        for op, mats in self._op_restrictions_by_operation.items():
            if not isinstance(mats, list):
                continue
            for i, mat in enumerate(list(mats)):
                if mat == old:
                    mats[i] = new
            # Keep deterministic ordering and avoid duplicates on rename.
            uniq = []
            seen = set()
            for m in mats:
                if m in seen:
                    continue
                seen.add(m)
                uniq.append(m)
            self._op_restrictions_by_operation[op] = uniq

        self._on_category_changed(cat)
        self._reload_comp_targets()
        self._reload_dep_targets()
        self._reload_bond_targets()
        self._reload_strip_targets()
        self._reload_liftoff_targets()
        self._reload_clean_targets()
        self._reload_restrictions_for_operation(self._current_restriction_operation())
        self._refresh_compat_material_tree()
        self._refresh_restrict_material_tree()

    def _delete_material(self):
        cat = self.cat_combo.currentText()
        if not cat:
            QMessageBox.warning(
                self,
                "Choose a category",
                "Select the category that contains the material.",
            )
            return
        item = self.material_list.currentItem()
        if item is None:
            QMessageBox.warning(
                self,
                "Choose a material",
                "Select the material to delete.",
            )
            return
        name = item.text()
        if not self._confirm_delete(
            f"Delete {name}?",
            f"This removes the material from {cat} and clears its compatibility "
            "and restriction settings.",
        ):
            return
        lst = self._library.get(cat, [])
        try:
            lst.remove(name)
        except ValueError:
            pass

        key = self._material_date_key(cat, name)
        if key in self._material_created_at:
            self._material_created_at.pop(key, None)
            self._save_material_created_at_map()

        # Remove all compat for this material if it exists
        self._compat_by_material.pop(name, None)
        self._compat_deposit_by_material.pop(name, None)
        self._compat_bond_by_material.pop(name, None)
        self._compat_strip_by_material.pop(name, None)
        self._compat_liftoff_by_material.pop(name, None)
        self._compat_clean_by_material.pop(name, None)
        for mats in self._op_restrictions_by_operation.values():
            if isinstance(mats, list):
                while name in mats:
                    mats.remove(name)

        self._on_category_changed(cat)
        self._reload_comp_targets()
        self._reload_dep_targets()
        self._reload_bond_targets()
        self._reload_strip_targets()
        self._reload_liftoff_targets()
        self._reload_clean_targets()
        self._reload_restrictions_for_operation(self._current_restriction_operation())
        self._refresh_compat_material_tree()
        self._refresh_restrict_material_tree()

    # ------------------------------------------------------------------
    # Etch compatibility helpers
    # ------------------------------------------------------------------
    def _current_scope(self):
        txt = self.comp_scope_combo.currentText().lower()
        if "material" in txt:
            return "material"
        return "category"

    def _reload_comp_targets(self):
        """Populate the etch target combo depending on scope."""
        scope = self._current_scope()
        self.comp_target_combo.blockSignals(True)
        self.comp_target_combo.clear()

        if scope == "material":
            names = set()
            for lst in self._library.values():
                for n in lst:
                    names.add(n)
            for n in self._compat_by_material.keys():
                names.add(n)
            targets = sorted(names)
        else:
            cats = set(self._library.keys()) | set(self._compat_by_category.keys())
            targets = sorted(cats)

        q = self._compat_target_query()
        if q:
            targets = [t for t in targets if q in t.lower()]

        self.comp_target_combo.addItems(targets)
        self.comp_target_combo.blockSignals(False)

        self._reload_comp_etchants(self.comp_target_combo.currentText())

    def _reload_comp_etchants(self, target):
        self.comp_etch_list.clear()
        if not target:
            return
        scope = self._current_scope()
        if scope == "material":
            etchants = self._compat_by_material.get(target, [])
        else:
            etchants = self._compat_by_category.get(target, [])
        q = self._compat_entry_query()
        for e in etchants:
            if q and q not in str(e).lower():
                continue
            self.comp_etch_list.addItem(e)

    def _add_etchant(self):
        target = self.comp_target_combo.currentText()
        if not target:
            QMessageBox.warning(
                self,
                "Select a target",
                "Choose the material or category that will receive this etchant.",
            )
            return
        scope = self._current_scope()
        etch, ok = QInputDialog.getText(self, "Add etchant", "Etchant name:")
        if not ok or not etch.strip():
            return
        etch = etch.strip()

        if scope == "material":
            lst = self._compat_by_material.setdefault(target, [])
        else:
            lst = self._compat_by_category.setdefault(target, [])

        if etch in lst:
            QMessageBox.warning(
                self,
                "Etchant already assigned",
                f"{etch} is already available for {target}.",
            )
            return
        lst.append(etch)
        self._reload_comp_etchants(target)

    def _delete_etchant(self):
        target = self.comp_target_combo.currentText()
        if not target:
            QMessageBox.warning(
                self,
                "Choose a target",
                "Select the material or category to update.",
            )
            return
        item = self.comp_etch_list.currentItem()
        if item is None:
            QMessageBox.warning(
                self,
                "Choose an etchant",
                "Select the etchant to remove.",
            )
            return
        etch = item.text()
        scope = self._current_scope()

        if scope == "material":
            lst = self._compat_by_material.get(target, [])
        else:
            lst = self._compat_by_category.get(target, [])

        try:
            lst.remove(etch)
        except ValueError:
            pass

        self._reload_comp_etchants(target)

    # ------------------------------------------------------------------
    # Deposit compatibility helpers
    # ------------------------------------------------------------------
    def _current_dep_scope(self):
        txt = self.dep_scope_combo.currentText().lower()
        if "material" in txt:
            return "material"
        return "category"

    def _reload_dep_targets(self):
        """Populate the deposit target combo depending on scope."""
        scope = self._current_dep_scope()
        self.dep_target_combo.blockSignals(True)
        self.dep_target_combo.clear()

        if scope == "material":
            names = set()
            for lst in self._library.values():
                for n in lst:
                    names.add(n)
            for n in self._compat_deposit_by_material.keys():
                names.add(n)
            targets = sorted(names)
        else:
            cats = set(self._library.keys()) | set(self._compat_deposit_by_category.keys())
            targets = sorted(cats)

        q = self._compat_target_query()
        if q:
            targets = [t for t in targets if q in t.lower()]

        self.dep_target_combo.addItems(targets)
        self.dep_target_combo.blockSignals(False)

        self._reload_dep_methods(self.dep_target_combo.currentText())

    def _reload_dep_methods(self, target):
        self.dep_method_list.clear()
        if not target:
            return
        scope = self._current_dep_scope()
        if scope == "material":
            methods = self._compat_deposit_by_material.get(target, [])
        else:
            methods = self._compat_deposit_by_category.get(target, [])
        q = self._compat_entry_query()
        for m in methods:
            if q and q not in str(m).lower():
                continue
            self.dep_method_list.addItem(m)

    def _add_dep_method(self):
        target = self.dep_target_combo.currentText()
        if not target:
            QMessageBox.warning(
                self,
                "Select a target",
                "Choose the material or category that will receive this method.",
            )
            return

        scope = self._current_dep_scope()

        # Current methods already assigned to this target
        if scope == "material":
            current = self._compat_deposit_by_material.get(target, [])
        else:
            current = self._compat_deposit_by_category.get(target, [])

        # Methods still available to add
        available = [m for m in DEFAULT_DEPOSITION_METHODS if m not in current]
        if not available:
            QMessageBox.information(
                self,
                "All methods assigned",
                f"Every available deposition method is already assigned to {target}.",
            )
            return

        # Let the user choose one of the remaining methods
        method, ok = QInputDialog.getItem(
            self,
            "Add deposition method",
            "Select method:",
            available,
            0,
            False,
        )
        if not ok or not method:
            return

        if scope == "material":
            lst = self._compat_deposit_by_material.setdefault(target, [])
        else:
            lst = self._compat_deposit_by_category.setdefault(target, [])

        if method in lst:
            QMessageBox.warning(
                self,
                "Method already assigned",
                f"{method} is already available for {target}.",
            )
            return

        lst.append(method)
        self._reload_dep_methods(target)

    def _delete_dep_method(self):
        target = self.dep_target_combo.currentText()
        if not target:
            QMessageBox.warning(
                self,
                "Choose a target",
                "Select the material or category to update.",
            )
            return
        item = self.dep_method_list.currentItem()
        if item is None:
            QMessageBox.warning(
                self,
                "Choose a method",
                "Select the deposition method to remove.",
            )
            return
        method = item.text()
        scope = self._current_dep_scope()

        if scope == "material":
            lst = self._compat_deposit_by_material.get(target, [])
        else:
            lst = self._compat_deposit_by_category.get(target, [])

        try:
            lst.remove(method)
        except ValueError:
            pass

        self._reload_dep_methods(target)

    # ------------------------------------------------------------------
    # Bond compatibility helpers
    # ------------------------------------------------------------------
    def _current_bond_scope(self):
        txt = self.bond_scope_combo.currentText().lower()
        if "material" in txt:
            return "material"
        return "category"

    def _reload_bond_targets(self):
        scope = self._current_bond_scope()
        self.bond_target_combo.blockSignals(True)
        self.bond_target_combo.clear()

        if scope == "material":
            names = set()
            for lst in self._library.values():
                for n in lst:
                    names.add(n)
            for n in self._compat_bond_by_material.keys():
                names.add(n)
            targets = sorted(names)
        else:
            cats = set(self._library.keys()) | set(self._compat_bond_by_category.keys())
            targets = sorted(cats)

        q = self._compat_target_query()
        if q:
            targets = [t for t in targets if q in t.lower()]

        self.bond_target_combo.addItems(targets)
        self.bond_target_combo.blockSignals(False)

    def _reload_bond_methods(self, target):
        self.bond_method_list.clear()
        if not target:
            return
        scope = self._current_bond_scope()
        if scope == "material":
            methods = self._compat_bond_by_material.get(target, [])
        else:
            methods = self._compat_bond_by_category.get(target, [])
        q = self._compat_entry_query()
        for m in methods:
            if q and q not in str(m).lower():
                continue
            self.bond_method_list.addItem(m)

    def _add_bond_method(self):
        target = self.bond_target_combo.currentText()
        if not target:
            QMessageBox.warning(
                self,
                "Select a target",
                "Choose the material or category that will receive this method.",
            )
            return

        scope = self._current_bond_scope()
        text, ok = QInputDialog.getText(
            self,
            "Add bond method",
            "Bond method:",
        )
        if not ok or not text.strip():
            return
        text = text.strip()

        if scope == "material":
            lst = self._compat_bond_by_material.setdefault(target, [])
        else:
            lst = self._compat_bond_by_category.setdefault(target, [])

        if text in lst:
            QMessageBox.warning(
                self,
                "Method already assigned",
                f"{text} is already available for {target}.",
            )
            return

        lst.append(text)
        self._reload_bond_methods(target)

    def _delete_bond_method(self):
        target = self.bond_target_combo.currentText()
        if not target:
            QMessageBox.warning(
                self,
                "Choose a target",
                "Select the material or category to update.",
            )
            return
        item = self.bond_method_list.currentItem()
        if item is None:
            QMessageBox.warning(
                self,
                "Choose a method",
                "Select the bond method to remove.",
            )
            return
        text = item.text()
        scope = self._current_bond_scope()

        if scope == "material":
            lst = self._compat_bond_by_material.get(target, [])
        else:
            lst = self._compat_bond_by_category.get(target, [])

        try:
            lst.remove(text)
        except ValueError:
            pass

        self._reload_bond_methods(target)

    # ------------------------------------------------------------------
    # Strip compatibility helpers
    # ------------------------------------------------------------------
    def _current_strip_scope(self):
        txt = self.strip_scope_combo.currentText().lower()
        if "material" in txt:
            return "material"
        return "category"

    def _reload_strip_targets(self):
        scope = self._current_strip_scope()
        self.strip_target_combo.blockSignals(True)
        self.strip_target_combo.clear()

        if scope == "material":
            names = set()
            for lst in self._library.values():
                for n in lst:
                    names.add(n)
            for n in self._compat_strip_by_material.keys():
                names.add(n)
            targets = sorted(names)
        else:
            cats = set(self._library.keys()) | set(self._compat_strip_by_category.keys())
            targets = sorted(cats)

        q = self._compat_target_query()
        if q:
            targets = [t for t in targets if q in t.lower()]

        self.strip_target_combo.addItems(targets)
        self.strip_target_combo.blockSignals(False)

    def _reload_strip_methods(self, target):
        self.strip_method_list.clear()
        if not target:
            return
        scope = self._current_strip_scope()
        if scope == "material":
            methods = self._compat_strip_by_material.get(target, [])
        else:
            methods = self._compat_strip_by_category.get(target, [])
        q = self._compat_entry_query()
        for m in methods:
            if q and q not in str(m).lower():
                continue
            self.strip_method_list.addItem(m)

    def _add_strip_method(self):
        target = self.strip_target_combo.currentText()
        if not target:
            QMessageBox.warning(
                self,
                "Select a target",
                "Choose the material or category that will receive this method.",
            )
            return

        scope = self._current_strip_scope()
        text, ok = QInputDialog.getText(
            self,
            "Add strip method",
            "Strip method:",
        )
        if not ok or not text.strip():
            return
        text = text.strip()

        if scope == "material":
            lst = self._compat_strip_by_material.setdefault(target, [])
        else:
            lst = self._compat_strip_by_category.setdefault(target, [])

        if text in lst:
            QMessageBox.warning(
                self,
                "Method already assigned",
                f"{text} is already available for {target}.",
            )
            return

        lst.append(text)
        self._reload_strip_methods(target)

    def _delete_strip_method(self):
        target = self.strip_target_combo.currentText()
        if not target:
            QMessageBox.warning(
                self,
                "Choose a target",
                "Select the material or category to update.",
            )
            return
        item = self.strip_method_list.currentItem()
        if item is None:
            QMessageBox.warning(
                self,
                "Choose a method",
                "Select the strip method to remove.",
            )
            return
        text = item.text()
        scope = self._current_strip_scope()

        if scope == "material":
            lst = self._compat_strip_by_material.get(target, [])
        else:
            lst = self._compat_strip_by_category.get(target, [])

        try:
            lst.remove(text)
        except ValueError:
            pass

        self._reload_strip_methods(target)

    # ------------------------------------------------------------------
    # Lift-off compatibility helpers
    # ------------------------------------------------------------------
    def _current_liftoff_scope(self):
        txt = self.liftoff_scope_combo.currentText().lower()
        if "material" in txt:
            return "material"
        return "category"

    def _reload_liftoff_targets(self):
        scope = self._current_liftoff_scope()
        self.liftoff_target_combo.blockSignals(True)
        self.liftoff_target_combo.clear()

        if scope == "material":
            names = set()
            for lst in self._library.values():
                for n in lst:
                    names.add(n)
            for n in self._compat_liftoff_by_material.keys():
                names.add(n)
            targets = sorted(names)
        else:
            cats = set(self._library.keys()) | set(self._compat_liftoff_by_category.keys())
            targets = sorted(cats)

        q = self._compat_target_query()
        if q:
            targets = [t for t in targets if q in t.lower()]

        self.liftoff_target_combo.addItems(targets)
        self.liftoff_target_combo.blockSignals(False)

        self._reload_liftoff_methods(self.liftoff_target_combo.currentText())

    def _reload_liftoff_methods(self, target):
        self.liftoff_method_list.clear()
        if not target:
            return
        scope = self._current_liftoff_scope()
        if scope == "material":
            methods = self._compat_liftoff_by_material.get(target, [])
        else:
            methods = self._compat_liftoff_by_category.get(target, [])
        q = self._compat_entry_query()
        for m in methods:
            if q and q not in str(m).lower():
                continue
            self.liftoff_method_list.addItem(m)

    def _add_liftoff_method(self):
        target = self.liftoff_target_combo.currentText()
        if not target:
            QMessageBox.warning(
                self,
                "Select a target",
                "Choose the material or category that will receive this method.",
            )
            return

        scope = self._current_liftoff_scope()
        text, ok = QInputDialog.getText(
            self,
            "Add lift-off method",
            "Lift-off method:",
        )
        if not ok or not text.strip():
            return
        text = text.strip()

        if scope == "material":
            lst = self._compat_liftoff_by_material.setdefault(target, [])
        else:
            lst = self._compat_liftoff_by_category.setdefault(target, [])

        if text in lst:
            QMessageBox.warning(
                self,
                "Method already assigned",
                f"{text} is already available for {target}.",
            )
            return

        lst.append(text)
        self._reload_liftoff_methods(target)

    def _delete_liftoff_method(self):
        target = self.liftoff_target_combo.currentText()
        if not target:
            QMessageBox.warning(
                self,
                "Choose a target",
                "Select the material or category to update.",
            )
            return
        item = self.liftoff_method_list.currentItem()
        if item is None:
            QMessageBox.warning(
                self,
                "Choose a method",
                "Select the lift-off method to remove.",
            )
            return
        text = item.text()
        scope = self._current_liftoff_scope()

        if scope == "material":
            lst = self._compat_liftoff_by_material.get(target, [])
        else:
            lst = self._compat_liftoff_by_category.get(target, [])

        try:
            lst.remove(text)
        except ValueError:
            pass

        self._reload_liftoff_methods(target)

    # ------------------------------------------------------------------
    # Clean compatibility helpers
    # ------------------------------------------------------------------
    def _current_clean_scope(self):
        txt = self.clean_scope_combo.currentText().lower()
        if "material" in txt:
            return "material"
        return "category"

    def _reload_clean_targets(self):
        scope = self._current_clean_scope()
        self.clean_target_combo.blockSignals(True)
        self.clean_target_combo.clear()

        if scope == "material":
            names = set()
            for lst in self._library.values():
                for n in lst:
                    names.add(n)
            for n in self._compat_clean_by_material.keys():
                names.add(n)
            targets = sorted(names)
        else:
            cats = set(self._library.keys()) | set(self._compat_clean_by_category.keys())
            targets = sorted(cats)

        q = self._compat_target_query()
        if q:
            targets = [t for t in targets if q in t.lower()]

        self.clean_target_combo.addItems(targets)
        self.clean_target_combo.blockSignals(False)

    def _reload_clean_methods(self, target):
        self.clean_method_list.clear()
        if not target:
            return
        scope = self._current_clean_scope()
        if scope == "material":
            methods = self._compat_clean_by_material.get(target, [])
        else:
            methods = self._compat_clean_by_category.get(target, [])
        q = self._compat_entry_query()
        for m in methods:
            if q and q not in str(m).lower():
                continue
            self.clean_method_list.addItem(m)

    def _add_clean_method(self):
        target = self.clean_target_combo.currentText()
        if not target:
            QMessageBox.warning(
                self,
                "Select a target",
                "Choose the material or category that will receive this method.",
            )
            return

        scope = self._current_clean_scope()
        text, ok = QInputDialog.getText(
            self,
            "Add clean / surface prep method",
            "Clean or surface-preparation method:",
        )
        if not ok or not text.strip():
            return
        text = text.strip()

        if scope == "material":
            lst = self._compat_clean_by_material.setdefault(target, [])
        else:
            lst = self._compat_clean_by_category.setdefault(target, [])

        if text in lst:
            QMessageBox.warning(
                self,
                "Method already assigned",
                f"{text} is already available for {target}.",
            )
            return

        lst.append(text)
        self._reload_clean_methods(target)

    def _delete_clean_method(self):
        target = self.clean_target_combo.currentText()
        if not target:
            QMessageBox.warning(
                self,
                "Choose a target",
                "Select the material or category to update.",
            )
            return
        item = self.clean_method_list.currentItem()
        if item is None:
            QMessageBox.warning(
                self,
                "Choose a method",
                "Select the cleaning method to remove.",
            )
            return
        text = item.text()
        scope = self._current_clean_scope()

        if scope == "material":
            lst = self._compat_clean_by_material.get(target, [])
        else:
            lst = self._compat_clean_by_category.get(target, [])

        try:
            lst.remove(text)
        except ValueError:
            pass

        self._reload_clean_methods(target)

    # ------------------------------------------------------------------
    # Operation restrictions helpers
    # ------------------------------------------------------------------
    def _reload_restrictions_for_operation(self, operation):
        self.restrict_material_list.clear()
        if not operation:
            return
        mats = self._op_restrictions_by_operation.get(operation, [])
        if not isinstance(mats, list):
            return
        q = self._restrict_entry_query()
        for m in sorted(set(str(x) for x in mats), key=lambda s: s.lower()):
            if q and q not in m.lower():
                continue
            self.restrict_material_list.addItem(m)

    def _all_material_names(self):
        names = set()
        for lst in self._library.values():
            for n in lst:
                names.add(n)
        return sorted(names)

    def _add_operation_restriction(self):
        operation = self._current_restriction_operation()
        if not operation:
            return

        existing = set(self._op_restrictions_by_operation.get(operation, []))
        candidates = [n for n in self._all_material_names() if n not in existing]
        if not candidates:
            QMessageBox.information(
                self,
                "All materials restricted",
                f"Every material is already restricted for {operation_label(operation)}.",
            )
            return

        name, ok = QInputDialog.getItem(
            self,
            "Add restriction",
            "Material:",
            candidates,
            0,
            False,
        )
        if not ok or not name:
            return

        lst = self._op_restrictions_by_operation.setdefault(operation, [])
        if name not in lst:
            lst.append(name)
        self._reload_restrictions_for_operation(operation)

    def _delete_operation_restriction(self):
        operation = self._current_restriction_operation()
        if not operation:
            QMessageBox.warning(
                self,
                "Choose an operation",
                "Select the operation to update.",
            )
            return
        item = self.restrict_material_list.currentItem()
        if item is None:
            QMessageBox.warning(
                self,
                "Choose a restriction",
                "Select the material restriction to remove.",
            )
            return
        name = item.text()

        lst = self._op_restrictions_by_operation.get(operation, [])
        if not isinstance(lst, list):
            return
        try:
            lst.remove(name)
        except ValueError:
            pass
        self._reload_restrictions_for_operation(operation)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def _on_save(self):
        if self._save_callback is not None:
            self._save_callback()
        self.accept()
