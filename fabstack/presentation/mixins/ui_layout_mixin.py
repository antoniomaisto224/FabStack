"""UI layout construction helpers for MainWindow."""

from PyQt5.QtCore import QRect, QSize, Qt, QTimer
from PyQt5.QtGui import QFont, QIcon, QPainter, QPainterPath, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStyle,
    QTabBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fabstack.operations.contracts import UI_OPERATION_KEYS
from fabstack.resources import ASSISTANT_ICON_PATH
from fabstack.terminology import operation_label


class UiLayoutMixin:
    def _build_main_ui(
        self,
        *,
        canvas_cls,
        preview_export_as_main_view: bool,
        left_panel_top_min_height_px: int,
        left_panel_top_max_height_px: int,
        height_um_default: int,
    ):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout()
        central.setLayout(main_layout)
        # Compact outer spacing
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(2)

        # Project tabs (Canva-style multi-canvas entry point).
        tabs_row = QHBoxLayout()
        tabs_row.setContentsMargins(4, 2, 4, 0)
        tabs_row.setSpacing(4)
        self.project_tabs = QTabBar()
        self.project_tabs.setDocumentMode(True)
        self.project_tabs.setMovable(True)
        self.project_tabs.setTabsClosable(False)
        self.project_tabs.setExpanding(False)
        self.project_tabs.setElideMode(Qt.ElideNone)
        self.project_tabs.setUsesScrollButtons(True)
        self.project_tabs.currentChanged.connect(self.on_project_tab_changed)
        self.project_tabs.tabCloseRequested.connect(self.on_project_tab_close_requested)
        self.project_tabs.tabMoved.connect(self.on_project_tab_moved)
        tabs_row.addWidget(self.project_tabs, 1)

        self.new_tab_button = QPushButton("+")
        self.new_tab_button.setObjectName("newTabButton")
        self.new_tab_button.setToolTip("New process")
        self.new_tab_button.setFlat(True)
        self.new_tab_button.setFixedSize(20, 20)
        self.new_tab_button.clicked.connect(self.new_process)
        tabs_row.addWidget(self.new_tab_button, 0, Qt.AlignVCenter)
        main_layout.addLayout(tabs_row, 0)

        # ---- TOP PANEL (no scrolling) ----
        self.top_panel = QWidget()
        controls_layout = QVBoxLayout(self.top_panel)
        # keep these tight so tabs/groupboxes sit flush under the title bar
        controls_layout.setContentsMargins(4, 2, 4, 2)
        controls_layout.setSpacing(2)

        if preview_export_as_main_view:
            self.top_panel.setMinimumHeight(left_panel_top_min_height_px)
            self.top_panel.setMaximumHeight(left_panel_top_max_height_px)

        # wafer group (compact, single row: substrate + thickness + reset)
        wafer_group = QGroupBox("Substrate")
        wafer_layout = QVBoxLayout()
        wafer_group.setLayout(wafer_layout)

        wafer_row = QHBoxLayout()
        wafer_row.setSpacing(4)

        self.substrate_combo = QComboBox()
        self.substrate_combo.addItem("Silicon (Si)")
        self.substrate_thick_spin = QSpinBox()
        self.substrate_thick_spin.setRange(10, height_um_default)
        self.substrate_thick_spin.setValue(200)
        self.reset_button = QPushButton("Reset process")
        self.reset_button.setObjectName("resetWaferButton")
        assistant_diameter = int(round(max(40, self.reset_button.sizeHint().height() + 6) * 1.15))
        self.process_assistant_button = QToolButton()
        self.process_assistant_button.setObjectName("processAssistantButton")
        self.process_assistant_button.setToolTip("Open Fabrix assistant")
        self.process_assistant_button.setIcon(self._process_assistant_icon(assistant_diameter))
        self.process_assistant_button.setIconSize(QSize(assistant_diameter - 2, assistant_diameter - 2))
        self.process_assistant_button.setAutoRaise(False)
        self.process_assistant_button.setFixedSize(assistant_diameter, assistant_diameter)
        self.process_assistant_button.setStyleSheet(
            f"""
            QToolButton#processAssistantButton {{
                border: 1px solid palette(mid);
                border-radius: {assistant_diameter // 2}px;
                background-color: palette(base);
                padding: 0px;
            }}
            QToolButton#processAssistantButton:hover {{
                border-color: palette(highlight);
                background-color: palette(alternate-base);
            }}
            QToolButton#processAssistantButton:pressed {{
                background-color: palette(midlight);
            }}
            """
        )
        self.flip_button = QPushButton("⇅")
        self.flip_button.setToolTip("Flip active stack")
        # Keep the flip glyph legible even in compact layouts.
        flip_font = QFont(self.flip_button.font())
        flip_font.setBold(True)
        if flip_font.pointSize() > 0:
            flip_font.setPointSize(flip_font.pointSize() + 3)
        self.flip_button.setFont(flip_font)

        sub_label = QLabel("Substrate:")
        th_label = QLabel("Thickness (µm):")
        for lab in (sub_label, th_label):
            lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        wafer_row.addWidget(sub_label)
        wafer_row.addWidget(self.substrate_combo)
        wafer_row.addWidget(th_label)
        wafer_row.addWidget(self.substrate_thick_spin)
        wafer_row.addWidget(self.flip_button)
        wafer_row.addWidget(self.reset_button)
        wafer_row.addStretch(1)

        wafer_layout.addLayout(wafer_row)

        # Materials group (buttons + live list).
        materials_group = QGroupBox("Materials")
        mat_layout = QVBoxLayout()
        materials_group.setLayout(mat_layout)

        # Buttons in a horizontal row
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(0)

        self.add_from_library_button = QPushButton("+")
        self.add_from_library_button.setObjectName("addMaterialButton")
        self.add_from_library_button.setToolTip("Add material")
        self.add_from_library_button.setFlat(True)
        self.add_from_library_button.setFixedSize(20, 20)
        self.delete_material_button = QPushButton("-")
        self.delete_material_button.setObjectName("deleteMaterialButton")
        self.delete_material_button.setToolTip("Remove selected material")
        self.delete_material_button.setFlat(True)
        self.delete_material_button.setFixedSize(26, 20)

        btn_row.addWidget(self.add_from_library_button)
        btn_row.addWidget(self.delete_material_button)
        btn_row.addStretch(1)

        # List of current materials
        self.materials_list_widget = QListWidget()
        self.materials_list_widget.setMinimumWidth(0)
        self.materials_list_widget.setSelectionMode(QListWidget.SingleSelection)
        self.materials_list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.materials_list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.materials_list_widget.setTextElideMode(Qt.ElideNone)
        self.materials_list_widget.setWordWrap(False)
        self.materials_list_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.materials_list_widget.setMinimumHeight(0)

        # Vertical arrangement for left-side panel usage
        mat_layout.addLayout(btn_row)
        mat_layout.addWidget(self.materials_list_widget, 1)

        # operation group
        op_group = QGroupBox("Process step")
        op_outer_layout = QVBoxLayout()
        op_group.setLayout(op_outer_layout)
        op_layout = QGridLayout()
        op_outer_layout.addLayout(op_layout)
        self.op_layout = op_layout  # store layout for dynamic repositioning

        self.op_combo = QComboBox(op_group)
        self.op_combo.addItems(list(UI_OPERATION_KEYS))
        self.op_list_widget = QListWidget()
        self.op_list_widget.setSelectionMode(QListWidget.SingleSelection)
        for i in range(self.op_combo.count()):
            self.op_list_widget.addItem(operation_label(self.op_combo.itemText(i)))
        self.op_list_widget.setCurrentRow(0)
        # Keep operation list compact (no large empty area below items).
        row_h = self.op_list_widget.sizeHintForRow(0)
        if row_h <= 0:
            row_h = 20
        op_list_h = row_h * self.op_list_widget.count() + (2 * self.op_list_widget.frameWidth()) + 4
        self.op_list_widget.setFixedHeight(op_list_h)

        self.material_combo = QComboBox()
        self.material_combo.addItem("Silicon (Si)")

        # thickness/depth are stored in µm. Minimum 0.001 µm = 1 nm.
        self.thick_spin = QDoubleSpinBox()
        self.thick_spin.setDecimals(3)
        self.thick_spin.setRange(0.001, height_um_default)
        self.thick_spin.setSingleStep(0.001)
        self.thick_spin.setValue(1.000)

        self.depth_spin = QDoubleSpinBox()
        self.depth_spin.setDecimals(3)
        self.depth_spin.setRange(0.001, height_um_default)
        self.depth_spin.setSingleStep(0.001)
        self.depth_spin.setValue(20.000)

        # Process temperature & exposure time (used in Method dialog)
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(-40.0, 400.0)
        self.temp_spin.setDecimals(1)
        self.temp_spin.setSuffix(" °C")
        self.temp_spin.setValue(0.0)

        self.exposure_spin = QDoubleSpinBox()
        self.exposure_spin.setRange(0.0, 10000.0)
        self.exposure_spin.setDecimals(1)
        self.exposure_spin.setSuffix(" s")
        self.exposure_spin.setValue(0.0)

        # Etchant/method combo (editable): display pretty, store ASCII in userData
        self.etchant_combo = QComboBox()
        self.etchant_combo.setEditable(True)
        self._populate_etchant_combo(self._etchants_from_library_or_default())

        # Etch-type checkbox (used in Method dialog, controls isotropic undercut)
        self.isotropic_etch_checkbox = QCheckBox("Isotropic etch")
        self.isotropic_etch_checkbox.setToolTip(
            "Etch laterally as well as vertically"
        )
        self.isotropic_etch_checkbox.setChecked(False)
        self.isotropic_etch_checkbox.toggled.connect(self.on_isotropic_checkbox_toggled)

        # Deposit method combo (for deposition operations only)
        self.deposit_method_combo = QComboBox()
        # Items will be populated dynamically by update_deposit_methods_for_material()

        # --- enforce comfortable minimum widths so they don't shrink too much ---
        self.material_combo.setFixedWidth(145)
        self.material_combo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.etchant_combo.setMinimumWidth(150)
        for sb in (self.thick_spin, self.depth_spin):
            sb.setFixedWidth(95)
            sb.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        for sb in (self.temp_spin, self.exposure_spin):
            sb.setMinimumWidth(70)

        # --- explicit labels ---
        self.material_label = QLabel("Material:")
        self.thickness_label = QLabel("Thickness (µm):")
        self.etch_depth_label = QLabel("Depth (µm):")
        self.temp_label = QLabel("Temperature (°C):")
        self.time_label = QLabel("Time (s):")
        self.material_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        for _lab in (self.thickness_label, self.etch_depth_label):
            _lab.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # Left operation panel: keep all operations visible in a list.
        op_layout.addWidget(self.op_list_widget, 0, 0, 1, 2)
        # Material row: independent from thickness/depth columns.
        self.material_row_widget = QWidget()
        material_row_layout = QHBoxLayout(self.material_row_widget)
        material_row_layout.setContentsMargins(0, 0, 0, 0)
        material_row_layout.setSpacing(2)
        material_row_layout.addWidget(self.material_label)
        material_row_layout.addWidget(self.material_combo)
        material_row_layout.addStretch(1)
        op_layout.addWidget(self.material_row_widget, 1, 0, 1, 2)

        self.lithography_invert_row_widget = QWidget()
        lithography_invert_layout = QHBoxLayout(self.lithography_invert_row_widget)
        lithography_invert_layout.setContentsMargins(0, 0, 0, 0)
        lithography_invert_layout.setSpacing(6)
        self.invert_lithography_checkbox = QCheckBox("Negative tone")
        self.invert_lithography_checkbox.setToolTip(
            "Keep resist inside each window and remove it elsewhere"
        )
        self.invert_lithography_checkbox.setChecked(False)
        lithography_invert_layout.addWidget(self.invert_lithography_checkbox)
        lithography_invert_layout.addStretch(1)
        op_layout.addWidget(self.lithography_invert_row_widget, 2, 0, 1, 2)

        # Thickness/Depth row: left-anchored and toggled by operation.
        self.dimension_row_widget = QWidget()
        dimension_row_layout = QHBoxLayout(self.dimension_row_widget)
        dimension_row_layout.setContentsMargins(0, 0, 0, 0)
        dimension_row_layout.setSpacing(1)
        dimension_row_layout.addWidget(self.thickness_label)
        dimension_row_layout.addWidget(self.thick_spin)
        dimension_row_layout.addWidget(self.etch_depth_label)
        dimension_row_layout.addWidget(self.depth_spin)
        dimension_row_layout.addStretch(1)
        op_layout.addWidget(self.dimension_row_widget, 3, 0, 1, 2)

        # Bond source row: shown only for Bond operation.
        self.bond_target_row_widget = QWidget()
        bond_target_layout = QHBoxLayout(self.bond_target_row_widget)
        bond_target_layout.setContentsMargins(0, 0, 0, 0)
        bond_target_layout.setSpacing(2)
        self.bond_target_button = QPushButton("Choose process…")
        self.bond_target_button.setObjectName("bondToButton")
        self.bond_target_file_label = QLabel("No process selected")
        self.bond_target_file_label.setToolTip("Choose a FabStack process to bond")
        self.bond_target_file_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        bond_target_layout.addWidget(self.bond_target_button)
        bond_target_layout.addWidget(self.bond_target_file_label, 1)
        bond_target_layout.addStretch(1)
        op_layout.addWidget(self.bond_target_row_widget, 4, 0, 1, 2)

        op_layout.setColumnStretch(0, 0)
        op_layout.setColumnStretch(1, 1)
        op_layout.setHorizontalSpacing(1)

        # Windowing widgets
        self.window_start_spin = QDoubleSpinBox()
        self.window_start_spin.setRange(0.0, 0.9)
        self.window_start_spin.setSingleStep(0.05)
        self.window_start_spin.setValue(0.35)

        self.window_end_spin = QDoubleSpinBox()
        self.window_end_spin.setRange(0.1, 1.0)
        self.window_end_spin.setSingleStep(0.05)
        self.window_end_spin.setValue(0.65)

        self.add_window_button = QPushButton("Add window")
        self.add_window_button.setObjectName("addWindowButton")
        self.add_window_button.setToolTip("Open the lithography window editor")
        self.clear_windows_button = QPushButton("Clear all")
        self.clear_windows_button.setObjectName("clearWindowsButton")
        self.clear_windows_button.setToolTip("Remove all lithography windows")
        self.window_list = QListWidget()

        # Text options controls (font size + font family + color, configured in dialog)
        self.text_size_spin = QSpinBox()
        self.text_size_spin.setRange(6, 40)
        self.text_size_spin.setValue(self.text_fontsize)

        self.material_font_size_spin = QSpinBox()
        self.material_font_size_spin.setRange(6, 40)
        self.material_font_size_spin.setValue(self.material_fontsize)

        # Windowing controls (shown when relevant) without a titled group box.
        self.windowing_group = QWidget()
        windowing_outer_layout = QVBoxLayout(self.windowing_group)
        windowing_outer_layout.setContentsMargins(0, 0, 0, 0)
        windowing_outer_layout.setSpacing(2)
        windowing_layout = QHBoxLayout()
        windowing_layout.setContentsMargins(0, 0, 0, 0)
        windowing_outer_layout.addLayout(windowing_layout)

        self.window_count_label = QLabel("Windows: 0")
        self.window_count_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        windowing_layout.addWidget(self.add_window_button)
        windowing_layout.addWidget(self.clear_windows_button)
        windowing_layout.addWidget(self.window_count_label)
        windowing_layout.addStretch(1)
        # Actions controls (without group box)
        actions_widget = QWidget()
        actions_layout = QHBoxLayout(actions_widget)
        actions_layout.setContentsMargins(0, 4, 0, 0)
        self.apply_button = QPushButton("Apply step")
        self.apply_button.setObjectName("applyStepButton")
        self.op_params_button = QPushButton("Step settings…")
        self.op_params_button.setObjectName("methodButton")

        # Put step settings before the primary action.
        actions_layout.addWidget(self.op_params_button)
        actions_layout.addWidget(self.apply_button)
        actions_layout.addStretch(1)

        self.history_list = QListWidget()
        self.history_list.setMinimumHeight(0)
        self.adjust_visual_thickness_button = QPushButton("Adjust display height…")
        self.adjust_visual_thickness_button.setObjectName("adjustVisualThicknessButton")
        self.adjust_visual_thickness_button.setToolTip(
            "Adjust one displayed layer or etched opening; physical dimensions stay unchanged"
        )

        # Move Windowing + Actions below Operation panel.
        op_outer_layout.addWidget(self.windowing_group)
        op_outer_layout.addWidget(actions_widget)
        op_outer_layout.addStretch(1)

        # Top panel: compact process controls remain here.
        top_row = QHBoxLayout()
        top_row.setSpacing(2)
        top_row.addWidget(wafer_group, 0)
        top_row.addSpacing(10)
        top_row.addWidget(self.adjust_visual_thickness_button, 0, Qt.AlignVCenter)
        top_row.addSpacing(10)
        top_row.addWidget(self.process_assistant_button, 0, Qt.AlignVCenter)
        top_row.addStretch(1)
        controls_layout.addLayout(top_row)

        # Left side: from left to right -> Materials panel, Operations panel.
        materials_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        op_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.windowing_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        actions_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # Start with a narrower materials panel (20% smaller than previous 210 px default).
        materials_group.setMinimumWidth(168)
        op_group.setMinimumWidth(250)

        # ---- compact button widths (text width + padding) ----
        def _compact_width(btn):
            fm = btn.fontMetrics()
            text_w = fm.horizontalAdvance(btn.text())
            target = int(text_w * 1.10) + 16
            btn.setMinimumWidth(target)
            btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        _compact_width(self.reset_button)
        _compact_width(self.flip_button)
        _compact_width(self.apply_button)
        _compact_width(self.op_params_button)
        _compact_width(self.adjust_visual_thickness_button)
        _compact_width(self.bond_target_button)
        _compact_width(self.add_window_button)
        _compact_width(self.clear_windows_button)

        # Keep the settings label fully visible.
        self.op_params_button.setMinimumWidth(self.op_params_button.minimumWidth() + 10)
        # Single-glyph labels can become too narrow with compact sizing.
        self.flip_button.setMinimumWidth(max(56, self.flip_button.minimumWidth()))

        # right area: top controls + workspace
        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        right_layout.setContentsMargins(0, 0, 0, 10)
        right_layout.setSpacing(0)

        # right figure area
        self.canvas = canvas_cls(self)
        self.figure_scroll = QScrollArea()

        if preview_export_as_main_view:
            self.figure_scroll.setWidgetResizable(False)
            self.figure_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            self.figure_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        else:
            self.figure_scroll.setWidgetResizable(False)
            self.figure_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            self.figure_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.figure_scroll.setWidget(self.canvas)
        right_layout.addWidget(self.top_panel, 0)
        right_layout.addWidget(self.figure_scroll, 1)

        # Process flow embedded below both left panels.
        process_flow_group = QGroupBox("Process steps")
        process_flow_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        process_flow_layout = QVBoxLayout(process_flow_group)
        process_flow_layout.addWidget(self.history_list, 1)

        # Left top area without a draggable separator: Materials | Operations.
        mat_w = materials_group.minimumWidth()
        op_w = op_group.minimumWidth()
        self._startup_material_panel_width = int(mat_w)
        self._startup_operations_panel_width = int(op_w)
        self.materials_group = materials_group
        self.operations_group = op_group
        self.materials_group.setFixedWidth(mat_w)
        self.materials_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.operations_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.left_top_row = QWidget()
        left_top_row_layout = QHBoxLayout(self.left_top_row)
        left_top_row_layout.setContentsMargins(0, 0, 0, 0)
        left_top_row_layout.setSpacing(2)
        left_top_row_layout.addWidget(self.materials_group, 0)
        left_top_row_layout.addWidget(self.operations_group, 1)

        # Left column: top panels + process flow panel.
        left_column = QWidget()
        left_layout = QVBoxLayout(left_column)
        left_layout.setContentsMargins(0, 0, 0, 10)
        left_layout.setSpacing(2)
        left_layout.addWidget(self.left_top_row, 0)
        left_layout.addWidget(process_flow_group, 1)

        # Main horizontal placement with draggable edges:
        # Left column (Materials + Operation + Process flow) | Right workspace
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.addWidget(left_column)
        self.main_splitter.addWidget(right_column)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        # Open with the left column near its minimum width to avoid manual shrinking each launch.
        left_min_w = mat_w + op_w + 12
        self.main_splitter.setSizes([left_min_w, 1200])
        # Keep stack viewport synchronized while dragging splitter handles.
        self.main_splitter.splitterMoved.connect(self.on_main_splitter_moved)
        main_layout.addWidget(self.main_splitter, 1)

    def _connect_main_ui_signals(self):
        # connections
        self.reset_button.clicked.connect(self.reset_wafer)
        self.flip_button.clicked.connect(self.flip_active_stack_from_top)
        self.process_assistant_button.clicked.connect(self.open_process_assistant_chat)
        self.substrate_combo.currentIndexChanged.connect(lambda _index: self.on_substrate_changed())
        self.substrate_thick_spin.editingFinished.connect(self.on_substrate_changed)
        # materials (panel)
        self.add_from_library_button.clicked.connect(self.add_material_from_library)
        self.delete_material_button.clicked.connect(self.on_delete_material_button_clicked)
        self.materials_list_widget.customContextMenuRequested.connect(self.on_material_context_menu)

        # process actions
        self.apply_button.clicked.connect(self.apply_operation)
        self.op_params_button.clicked.connect(self.open_operation_parameters_dialog)
        self.adjust_visual_thickness_button.clicked.connect(self.open_visual_thickness_dialog)

        self.op_list_widget.currentRowChanged.connect(self.on_op_list_row_changed)
        self.op_combo.currentIndexChanged.connect(self.sync_op_list_with_combo)
        self.op_combo.currentIndexChanged.connect(self.update_op_controls)
        self.op_combo.currentIndexChanged.connect(self.update_deposit_methods_for_material)

        self.history_list.itemDoubleClicked.connect(self.rename_step)
        self.history_list.currentRowChanged.connect(self.on_history_row_changed)

        self.add_window_button.clicked.connect(self.open_windowing_dialog)
        self.clear_windows_button.clicked.connect(self.clear_litho_windows)
        self.bond_target_button.clicked.connect(self.select_bond_target_file)

        self.text_size_spin.valueChanged.connect(self.on_text_size_spin_changed)
        self.material_font_size_spin.valueChanged.connect(self.on_material_text_size_spin_changed)

        # React to etchant changes for etch-mode auto selection
        self.etchant_combo.currentIndexChanged.connect(lambda _: self.on_etchant_changed())

        # keep Etchant/Method filtered to selected material where relevant
        self.material_combo.currentTextChanged.connect(self.update_etchant_options)
        self.material_combo.currentTextChanged.connect(self.update_deposit_methods_for_material)
        self.material_combo.currentTextChanged.connect(self.on_operation_material_changed)
        self.invert_lithography_checkbox.toggled.connect(self.on_invert_lithography_toggled)

        # Enforce operation-specific material restrictions in the selector.
        self.refresh_operation_material_combo()

        # adjust Deposit UI when method changes (Spincoating -> RPM, others -> degC)
        self.deposit_method_combo.currentTextChanged.connect(self.update_deposit_method_ui)

    def _initialize_main_ui_state(self, *, preview_export_as_main_view: bool):
        # init
        self.xsec = None
        self.reset_wafer()
        self.update_op_controls()
        if preview_export_as_main_view:
            self._apply_preview_left_panel_width()
            self._apply_export_preview_canvas_aspect()

        self.refresh_materials_list()
        self.refresh_recent_materials_list()
        # Initialize etch mode based on current etchant (if any)
        self.on_etchant_changed()
        # Ensure expert-only elements start disabled
        self.update_expert_user_controls()
        self._update_visualisation_controls_state()
        self._initialize_project_tabs_from_current_state()
        self._apply_autosave_enabled_state()
        # Re-apply startup panel sizes after final layout pass (avoids Qt ratio expansion).
        QTimer.singleShot(0, self._apply_startup_splitter_sizes)

    def _apply_startup_splitter_sizes(self):
        if not hasattr(self, "main_splitter"):
            return
        mat_w = int(getattr(self, "_startup_material_panel_width", 168))
        op_w = int(getattr(self, "_startup_operations_panel_width", 250))

        if hasattr(self, "materials_group"):
            self.materials_group.setFixedWidth(mat_w)

        main_total = self.main_splitter.width()
        if main_total > 0:
            left_w = mat_w + op_w + 12
            right_w = max(320, main_total - left_w - self.main_splitter.handleWidth())
            self.main_splitter.setSizes([left_w, right_w])

    def _process_assistant_icon(self, diameter: int) -> QIcon:
        source = QPixmap(ASSISTANT_ICON_PATH)
        if source.isNull():
            return self.style().standardIcon(QStyle.SP_MessageBoxQuestion)

        source_side = min(source.width(), source.height())
        crop_side = max(1, int(source_side * 0.76))
        crop_left = max(0, (source.width() - crop_side) // 2)
        crop_top = max(0, (source.height() - crop_side) // 2)
        cropped = source.copy(QRect(crop_left, crop_top, crop_side, crop_side))

        target_side = max(96, int(diameter) * 4)
        target = QPixmap(target_side, target_side)
        target.fill(Qt.transparent)

        painter = QPainter(target)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        path = QPainterPath()
        path.addEllipse(0, 0, target_side, target_side)
        painter.setClipPath(path)
        painter.drawPixmap(
            0,
            0,
            cropped.scaled(
                QSize(target_side, target_side),
                Qt.IgnoreAspectRatio,
                Qt.SmoothTransformation,
            ),
        )
        painter.end()
        return QIcon(target)
