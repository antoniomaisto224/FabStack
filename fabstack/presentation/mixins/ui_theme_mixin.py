"""UI theme helpers for MainWindow."""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QIcon, QPainter, QPalette, QPixmap
from PyQt5.QtWidgets import QApplication, QProxyStyle


class DarkThemeIconProxyStyle(QProxyStyle):
    """Recolor only low-contrast monochrome standard icons on the dark theme."""

    def __init__(self, base_style):
        super().__init__(base_style)
        self._icon_color = QColor("#e8ecf3")
        self._disabled_icon_color = QColor("#6f7a8e")

    def _luminance(self, color: QColor) -> float:
        return (
            (0.2126 * color.red())
            + (0.7152 * color.green())
            + (0.0722 * color.blue())
        )

    def _needs_dark_theme_tint(self, pixmap: QPixmap) -> bool:
        if pixmap is None or pixmap.isNull():
            return False

        image = pixmap.toImage()
        visible = 0
        dark = 0
        bright = 0
        colorful = 0
        lum_sum = 0.0

        for y in range(image.height()):
            for x in range(image.width()):
                color = image.pixelColor(x, y)
                if color.alpha() < 24:
                    continue

                visible += 1
                lum = self._luminance(color)
                lum_sum += lum
                if lum < 95:
                    dark += 1
                if lum > 145:
                    bright += 1
                if max(color.red(), color.green(), color.blue()) - min(
                    color.red(), color.green(), color.blue()
                ) > 35:
                    colorful += 1

        if visible == 0:
            return False

        coverage = visible / (image.width() * image.height())
        avg_lum = lum_sum / visible
        dark_ratio = dark / visible
        bright_ratio = bright / visible
        colorful_ratio = colorful / visible

        return (
            avg_lum < 105
            and dark_ratio > 0.55
            and bright_ratio < 0.18
            and colorful_ratio < 0.08
            and coverage < 0.45
        )

    def _tint_pixmap(self, pixmap: QPixmap, color: QColor) -> QPixmap:
        tinted = QPixmap(pixmap.size())
        tinted.setDevicePixelRatio(pixmap.devicePixelRatio())
        tinted.fill(Qt.transparent)

        painter = QPainter(tinted)
        painter.drawPixmap(0, 0, pixmap)
        painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        painter.fillRect(tinted.rect(), color)
        painter.end()
        return tinted

    def _fix_pixmap(self, pixmap: QPixmap, mode=QIcon.Normal) -> QPixmap:
        if mode == QIcon.Disabled:
            return pixmap
        if not self._needs_dark_theme_tint(pixmap):
            return pixmap
        return self._tint_pixmap(pixmap, self._icon_color)

    def _fix_icon(self, icon: QIcon) -> QIcon:
        if icon is None or icon.isNull():
            return icon

        fixed_icon = QIcon()
        modes = (QIcon.Normal, QIcon.Disabled, QIcon.Active, QIcon.Selected)
        states = (QIcon.Off, QIcon.On)
        sizes = []

        for mode in modes:
            for state in states:
                sizes.extend(icon.availableSizes(mode, state))

        if not sizes:
            sizes = [16, 20, 24, 32, 48]
        else:
            unique_sizes = {}
            for size in sizes:
                unique_sizes[(size.width(), size.height())] = size
            sizes = list(unique_sizes.values())

        for mode in modes:
            for state in states:
                for size in sizes:
                    if isinstance(size, int):
                        pixmap = icon.pixmap(size, size, mode, state)
                    else:
                        pixmap = icon.pixmap(size, mode, state)
                    if not pixmap.isNull():
                        fixed_icon.addPixmap(self._fix_pixmap(pixmap, mode), mode, state)

        return fixed_icon if not fixed_icon.isNull() else icon

    def standardIcon(self, standard_icon, option=None, widget=None):
        return self._fix_icon(super().standardIcon(standard_icon, option, widget))

    def standardPixmap(self, standard_pixmap, option=None, widget=None):
        return self._fix_pixmap(super().standardPixmap(standard_pixmap, option, widget))


class UiThemeMixin:
    def _build_light_palette(self) -> QPalette:
        palette = QPalette()
        bg = QColor("#f5f7fb")
        panel = QColor("#ffffff")
        surface = QColor("#ffffff")
        text = QColor("#172033")
        muted = QColor("#5f6f85")
        accent = QColor("#315fce")
        disabled = QColor("#9aa6b8")

        palette.setColor(QPalette.Window, bg)
        palette.setColor(QPalette.WindowText, text)
        palette.setColor(QPalette.Base, surface)
        palette.setColor(QPalette.AlternateBase, QColor("#eef2f8"))
        palette.setColor(QPalette.ToolTipBase, panel)
        palette.setColor(QPalette.ToolTipText, text)
        palette.setColor(QPalette.Text, text)
        palette.setColor(QPalette.Button, panel)
        palette.setColor(QPalette.ButtonText, text)
        palette.setColor(QPalette.BrightText, QColor("#ffffff"))
        palette.setColor(QPalette.Highlight, accent)
        palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
        palette.setColor(QPalette.Link, accent)
        palette.setColor(QPalette.PlaceholderText, muted)

        for role in (
            QPalette.WindowText,
            QPalette.Text,
            QPalette.ButtonText,
            QPalette.ToolTipText,
            QPalette.HighlightedText,
        ):
            palette.setColor(QPalette.Disabled, role, disabled)
        palette.setColor(QPalette.Disabled, QPalette.Button, QColor("#eef2f8"))
        palette.setColor(QPalette.Disabled, QPalette.Base, QColor("#eef2f8"))
        return palette

    def _build_dark_palette(self) -> QPalette:
        palette = QPalette()
        bg = QColor("#10141f")
        panel = QColor("#171d2a")
        surface = QColor("#1d2534")
        text = QColor("#eef3fb")
        muted = QColor("#a9b5c8")
        accent = QColor("#7aa2ff")
        disabled = QColor("#6f7a8e")

        palette.setColor(QPalette.Window, bg)
        palette.setColor(QPalette.WindowText, text)
        palette.setColor(QPalette.Base, surface)
        palette.setColor(QPalette.AlternateBase, QColor("#141a26"))
        palette.setColor(QPalette.ToolTipBase, panel)
        palette.setColor(QPalette.ToolTipText, text)
        palette.setColor(QPalette.Text, text)
        palette.setColor(QPalette.Button, panel)
        palette.setColor(QPalette.ButtonText, text)
        palette.setColor(QPalette.BrightText, text)
        palette.setColor(QPalette.Highlight, accent)
        palette.setColor(QPalette.HighlightedText, text)
        palette.setColor(QPalette.Link, accent)
        palette.setColor(QPalette.PlaceholderText, muted)

        for role in (
            QPalette.WindowText,
            QPalette.Text,
            QPalette.ButtonText,
            QPalette.ToolTipText,
            QPalette.HighlightedText,
        ):
            palette.setColor(QPalette.Disabled, role, disabled)
        palette.setColor(QPalette.Disabled, QPalette.Button, panel)
        palette.setColor(QPalette.Disabled, QPalette.Base, surface)
        return palette

    def _is_dark_color(self, c: QColor) -> bool:
        luminance = (0.2126 * c.red()) + (0.7152 * c.green()) + (0.0722 * c.blue())
        return luminance < 128.0

    def _is_dark_palette(self) -> bool:
        return self._is_dark_color(self.palette().color(QPalette.Window))

    def _build_modern_stylesheet(self, dark: bool) -> str:
        if dark:
            colors = {
                "bg": "#10141f",
                "panel": "#171d2a",
                "panel_alt": "#1b2332",
                "surface": "#1d2534",
                "surface_alt": "#232c3e",
                "border": "#2f3a4d",
                "border_strong": "#43506a",
                "text": "#eef3fb",
                "muted": "#a9b5c8",
                "accent": "#7aa2ff",
                "accent_hover": "#8eb1ff",
                "accent_soft": "#26385f",
                "hover": "#222b3d",
                "pressed": "#1a2230",
                "danger": "#ff7b7b",
                "danger_soft": "#402832",
                "success": "#66dda0",
                "success_soft": "#213d33",
                "disabled": "#6f7a8e",
                "scroll": "#3b465a",
            }
        else:
            colors = {
                "bg": "#f5f7fb",
                "panel": "#ffffff",
                "panel_alt": "#f8fafd",
                "surface": "#ffffff",
                "surface_alt": "#f2f5fa",
                "border": "#d9e1ee",
                "border_strong": "#b8c5d8",
                "text": "#172033",
                "muted": "#5f6f85",
                "accent": "#315fce",
                "accent_hover": "#234eb7",
                "accent_soft": "#e5edff",
                "hover": "#eef3fb",
                "pressed": "#e3eaf6",
                "danger": "#b42318",
                "danger_soft": "#fff0ee",
                "success": "#16835a",
                "success_soft": "#e6f6ef",
                "disabled": "#9aa6b8",
                "scroll": "#b8c5d8",
            }

        return """
        QMainWindow, QDialog, QMessageBox, QFileDialog, QColorDialog, QFontDialog, QInputDialog {
            background-color: %(bg)s;
        }
        QWidget {
            color: %(text)s;
            selection-background-color: %(accent)s;
            selection-color: #ffffff;
            font-size: 12px;
        }
        QLabel, QCheckBox, QRadioButton, QStatusBar, QDialog QLabel, QMessageBox QLabel {
            color: %(text)s;
        }
        QGroupBox {
            background-color: %(panel)s;
            border: 1px solid %(border)s;
            border-radius: 10px;
            margin-top: 14px;
            padding: 10px;
            font-weight: 600;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
            color: %(muted)s;
            background-color: %(bg)s;
        }
        QListWidget, QTreeWidget, QTableWidget, QTableView, QTreeView, QTextEdit, QPlainTextEdit,
        QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
            background-color: %(surface)s;
            border: 1px solid %(border)s;
            border-radius: 7px;
            padding: 5px 7px;
            color: %(text)s;
        }
        QListWidget, QTreeWidget, QTableWidget, QTableView, QTreeView {
            alternate-background-color: %(panel_alt)s;
        }
        QListWidget::item, QTreeWidget::item {
            border-radius: 5px;
            padding: 4px 6px;
            margin: 1px 2px;
        }
        QListWidget::item:hover, QTreeWidget::item:hover {
            background-color: %(hover)s;
        }
        QListWidget::item:selected, QTreeWidget::item:selected {
            background-color: %(accent_soft)s;
            color: %(text)s;
        }
        QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus,
        QListWidget:focus, QTreeWidget:focus, QTableWidget:focus, QTextEdit:focus, QPlainTextEdit:focus {
            border-color: %(accent)s;
        }
        QLineEdit[echoMode="2"] {
            color: %(text)s;
        }
        QComboBox::drop-down {
            border: none;
            width: 22px;
        }
        QComboBox QAbstractItemView {
            color: %(text)s;
            background-color: %(surface)s;
            selection-background-color: %(accent)s;
            selection-color: #ffffff;
        }
        QHeaderView::section {
            background-color: %(panel_alt)s;
            color: %(text)s;
            border: 1px solid %(border)s;
            padding: 4px;
        }
        QAbstractItemView {
            background-color: %(surface)s;
            border: 1px solid %(border)s;
            outline: none;
            color: %(text)s;
            selection-background-color: %(accent)s;
            selection-color: #ffffff;
        }
        QTabWidget::pane {
            border: 1px solid %(border)s;
            border-radius: 8px;
            background-color: %(panel)s;
            top: -1px;
        }
        QTabBar::tab {
            background: transparent;
            color: %(muted)s;
            padding: 7px 30px 7px 14px;
            min-width: 112px;
            border: 1px solid transparent;
            margin: 2px 2px 0 0;
            border-radius: 8px;
            font-weight: 600;
        }
        QTabBar::tab:selected {
            color: %(text)s;
            border-color: %(border)s;
            background: %(panel)s;
        }
        QTabBar::tab:hover {
            color: %(text)s;
            background: %(hover)s;
        }
        QPushButton, QToolButton {
            border: 1px solid %(border)s;
            border-radius: 7px;
            background-color: %(panel)s;
            color: %(text)s;
            padding: 5px 11px;
            font-weight: 600;
        }
        QPushButton:hover, QToolButton:hover {
            border-color: %(border_strong)s;
            background-color: %(hover)s;
        }
        QPushButton:pressed, QToolButton:pressed {
            background-color: %(pressed)s;
        }
        QPushButton:disabled, QToolButton:disabled {
            color: %(disabled)s;
            border-color: %(border)s;
            background-color: %(panel_alt)s;
        }
        QToolButton {
            padding: 3px;
        }
        QPushButton#newTabButton, QPushButton#addMaterialButton, QPushButton#deleteMaterialButton {
            border: 1px solid transparent;
            border-radius: 10px;
            background-color: transparent;
            color: %(muted)s;
            padding: 0px;
            font-size: 18px;
            font-weight: 700;
        }
        QPushButton#newTabButton:hover, QPushButton#addMaterialButton:hover, QPushButton#deleteMaterialButton:hover {
            background-color: %(accent_soft)s;
            color: %(accent)s;
            border-color: %(border)s;
        }
        QPushButton#simpleTabCloseButton {
            border: none;
            border-radius: 8px;
            background-color: transparent;
            color: %(muted)s;
            padding: 0px;
            font-size: 13px;
            font-weight: 400;
        }
        QPushButton#simpleTabCloseButton:hover {
            background-color: %(hover)s;
            color: %(text)s;
        }
        QPushButton#resetWaferButton, QPushButton#applyStepButton {
            border-radius: 7px;
            padding: 5px 11px;
        }
        QPushButton#resetWaferButton {
            color: %(danger)s;
            background-color: %(danger_soft)s;
            border-color: transparent;
            font-weight: 700;
        }
        QPushButton#resetWaferButton:hover {
            background-color: %(danger_soft)s;
            border-color: %(danger)s;
        }
        QPushButton#applyStepButton {
            color: #ffffff;
            background-color: %(accent)s;
            border-color: %(accent)s;
            font-weight: 700;
        }
        QPushButton#applyStepButton:hover {
            background-color: %(accent_hover)s;
            border-color: %(accent_hover)s;
        }
        QPushButton#methodButton, QPushButton#addWindowButton, QPushButton#clearWindowsButton, QPushButton#bondToButton, QPushButton#adjustVisualThicknessButton {
            border: 1px solid transparent;
            border-radius: 6px;
            background: transparent;
            color: %(muted)s;
            padding: 3px 4px;
            text-align: left;
        }
        QPushButton#methodButton:hover, QPushButton#addWindowButton:hover, QPushButton#clearWindowsButton:hover, QPushButton#bondToButton:hover, QPushButton#adjustVisualThicknessButton:hover {
            color: %(accent)s;
            background: %(accent_soft)s;
            border-color: transparent;
        }
        QCheckBox::indicator {
            width: 14px;
            height: 14px;
            border: 1px solid %(border_strong)s;
            border-radius: 4px;
            background-color: %(surface)s;
        }
        QCheckBox::indicator:checked {
            background-color: %(accent)s;
            border-color: %(accent)s;
        }
        QScrollArea {
            border: none;
            background-color: transparent;
        }
        QSplitter::handle {
            background: %(bg)s;
        }
        QSplitter::handle:hover {
            background: %(border)s;
        }
        QScrollBar:vertical, QScrollBar:horizontal {
            background: transparent;
            border: none;
            margin: 2px;
        }
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
            background: %(scroll)s;
            border-radius: 5px;
            min-height: 24px;
            min-width: 24px;
        }
        QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
            background: %(accent)s;
        }
        QScrollBar::add-line, QScrollBar::sub-line {
            width: 0px;
            height: 0px;
            border: none;
            background: transparent;
        }
        QMenuBar {
            background-color: %(panel)s;
            border-bottom: 1px solid %(border)s;
            color: %(text)s;
        }
        QMenuBar::item {
            background: transparent;
            padding: 4px 8px;
            color: %(text)s;
        }
        QMenuBar::item:selected {
            background: %(hover)s;
            border-radius: 4px;
        }
        QMenu {
            background-color: %(panel)s;
            border: 1px solid %(border)s;
            border-radius: 8px;
            color: %(text)s;
            padding: 4px;
        }
        QMenu::item {
            padding: 7px 22px;
            border-radius: 5px;
            color: %(text)s;
        }
        QMenu::item:selected {
            background: %(accent)s;
            color: #ffffff;
        }
        QDialogButtonBox QPushButton {
            border: 1px solid %(border)s;
            border-radius: 7px;
            background-color: %(panel)s;
            color: %(text)s;
            padding: 6px 12px;
            min-width: 72px;
        }
        QToolTip {
            color: %(text)s;
            background-color: %(panel)s;
            border: 1px solid %(border)s;
            padding: 3px 6px;
        }
        """ % colors

    def _apply_modern_theme(self):
        app = QApplication.instance()
        dark = True
        if app is not None:
            dark = self._is_dark_color(app.palette().color(QPalette.Window))
            app.setStyle("Fusion")
            app.setPalette(self._build_dark_palette() if dark else self._build_light_palette())
            if dark:
                proxy = DarkThemeIconProxyStyle(app.style())
                app.setStyle(proxy)
                app._fabstack_dark_icon_style = proxy
            elif hasattr(app, "_fabstack_dark_icon_style"):
                app._fabstack_dark_icon_style = None
        qss = self._build_modern_stylesheet(dark)
        if app is not None:
            app.setStyleSheet(qss)
        else:
            self.setStyleSheet(qss)
