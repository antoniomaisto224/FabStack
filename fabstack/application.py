"""Entry point for the FabStack application.

This module intentionally stays small. UI composition lives in the
``fabstack.presentation`` package and geometry lives in ``fabstack.domain``.
"""

import sys
import os

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

from fabstack.presentation.main_window import MainWindow
from fabstack.resources import APP_ICON_PATH
from fabstack.version import __version__


# ----------------- APPLICATION METADATA -----------------
__app_name__ = "FabStack"
__app_version__ = __version__
__app_author__ = "Antonio Maisto"
__app_description__ = "Fabrication process cross-section visualiser"


def main():
    """Create the Qt application, set the app icon, and show the main window."""
    QApplication.setAttribute(Qt.AA_DontUseNativeDialogs, True)
    app = QApplication(sys.argv)
    app.setApplicationName(__app_name__)
    app.setApplicationDisplayName(__app_name__)
    app.setApplicationVersion(__app_version__)

    # Global app icon (taskbar, title bar, etc.) – always from iconlogo.png
    if os.path.exists(APP_ICON_PATH):
        try:
            app.setWindowIcon(QIcon(APP_ICON_PATH))
        except Exception:
            # Icon is not critical; if something goes wrong, just continue
            pass

    window = MainWindow()
    window.showMaximized()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
