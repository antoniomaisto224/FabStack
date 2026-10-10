"""Read-only bundle resources and writable per-user application data."""

import os
import shutil
import sys

from fabstack.resource_paths import resource_path


def user_data_dir() -> str:
    """Return FabStack's writable application-data directory.

    ``FABSTACK_DATA_DIR`` is primarily useful for tests and managed
    deployments.  Production defaults follow the native per-user convention
    on each supported platform.
    """
    override = os.environ.get("FABSTACK_DATA_DIR", "").strip()
    if override:
        return os.path.abspath(os.path.expanduser(override))
    if sys.platform == "darwin":
        root = os.path.expanduser("~/Library/Application Support")
    elif os.name == "nt":
        root = os.environ.get("APPDATA") or os.path.expanduser("~/AppData/Roaming")
    else:
        root = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(root, "FabStack")


def user_data_path(filename: str, *, seed_from_bundle: bool = False) -> str:
    """Return a writable data path, optionally seeded from a bundled default."""
    directory = user_data_dir()
    os.makedirs(directory, exist_ok=True)
    destination = os.path.join(directory, filename)
    if seed_from_bundle and not os.path.exists(destination):
        source = resource_path(filename)
        if os.path.isfile(source):
            shutil.copyfile(source, destination)
    return destination


# Canonical resource locations (relative to project root)
LOGO_FILENAME = "fabstack_logo.png"     # left-panel logo
MATERIALS_LIBRARY_FILENAME = "materials_library.json"
APP_ICON_FILENAME = "iconlogo.png"      # separate app/window icon
ASSISTANT_ICON_FILENAME = "AI button.jpg"
PARAM_DEFAULTS_FILENAME = "parameter_defaults.json"
METHOD_DEFAULTS_FILENAME = "method_defaults.json"
METHOD_RECIPES_FILENAME = "method_recipes.json"

# Convenience absolute paths used by the application
LOGO_PATH = resource_path(LOGO_FILENAME)
APP_ICON_PATH = resource_path(APP_ICON_FILENAME)
ASSISTANT_ICON_PATH = resource_path(ASSISTANT_ICON_FILENAME)
MATERIALS_LIBRARY_PATH = user_data_path(
    MATERIALS_LIBRARY_FILENAME,
    seed_from_bundle=True,
)
PARAM_DEFAULTS_PATH = user_data_path(PARAM_DEFAULTS_FILENAME)
METHOD_DEFAULTS_PATH = user_data_path(
    METHOD_DEFAULTS_FILENAME,
    seed_from_bundle=True,
)
METHOD_RECIPES_PATH = user_data_path(METHOD_RECIPES_FILENAME)

__all__ = [
    "APP_ICON_FILENAME",
    "APP_ICON_PATH",
    "ASSISTANT_ICON_FILENAME",
    "ASSISTANT_ICON_PATH",
    "LOGO_FILENAME",
    "LOGO_PATH",
    "MATERIALS_LIBRARY_FILENAME",
    "MATERIALS_LIBRARY_PATH",
    "METHOD_DEFAULTS_FILENAME",
    "METHOD_DEFAULTS_PATH",
    "METHOD_RECIPES_FILENAME",
    "METHOD_RECIPES_PATH",
    "PARAM_DEFAULTS_FILENAME",
    "PARAM_DEFAULTS_PATH",
    "resource_path",
    "user_data_dir",
    "user_data_path",
]
