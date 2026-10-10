"""Side-effect-free lookup for bundled and source resources."""

import os
import sys


def resource_path(relative_path: str) -> str:
    """Resolve a resource without creating or modifying user data."""
    bundle_root = getattr(sys, "_MEIPASS", "")
    if bundle_root:
        return os.path.join(bundle_root, relative_path)

    package_asset = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "assets",
        relative_path,
    )
    if os.path.exists(package_asset):
        return package_asset

    # Large optional resources such as the GGUF model remain release-managed
    # beside the source package rather than being installed as package data.
    source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(source_root, relative_path)


__all__ = ["resource_path"]
