"""Compatibility facade for material-library schema validation."""

from fabstack.materials.schema import (
    MaterialLibraryError,
    validate_material_library,
)

__all__ = [
    "MaterialLibraryError",
    "validate_material_library",
]
