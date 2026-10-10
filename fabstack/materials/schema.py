"""Validation for the editable FabStack materials-library document."""


class MaterialLibraryError(ValueError):
    """Raised when library references cannot be resolved canonically."""


def validate_material_library(data) -> dict:
    """Validate categories and compatibility references before activation."""
    if not isinstance(data, dict):
        raise MaterialLibraryError("The library root must be an object.")
    categories = data.get("materials")
    if not isinstance(categories, dict) or not categories:
        raise MaterialLibraryError("The library must define material categories.")

    material_names = set()
    for category, values in categories.items():
        if not isinstance(category, str) or not category.strip():
            raise MaterialLibraryError("Material category names must be non-empty text.")
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise MaterialLibraryError(f"Category {category!r} must contain text values.")
        material_names.update(values)

    compatibility = data.get("compatibility", {})
    if not isinstance(compatibility, dict):
        raise MaterialLibraryError("Compatibility rules must be an object.")

    for section, rules in compatibility.items():
        if not isinstance(rules, dict):
            raise MaterialLibraryError(f"Compatibility section {section!r} must be an object.")
        if section.endswith("_by_material"):
            unknown = sorted(set(rules) - material_names)
            if unknown:
                raise MaterialLibraryError(
                    f"{section} references unknown materials: {', '.join(unknown)}"
                )
        elif section.endswith("_by_category"):
            unknown = sorted(set(rules) - set(categories))
            if unknown:
                raise MaterialLibraryError(
                    f"{section} references unknown categories: {', '.join(unknown)}"
                )
        if section == "etch_modes_by_etchant":
            if not all(
                isinstance(value, str)
                and value.lower() in {"anisotropic", "isotropic"}
                for value in rules.values()
            ):
                raise MaterialLibraryError(
                    "Etch modes must be 'anisotropic' or 'isotropic'."
                )
        elif not all(isinstance(values, list) for values in rules.values()):
            raise MaterialLibraryError(f"{section} rules must contain lists.")

    restrictions = compatibility.get("restrictions_by_operation", {})
    unknown_restrictions = sorted(
        {
            material
            for values in restrictions.values()
            for material in values
            if material not in material_names
        }
    )
    if unknown_restrictions:
        raise MaterialLibraryError(
            "Operation restrictions reference unknown materials: "
            + ", ".join(unknown_restrictions)
        )
    return data
