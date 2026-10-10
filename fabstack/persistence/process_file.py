"""Safe, atomic persistence helpers for FabStack ``.fab`` process files.

The legacy representation is a pickle containing plain Python values and
NumPy arrays. Loading it through :class:`RestrictedProcessUnpickler` retains
backward compatibility without allowing arbitrary Python globals.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path

import numpy as np

from fabstack.operations.contracts import OPERATION_KEYS

MAX_PROCESS_FILE_BYTES = 256 * 1024 * 1024


class ProcessFileError(ValueError):
    """Raised when a process file is unsafe or structurally invalid."""


class RestrictedProcessUnpickler(pickle.Unpickler):
    """Unpickle only the NumPy constructors used by FabStack process data."""

    _ALLOWED_GLOBALS = {
        ("numpy", "dtype"),
        ("numpy.core.numeric", "_frombuffer"),
        ("numpy._core.numeric", "_frombuffer"),
    }

    def find_class(self, module, name):
        if (module, name) not in self._ALLOWED_GLOBALS:
            raise ProcessFileError(
                f"Unsupported object in process file: {module}.{name}"
            )
        if name == "dtype":
            return np.dtype
        return _safe_numpy_frombuffer


def _safe_numpy_frombuffer(data, dtype, shape, order):
    """Rebuild an array without importing NumPy's version-specific private path."""
    if not isinstance(data, (bytes, bytearray)):
        raise ProcessFileError("Invalid NumPy buffer in process file.")
    try:
        resolved_dtype = np.dtype(dtype)
        if resolved_dtype.hasobject:
            raise ProcessFileError("Object arrays are not allowed in process files.")
        resolved_shape = tuple(int(size) for size in shape)
        if (
            not resolved_shape
            or any(size < 0 for size in resolved_shape)
            or order not in {"C", "F"}
        ):
            raise ProcessFileError("Invalid NumPy array shape in process file.")
        array = np.frombuffer(data, dtype=resolved_dtype)
        return array.reshape(resolved_shape, order=order)
    except ProcessFileError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProcessFileError("Invalid NumPy array in process file.") from exc


def _validate_history_snapshot(snapshot: dict, index: int, material_count: int) -> None:
    """Reject snapshots that cannot be restored as a cross-section."""
    title = snapshot.get("title")
    if not isinstance(title, str):
        raise ProcessFileError(
            f"Invalid process file: history snapshot {index} has no text title."
        )

    grid = snapshot.get("grid")
    if (
        not isinstance(grid, np.ndarray)
        or grid.ndim != 2
        or not grid.size
        or not np.issubdtype(grid.dtype, np.integer)
    ):
        raise ProcessFileError(
            f"Invalid process file: history snapshot {index} has an invalid grid."
        )
    if int(grid.min()) < 0 or int(grid.max()) >= material_count:
        raise ProcessFileError(
            f"Invalid process file: history snapshot {index} references an unknown material."
        )

    row_heights = snapshot.get("row_heights_um")
    if row_heights is not None:
        try:
            heights = np.asarray(row_heights, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProcessFileError(
                f"Invalid process file: history snapshot {index} has invalid row heights."
            ) from exc
        if (
            heights.ndim != 1
            or heights.shape[0] != grid.shape[0]
            or not np.all(np.isfinite(heights))
            or not np.all(heights > 0.0)
        ):
            raise ProcessFileError(
                f"Invalid process file: history snapshot {index} has invalid row heights."
            )

    for key in ("structural_role_grid", "planarizer_role_grid"):
        roles = snapshot.get(key)
        if roles is None:
            continue
        if (
            not isinstance(roles, np.ndarray)
            or roles.shape != grid.shape
            or not np.issubdtype(roles.dtype, np.integer)
        ):
            raise ProcessFileError(
                f"Invalid process file: history snapshot {index} has an invalid {key}."
            )

    if "visual_geometry" in snapshot and not isinstance(
        snapshot["visual_geometry"], list
    ):
        raise ProcessFileError(
            f"Invalid process file: history snapshot {index} has invalid visual geometry."
        )
    for key in ("structural_bodies", "planarizer_roles"):
        if key in snapshot and not isinstance(snapshot[key], dict):
            raise ProcessFileError(
                f"Invalid process file: history snapshot {index} has invalid {key}."
            )


def _validate_process_step(step: dict, index: int, material_names: list[str]) -> None:
    """Validate replay inputs before they can mutate application state."""
    operation = step.get("operation")
    allowed_operations = set(OPERATION_KEYS)
    if not isinstance(operation, str) or operation not in allowed_operations:
        raise ProcessFileError(
            f"Invalid process file: replay step {index} has an unknown operation."
        )
    if "title" in step and not isinstance(step["title"], str):
        raise ProcessFileError(
            f"Invalid process file: replay step {index} has an invalid title."
        )

    material_operations = allowed_operations - {"Flip", "Clean / Surface prep"}
    if operation in material_operations:
        material = step.get("material_name")
        if not isinstance(material, str) or material not in material_names:
            raise ProcessFileError(
                f"Invalid process file: replay step {index} references an unknown material."
            )

    numeric_key = None
    if operation in {"Deposit", "Spincoating", "Bond"}:
        numeric_key = "thickness_um"
    elif operation == "Etch":
        numeric_key = "depth_um"
    if numeric_key:
        try:
            numeric_value = float(step[numeric_key])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ProcessFileError(
                f"Invalid process file: replay step {index} has invalid {numeric_key}."
            ) from exc
        if not np.isfinite(numeric_value) or numeric_value < 0.0:
            raise ProcessFileError(
                f"Invalid process file: replay step {index} has invalid {numeric_key}."
            )

    if operation == "Etch" and "etch_mode" in step:
        if step["etch_mode"] not in {"anisotropic", "isotropic"}:
            raise ProcessFileError(
                f"Invalid process file: replay step {index} has an invalid etch mode."
            )
    if operation == "Bond" and "bond_source_info" in step:
        if not isinstance(step["bond_source_info"], dict):
            raise ProcessFileError(
                f"Invalid process file: replay step {index} has invalid bond data."
            )
    if operation == "Lithography":
        windows = step.get("windows", [])
        if not isinstance(windows, list):
            raise ProcessFileError(
                f"Invalid process file: replay step {index} has invalid windows."
            )
        if windows:
            window_values = windows
        else:
            window_values = [
                (step.get("window_start", 0.35), step.get("window_end", 0.65))
            ]
        for window in window_values:
            try:
                start, end = window
                start = float(start)
                end = float(end)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ProcessFileError(
                    f"Invalid process file: replay step {index} has invalid windows."
                ) from exc
            if (
                not np.isfinite(start)
                or not np.isfinite(end)
                or start < 0.0
                or end > 1.0
                or start >= end
            ):
                raise ProcessFileError(
                    f"Invalid process file: replay step {index} has invalid windows."
                )

def validate_process_data(data) -> dict:
    """Validate the stable top-level shape shared by supported files."""
    if not isinstance(data, dict):
        raise ProcessFileError("Invalid process file: expected an object.")

    version = data.get("format_version", 1)
    if not isinstance(version, int) or version < 1 or version > 1:
        raise ProcessFileError(f"Unsupported process format version: {version!r}")

    material_names = data.get("material_names")
    material_colors = data.get("material_colors")
    history = data.get("history")
    if not isinstance(material_names, list) or not material_names:
        raise ProcessFileError("Invalid process file: missing material list.")
    if not all(isinstance(name, str) for name in material_names):
        raise ProcessFileError("Invalid process file: material names must be text.")
    if (
        not isinstance(material_colors, list)
        or len(material_colors) != len(material_names)
        or not all(isinstance(color, str) for color in material_colors)
    ):
        raise ProcessFileError("Invalid process file: missing material colors.")
    if not isinstance(history, list) or not all(isinstance(item, dict) for item in history):
        raise ProcessFileError("Invalid process file: process history is malformed.")
    for index, snapshot in enumerate(history):
        _validate_history_snapshot(snapshot, index, len(material_names))

    steps = data.get("process_steps")
    if steps is not None and (
        not isinstance(steps, list)
        or not all(isinstance(step, dict) for step in steps)
    ):
        raise ProcessFileError("Invalid process file: replay steps are malformed.")
    if steps is not None:
        for index, step in enumerate(steps):
            _validate_process_step(step, index, material_names)
    return data


def load_process_file(path) -> dict:
    """Load and validate a legacy-compatible FabStack process file."""
    process_path = Path(path)
    try:
        size = process_path.stat().st_size
        if size > MAX_PROCESS_FILE_BYTES:
            raise ProcessFileError(
                f"Process file is too large ({size} bytes; limit {MAX_PROCESS_FILE_BYTES})."
            )
        with process_path.open("rb") as handle:
            data = RestrictedProcessUnpickler(handle).load()
    except ProcessFileError:
        raise
    except (OSError, EOFError, pickle.UnpicklingError, ValueError, TypeError) as exc:
        raise ProcessFileError(f"Could not read process file: {exc}") from exc
    return validate_process_data(data)


def save_process_file(path, data) -> None:
    """Validate and atomically replace a process file on the same filesystem."""
    validate_process_data(data)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
            delete=False,
        ) as handle:
            temp_name = handle.name
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            if handle.tell() > MAX_PROCESS_FILE_BYTES:
                raise ProcessFileError(
                    f"Process file exceeds the {MAX_PROCESS_FILE_BYTES}-byte limit."
                )
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
    except Exception:
        if temp_name:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
        raise
