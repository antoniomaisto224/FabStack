"""Shared material-operation routing and persisted-step replay."""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from fabstack.operations.contracts import UNSET


BondExternalResult = tuple[str, dict[str, Any], list[str]]


@dataclass(frozen=True)
class _RouteSemantics:
    allow_legacy_fallback: bool
    guard_bake_xsec_after_material_resolution: bool
    coerce_legacy_lithography_window_values: bool


_UI_ROUTE = _RouteSemantics(
    allow_legacy_fallback=False,
    guard_bake_xsec_after_material_resolution=True,
    coerce_legacy_lithography_window_values=False,
)
_REPLAY_ROUTE = _RouteSemantics(
    allow_legacy_fallback=True,
    guard_bake_xsec_after_material_resolution=False,
    coerce_legacy_lithography_window_values=True,
)


class MaterialOperationHost(Protocol):
    """Host surface needed by the shared material-operation kernel."""

    xsec: Any

    def um_to_process_px(
        self,
        value_um: float,
        *,
        mode: str,
        minimum: int,
    ) -> int: ...


class BakeOperationHost(Protocol):
    """Host surface needed by the specialized bake kernel."""

    VOID_INDEX: int
    xsec: Any

    def _bake_material(self, material_index: int) -> int: ...


class FlipOperationHost(Protocol):
    """Host surface needed by the specialized flip kernel."""

    xsec: Any


class BondOperationHost(Protocol):
    """Host surface needed by the specialized bond kernels."""

    bond_source_info: Any
    xsec: Any

    def _apply_bond_with_external_stack(
        self,
        material_index: int,
        thickness_um: float,
    ) -> BondExternalResult: ...


class ReplayExecutionHost(Protocol):
    """Main-window capabilities required by deterministic process replay."""

    VOID_INDEX: int
    bond_source_info: Any
    xsec: Any

    def _apply_bond_with_external_stack(
        self,
        material_index: int,
        thickness_um: float,
    ): ...

    def _bake_material(self, material_index: int) -> int: ...
    def _default_lithography_invert_for_material(self, material_name: str): ...
    def _lithography_invert_from_text(self, method): ...
    def get_geometry_etch_mask(self, material_index: int): ...
    def material_index(self, material_name: str) -> int: ...
    def um_to_process_px(
        self,
        value_um: float,
        *,
        mode: str,
        minimum: int,
    ) -> int: ...


def _execute_material_operation(
    host: MaterialOperationHost,
    operation: str,
    material_index: int,
    *,
    thickness_um: Any = UNSET,
    depth_um: Any = UNSET,
    mask: Any = UNSET,
    etch_mode: Any = UNSET,
    windows: Any = UNSET,
    tone: Any = UNSET,
    semantics: _RouteSemantics,
) -> bool:
    if operation == "Clean / Surface prep":
        return True

    if operation == "Strip material":
        host.xsec.strip_top_only(material_index)
        return True

    if operation == "Deposit":
        if semantics.allow_legacy_fallback and not hasattr(
            host.xsec,
            "deposit_physical",
        ):
            thickness_px = host.um_to_process_px(
                thickness_um,
                mode="ceil",
                minimum=1,
            )
            host.xsec.deposit(material_index, thickness_px=thickness_px)
        else:
            host.xsec.deposit_physical(material_index, thickness_um)
        return True

    if operation == "Spincoating":
        if semantics.allow_legacy_fallback and not hasattr(
            host.xsec,
            "spincoat_physical",
        ):
            thickness_px = host.um_to_process_px(
                thickness_um,
                mode="ceil",
                minimum=1,
            )
            host.xsec.spincoat(material_index, thickness_px=thickness_px)
        else:
            host.xsec.spincoat_physical(material_index, thickness_um)
        return True

    if operation == "Etch":
        if semantics.allow_legacy_fallback and not hasattr(
            host.xsec,
            "etch_physical",
        ):
            depth_px = host.um_to_process_px(
                depth_um,
                mode="ceil",
                minimum=1,
            )
            host.xsec.etch(
                material_index,
                depth_px=depth_px,
                mask=mask,
                mode=etch_mode,
            )
        else:
            host.xsec.etch_physical(
                material_index,
                depth_um=depth_um,
                mask=mask,
                mode=etch_mode,
            )
        return True

    if operation == "Lift-off":
        host.xsec.liftoff(material_index)
        return True

    if operation == "Lithography":
        if hasattr(host.xsec, "pattern_pr_lithography"):
            host.xsec.pattern_pr_lithography(
                material_index,
                windows,
                tone=tone,
            )
        else:
            for start, end in windows:
                if semantics.coerce_legacy_lithography_window_values:
                    start = float(start)
                    end = float(end)
                host.xsec.pattern_pr_open_window(material_index, start, end)
        return True

    return False


def execute_ui_material_operation(
    host: MaterialOperationHost,
    operation: str,
    material_index: int,
    *,
    thickness_um: Any = UNSET,
    depth_um: Any = UNSET,
    mask: Any = UNSET,
    etch_mode: Any = UNSET,
    windows: Any = UNSET,
    tone: Any = UNSET,
) -> bool:
    return _execute_material_operation(
        host,
        operation,
        material_index,
        thickness_um=thickness_um,
        depth_um=depth_um,
        mask=mask,
        etch_mode=etch_mode,
        windows=windows,
        tone=tone,
        semantics=_UI_ROUTE,
    )


def execute_replay_material_operation(
    host: MaterialOperationHost,
    operation: str,
    material_index: int,
    *,
    thickness_um: Any = UNSET,
    depth_um: Any = UNSET,
    mask: Any = UNSET,
    etch_mode: Any = UNSET,
    windows: Any = UNSET,
    tone: Any = UNSET,
) -> bool:
    return _execute_material_operation(
        host,
        operation,
        material_index,
        thickness_um=thickness_um,
        depth_um=depth_um,
        mask=mask,
        etch_mode=etch_mode,
        windows=windows,
        tone=tone,
        semantics=_REPLAY_ROUTE,
    )


def _execute_bake_operation(
    host: BakeOperationHost,
    material_index: int,
    *,
    semantics: _RouteSemantics,
) -> int:
    baked_idx = host._bake_material(material_index)
    if baked_idx != material_index:
        if semantics.guard_bake_xsec_after_material_resolution and host.xsec is None:
            return baked_idx
        grid = host.xsec.grid
        h, w = grid.shape
        for x in range(w):
            col = grid[:, x]
            nz = np.where(col != host.VOID_INDEX)[0]
            if nz.size == 0:
                continue
            y0 = nz.min()
            if col[y0] != material_index:
                continue
            y = y0
            while y < h and col[y] == material_index:
                col[y] = baked_idx
                y += 1
        host.xsec.grid = grid
    return baked_idx


def execute_ui_bake_operation(
    host: BakeOperationHost,
    material_index: int,
) -> int:
    return _execute_bake_operation(
        host,
        material_index,
        semantics=_UI_ROUTE,
    )


def execute_replay_bake_operation(
    host: BakeOperationHost,
    material_index: int,
) -> int:
    return _execute_bake_operation(
        host,
        material_index,
        semantics=_REPLAY_ROUTE,
    )


def _execute_flip_operation(host: FlipOperationHost):
    return host.xsec.flip_active_stack()


def execute_ui_flip_operation(host: FlipOperationHost):
    return _execute_flip_operation(host)


def execute_replay_flip_operation(host: FlipOperationHost):
    return _execute_flip_operation(host)


def _execute_simple_bond_operation(
    host: BondOperationHost,
    material_index: int,
    thickness_um: float,
    *,
    thickness_px: Any = UNSET,
    semantics: _RouteSemantics,
) -> None:
    if semantics.allow_legacy_fallback and not hasattr(host.xsec, "bond_physical"):
        host.xsec.bond(material_index, thickness_px=thickness_px)
    else:
        host.xsec.bond_physical(material_index, thickness_um)


def _execute_external_bond_operation(
    host: BondOperationHost,
    material_index: int,
    thickness_um: float,
) -> BondExternalResult:
    return host._apply_bond_with_external_stack(material_index, thickness_um)


def execute_ui_simple_bond_operation(
    host: BondOperationHost,
    material_index: int,
    thickness_um: float,
) -> None:
    _execute_simple_bond_operation(
        host,
        material_index,
        thickness_um,
        semantics=_UI_ROUTE,
    )


def execute_ui_external_bond_operation(
    host: BondOperationHost,
    material_index: int,
    thickness_um: float,
) -> BondExternalResult:
    return _execute_external_bond_operation(host, material_index, thickness_um)


def execute_replay_bond_operation(
    host: BondOperationHost,
    material_index: int,
    thickness_um: float,
    *,
    thickness_px: Any,
    source_info: Any,
) -> None:
    if isinstance(source_info, dict):
        previous_source = host.bond_source_info
        try:
            host.bond_source_info = source_info
            _execute_external_bond_operation(
                host,
                material_index,
                thickness_um,
            )
        finally:
            host.bond_source_info = previous_source
    else:
        _execute_simple_bond_operation(
            host,
            material_index,
            thickness_um,
            thickness_px=thickness_px,
            semantics=_REPLAY_ROUTE,
        )
    return None


def apply_process_step_to_current_xsec(
    host: ReplayExecutionHost,
    step: dict,
) -> bool:
    if host.xsec is None or not isinstance(step, dict):
        return False

    op = str(step.get("operation", "")).strip()
    material_name = str(step.get("material_name", "")).strip()
    mat_index = host.material_index(material_name) if material_name else -1

    if op in {"Deposit", "Spincoating", "Bond", "Etch", "Strip material", "Lift-off", "Bake/Annealing", "Lithography"}:
        if mat_index < 0:
            return False

    if op == "Deposit":
        thickness_um = float(step.get("thickness_um", 0.0))
        execute_replay_material_operation(
            host,
            op,
            mat_index,
            thickness_um=thickness_um,
        )
        return True

    if op == "Spincoating":
        thickness_um = float(step.get("thickness_um", 0.0))
        execute_replay_material_operation(
            host,
            op,
            mat_index,
            thickness_um=thickness_um,
        )
        return True

    if op == "Bond":
        thickness_um = float(step.get("thickness_um", 0.0))
        thickness_px = host.um_to_process_px(thickness_um, mode="ceil", minimum=1)
        source_info = step.get("bond_source_info")
        execute_replay_bond_operation(
            host,
            mat_index,
            thickness_um,
            thickness_px=thickness_px,
            source_info=source_info,
        )
        return True

    if op == "Etch":
        depth_um = float(step.get("depth_um", 0.0))
        if depth_um <= 0.0:
            return True
        mask = host.get_geometry_etch_mask(mat_index)
        etch_mode = str(step.get("etch_mode", "anisotropic") or "anisotropic")
        execute_replay_material_operation(
            host,
            op,
            mat_index,
            depth_um=depth_um,
            mask=mask,
            etch_mode=etch_mode,
        )
        return True

    if op == "Strip material":
        execute_replay_material_operation(host, op, mat_index)
        return True

    if op == "Lift-off":
        execute_replay_material_operation(host, op, mat_index)
        return True

    if op == "Bake/Annealing":
        execute_replay_bake_operation(host, mat_index)
        return True

    if op == "Lithography":
        windows = step.get("windows", [])
        if not isinstance(windows, list) or not windows:
            windows = [(float(step.get("window_start", 0.35)), float(step.get("window_end", 0.65)))]
        if "invert_lithography" in step:
            invert_lithography = bool(step.get("invert_lithography"))
        else:
            method_invert = host._lithography_invert_from_text(step.get("method", ""))
            material_invert = host._default_lithography_invert_for_material(material_name)
            invert_lithography = bool(
                method_invert
                if method_invert is not None
                else (material_invert if material_invert is not None else False)
            )
        execute_replay_material_operation(
            host,
            op,
            mat_index,
            windows=windows,
            tone="negative" if invert_lithography else "positive",
        )
        return True

    if op == "Flip":
        execute_replay_flip_operation(host)
        return True

    if op == "Clean / Surface prep":
        execute_replay_material_operation(host, op, mat_index)
        return True

    return False


__all__ = [
    "BakeOperationHost",
    "BondExternalResult",
    "BondOperationHost",
    "FlipOperationHost",
    "MaterialOperationHost",
    "ReplayExecutionHost",
    "apply_process_step_to_current_xsec",
    "execute_replay_bake_operation",
    "execute_replay_bond_operation",
    "execute_replay_flip_operation",
    "execute_replay_material_operation",
    "execute_ui_bake_operation",
    "execute_ui_external_bond_operation",
    "execute_ui_flip_operation",
    "execute_ui_material_operation",
    "execute_ui_simple_bond_operation",
]
