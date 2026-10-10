"""Stable operation vocabulary and boundary-shape contracts."""

from typing import Any, Literal, TypedDict

from fabstack.terminology import OPERATION_LABELS


OperationKey = Literal[
    "Deposit",
    "Spincoating",
    "Bond",
    "Etch",
    "Strip material",
    "Lift-off",
    "Clean / Surface prep",
    "Bake/Annealing",
    "Lithography",
    "Flip",
]

OPERATION_KEYS: tuple[OperationKey, ...] = tuple(OPERATION_LABELS)
UI_OPERATION_KEYS: tuple[OperationKey, ...] = tuple(
    operation for operation in OPERATION_KEYS if operation != "Flip"
)

OperationSource = Literal["interactive", "assistant", "replay"]


class _UnsetType:
    """Marker for a field that was not supplied at a contract boundary."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"


UNSET = _UnsetType()


class PersistedOperationStep(TypedDict, total=False):
    operation: OperationKey
    material_name: str
    title: str
    thickness_um: float
    depth_um: float
    method: str
    etch_mode: str
    bond_source_info: dict[str, Any]
    windows: list[Any]
    window_start: float
    window_end: float
    invert_lithography: bool


class AssistantOperationCommand(TypedDict, total=False):
    action: Literal["operation"]
    operation: OperationKey
    material: str
    thickness_um: float
    depth_um: float
    temperature_c: float | None
    time_s: float | None
    method: str
    etch_mode: str
    invert_lithography: bool | None
    windowing_mode: str
    windows: list[Any]


__all__ = [
    "AssistantOperationCommand",
    "OPERATION_KEYS",
    "OperationKey",
    "OperationSource",
    "PersistedOperationStep",
    "UI_OPERATION_KEYS",
    "UNSET",
]
