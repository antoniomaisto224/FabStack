"""Shared user-facing terminology for FabStack.

Internal operation keys are persisted in ``.fab`` files and must remain stable.
The labels below keep those identifiers separate from concise UI copy.
"""

OPERATION_LABELS = {
    "Deposit": "Deposit",
    "Spincoating": "Spin coat",
    "Bond": "Bond",
    "Etch": "Etch",
    "Strip material": "Strip",
    "Lift-off": "Lift-off",
    "Clean / Surface prep": "Clean / surface prep",
    "Bake/Annealing": "Bake / anneal",
    "Lithography": "Lithography",
    "Flip": "Flip stack",
}


def operation_label(operation: str) -> str:
    """Return concise display copy without changing the persisted key."""
    key = str(operation or "").strip()
    return OPERATION_LABELS.get(key, key)


__all__ = ["OPERATION_LABELS", "operation_label"]
