"""Generic, auditable operations over DialogueState.

The construction engine works with state paths instead of benchmark names.
Domain adapters may override high-level hooks, while these helpers provide
a safe default for declarative transitions.
"""

from __future__ import annotations
from copy import deepcopy
from typing import Any

from .ebir import DialogueState, LifecycleEntry, LifecycleLabel, TransitionContract


class StatePathError(ValueError):
    """Raised when a contract references an invalid or unsafe state path."""


def _parts(path: str) -> list[str]:
    return path.split(".")


def get_path(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in _parts(path):
        if not isinstance(current, dict) or part not in current:
            raise StatePathError(f"state path does not exist: {path}")
        current = current[part]
    return current


def set_path(data: dict[str, Any], path: str, value: Any) -> None:
    parts = _parts(path)
    current: Any = data
    for part in parts[:-1]:
        if not isinstance(current, dict):
            raise StatePathError(f"state path crosses a non-object value: {path}")
        child = current.get(part)
        if child is None:
            child = {}
            current[part] = child
        if not isinstance(child, dict):
            raise StatePathError(f"state path crosses a non-object value: {path}")
        current = child
    if not isinstance(current, dict):
        raise StatePathError(f"cannot assign state path: {path}")
    current[parts[-1]] = deepcopy(value)


def changed_paths(before: DialogueState, after: DialogueState) -> set[str]:
    """Return minimal semantic paths that changed between two states."""
    before_state = before.semantic_dict()
    after_state = after.semantic_dict()
    before_state.pop("lifecycle", None)
    after_state.pop("lifecycle", None)
    return _diff(before_state, after_state)


def _diff(a: Any, b: Any, prefix: str = "") -> set[str]:
    result: set[str] = set()
    if type(a) is not type(b):
        return {prefix or "<root>"}
    if isinstance(a, dict):
        all_keys = set(a.keys()) | set(b.keys())
        for k in all_keys:
            p = f"{prefix}.{k}" if prefix else k
            if k not in a or k not in b:
                result.add(p)
            else:
                result |= _diff(a[k], b[k], p)
    elif isinstance(a, list):
        if len(a) != len(b) or a != b:
            return {prefix or "<root>"}
    else:
        if a != b:
            return {prefix or "<root>"}
    return result


def apply_contract(state: DialogueState, contract: TransitionContract) -> DialogueState:
    """Apply a declarative contract without benchmark-specific branching."""
    if not contract.is_valid():
        raise StatePathError("invalid transition contract")
    state_dict = state.to_dict()
    for path, value in contract.updates.items():
        set_path(state_dict, path, value)
    next_state = DialogueState.from_dict(state_dict)
    _apply_lifecycle(next_state, contract)
    return next_state


def _apply_lifecycle(state: DialogueState, contract: TransitionContract) -> None:
    """Update lifecycle labels based on the contract's deprecated slots."""
    for slot in contract.deprecated_slots:
        parts = _parts(slot)
        leaf = parts[-1]
        for entry in state.lifecycle:
            if entry.slot == leaf and entry.label == LifecycleLabel.ACTIVE:
                entry.label = LifecycleLabel.SUPERSEDED


def validate_transition(
    before: DialogueState,
    after: DialogueState,
    contract: TransitionContract,
) -> list[str]:
    """Return contract violations; empty list means valid transition."""
    reasons: list[str] = []
    if not contract.is_valid():
        reasons.append("contract is incomplete or has overlapping slot sets")
        return reasons

    # Check preserved fields are unchanged
    before_dict = before.to_dict()
    after_dict = after.to_dict()
    for slot in contract.preserved_slots:
        try:
            b = get_path(before_dict, slot)
            a = get_path(after_dict, slot)
            if b != a:
                reasons.append(f"preserved field '{slot}' was modified")
        except StatePathError:
            pass

    # Check deprecated fields are no longer active
    for slot in contract.deprecated_slots:
        parts = _parts(slot)
        leaf = parts[-1]
        for entry in after.lifecycle:
            if entry.slot == leaf and entry.label == LifecycleLabel.ACTIVE:
                reasons.append(f"deprecated field '{slot}' is still active")

    return reasons


def contracts_equivalent(c1: TransitionContract, c2: TransitionContract) -> bool:
    """Check if two contracts are semantically equivalent."""
    return (c1.operator == c2.operator
            and set(c1.mutable_slots) == set(c2.mutable_slots)
            and set(c1.preserved_slots) == set(c2.preserved_slots)
            and set(c1.deprecated_slots) == set(c2.deprecated_slots)
            and c1.updates == c2.updates)
