"""
Interaction Evolution Operators.

Paper Section 3.1: Three semantic types of task-state change.
Each operator defines: allowed changes, must-preserve fields, must-invalidate fields.

Operator → Revision Type mapping:
  - Introduce: add, constraint_accumulation, preserve, format_locking, delayed_dependency
  - Replace: replace, rebind, contract_update, partial_revision, valid_correction, scope_switch
  - Withdraw: revoke, recovery_probe
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

from ..core.ebir import DialogueState, RevisionType


@dataclass
class OperatorSpec:
    """Operator specification."""
    name: str
    description: str
    revision_type: str  # "introduce", "replace", "withdraw"
    target_weakness: str
    preconditions: list[str] = field(default_factory=list)
    allowed_change_fields: list[str] = field(default_factory=list)
    must_preserve_fields: list[str] = field(default_factory=list)
    must_not_preserve_fields: list[str] = field(default_factory=list)

    def applies_to(self, state: DialogueState) -> bool:
        """Check if this operator is applicable to the current state."""
        for precond in self.preconditions:
            if not self._check_precondition(precond, state):
                return False
        return True

    def _check_precondition(self, precond: str, state: DialogueState) -> bool:
        if precond == "has_existing_constraints":
            return len(state.constraints) > 0
        if precond == "has_prior_response":
            return len(state.model_commitments) > 0
        if precond == "has_facts":
            return len(state.facts) > 0
        if precond == "has_entities":
            return len(state.entities) > 0
        if precond == "has_output_contract":
            return state.output_contract is not None
        if precond == "has_environment":
            return len(state.environment) > 0
        if precond == "has_prior_context":
            return bool(state.model_commitments or state.lifecycle)
        if precond == "has_conditions":
            return bool(state.facts or any(c.type == "condition" for c in state.constraints))
        return True  # Unknown preconditions default to True (safe)


# ============================================================
# Operator Registry — aligned with paper's three revision types
# ============================================================

OPERATOR_SPECS: dict[str, OperatorSpec] = {
    # --- Introduce: new info, nothing invalidated ---
    "add": OperatorSpec(
        name="add", description="Add a new constraint or fact",
        revision_type=RevisionType.INTRODUCE.value,
        target_weakness="constraint_accumulation",
        preconditions=["has_existing_constraints"],
        allowed_change_fields=["constraints", "facts"],
        must_preserve_fields=["goal", "output_contract"],
    ),
    "constraint_accumulation": OperatorSpec(
        name="constraint_accumulation", description="Accumulate an additional constraint",
        revision_type=RevisionType.INTRODUCE.value,
        target_weakness="constraint_accumulation",
        preconditions=["has_existing_constraints"],
        allowed_change_fields=["constraints"],
        must_preserve_fields=["goal", "entities", "output_contract"],
    ),
    "preserve": OperatorSpec(
        name="preserve", description="Introduce a preserving fact or binding",
        revision_type=RevisionType.INTRODUCE.value,
        target_weakness="information_preservation",
        preconditions=["has_facts"],
        allowed_change_fields=["facts"],
        must_preserve_fields=["goal", "constraints", "output_contract"],
    ),
    "format_locking": OperatorSpec(
        name="format_locking", description="Lock or tighten the output format",
        revision_type=RevisionType.INTRODUCE.value,
        target_weakness="format_adherence",
        preconditions=["has_output_contract"],
        allowed_change_fields=["output_contract"],
        must_preserve_fields=["goal", "constraints"],
    ),
    # --- Replace: new value replaces active one ---
    "replace": OperatorSpec(
        name="replace", description="Replace a parameter value",
        revision_type=RevisionType.REPLACE.value,
        target_weakness="parameter_revision",
        preconditions=["has_existing_constraints"],
        allowed_change_fields=["constraints", "facts", "environment"],
        must_preserve_fields=["goal"],
        must_not_preserve_fields=["constraints"],
    ),
    "partial_revision": OperatorSpec(
        name="partial_revision", description="Revise part of the task specification",
        revision_type=RevisionType.REPLACE.value,
        target_weakness="partial_update",
        preconditions=["has_existing_constraints", "has_facts"],
        allowed_change_fields=["constraints", "facts"],
        must_preserve_fields=["goal", "output_contract"],
        must_not_preserve_fields=["constraints"],
    ),
    "valid_correction": OperatorSpec(
        name="valid_correction", description="User provides a correct correction",
        revision_type=RevisionType.REPLACE.value,
        target_weakness="error_correction",
        preconditions=["has_prior_response"],
        allowed_change_fields=["constraints", "facts"],
        must_preserve_fields=["goal", "output_contract"],
        must_not_preserve_fields=["facts"],
    ),
    "scope_switch": OperatorSpec(
        name="scope_switch", description="Switch the scope of the query",
        revision_type=RevisionType.REPLACE.value,
        target_weakness="scope_tracking",
        preconditions=["has_entities"],
        allowed_change_fields=["entities", "facts"],
        must_preserve_fields=["goal", "output_contract"],
        must_not_preserve_fields=["entities"],
    ),
    # --- Withdraw: revoke without replacement ---
    "revoke": OperatorSpec(
        name="revoke", description="Revoke a requirement",
        revision_type=RevisionType.WITHDRAW.value,
        target_weakness="requirement_withdrawal",
        preconditions=["has_existing_constraints"],
        allowed_change_fields=["constraints"],
        must_preserve_fields=["goal", "output_contract"],
        must_not_preserve_fields=["constraints"],
    ),
    "recovery_probe": OperatorSpec(
        name="recovery_probe", description="Recover from a superseded requirement",
        revision_type=RevisionType.WITHDRAW.value,
        target_weakness="superseded_recovery",
        preconditions=["has_prior_context"],
        allowed_change_fields=["constraints", "facts"],
        must_preserve_fields=["goal", "output_contract"],
        must_not_preserve_fields=["constraints"],
    ),
}

OPERATOR_REGISTRY: dict[str, OperatorSpec] = OPERATOR_SPECS


def get_operator_spec(name: str) -> OperatorSpec | None:
    return OPERATOR_SPECS.get(name)


def get_revision_type(operator: str) -> str:
    """Map an operator name to its revision type."""
    spec = OPERATOR_SPECS.get(operator)
    return spec.revision_type if spec else RevisionType.INTRODUCE.value


def is_task_revision(operator: str) -> bool:
    """Whether an operator constitutes a task revision (Replace or Withdraw)."""
    rt = get_revision_type(operator)
    return rt in (RevisionType.REPLACE.value, RevisionType.WITHDRAW.value)


def select_operator(
    state: DialogueState,
    available: list[str],
    strategy: str = "coverage",
    model_hint: str | None = None,
    used_operators: list[str] | None = None,
) -> str | None:
    """
    Select an operator from the available set.
    Strategy: "coverage" (diversify) or "targeted" (pick specific weakness).
    """
    if not available:
        return None

    applicable = [op for op in available if op in OPERATOR_SPECS and OPERATOR_SPECS[op].applies_to(state)]
    if not applicable:
        return None

    used = set(used_operators or [])

    if strategy == "coverage":
        # Prefer unused operators for coverage
        unused = [op for op in applicable if op not in used]
        if unused:
            return unused[0]
        return applicable[0]
    elif strategy == "targeted" and model_hint:
        # Target a specific weakness type
        for op in applicable:
            if OPERATOR_SPECS[op].target_weakness == model_hint:
                return op

    return applicable[0]
