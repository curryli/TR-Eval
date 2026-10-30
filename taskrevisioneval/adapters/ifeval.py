"""
IFEval Adapter: Instruction Following benchmark adapter.

Source: IFEval
  GitHub:   https://github.com/google-research/google-research/tree/master/instruction_following_eval
  Format:   {"key": int, "prompt": str, "instruction_id_list": [str], "kwargs": [{...}]}

Paper Table 15: Instruction following uses constraint rebuilds and
deterministic rule checkers (length, keyword, format constraints).

Verifier: deterministic IFEval constraint checker (highest priority).
Suitable operators: constraint_accumulation, partial_revision, format_locking.

Usage:
    from datasets import load_dataset
    ds = load_dataset("google/ifeval", split="train")
    seed = IFEvalAdapter().parse_seed(ds[0])
"""

from __future__ import annotations
import json
from typing import Any

from .base import BenchmarkAdapter
from ..core.ebir import (
    EBIR, DialogueState, EvolutionTest, CanonicalProbe, EvolvedTestSuite,
    StateConstraint, OutputContract, LifecycleLabel, LifecycleEntry,
    TransitionContract,
)
from ..evaluation.verifier import VerifierResult, verify_ifeval_constraints


class IFEvalAdapter(BenchmarkAdapter):
    """IFEval benchmark evolution adapter."""

    family = "instruction_following"

    @property
    def editable_fields(self) -> list[str]:
        return ["constraints", "output_contract"]

    @property
    def protected_fields(self) -> list[str]:
        return ["goal", "entities"]

    def parse_seed(self, raw_item: dict) -> dict:
        """
        Parse IFEval raw sample into unified seed.
        Raw format: {"key": int, "prompt": str, "instruction_id_list": [str], "kwargs": [{...}]}
        """
        return {
            "task_id": f"ifeval_{raw_item['key']}",
            "input": raw_item["prompt"],
            "reference_answer": "",
            "metadata": {
                "domain": "instruction_following",
                "language": "en",
                "instruction_id_list": raw_item.get("instruction_id_list", []),
                "kwargs": raw_item.get("kwargs", []),
            },
        }

    def lift_state(self, seed: dict) -> EBIR:
        """Convert IFEval seed to EBIR with structured constraints."""
        meta = seed.get("metadata", {})
        inst_ids = meta.get("instruction_id_list", [])
        kwargs_list = meta.get("kwargs", [])

        constraints = []
        for inst_id, kwargs in zip(inst_ids, kwargs_list):
            clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
            constraints.append(StateConstraint.instruction(inst_id, clean_kwargs))

        lifecycle = [LifecycleEntry(slot=c.key or c.instruction_id, label=LifecycleLabel.ACTIVE)
                     for c in constraints]

        state = DialogueState(
            goal="Follow all instructions and constraints in the prompt.",
            facts=[],
            entities=[],
            constraints=constraints,
            output_contract=OutputContract(format="free_text"),
            verifier_spec={
                "type": "ifeval_constraint_check",
                "instruction_id_list": inst_ids,
                "kwargs_list": kwargs_list,
            },
            reference={"type": "constraint_based"},
            lifecycle=lifecycle,
        )

        return EBIR(
            task_id=seed["task_id"],
            domain="instruction_following",
            input=seed["input"],
            reference=seed.get("reference_answer", ""),
            initial_state=state,
            mutable_slots=["constraints", "output_contract"],
            preserved_slots=["goal", "entities"],
            supported_operators=[
                "constraint_accumulation", "partial_revision",
                "format_locking", "valid_correction", "revoke",
            ],
            metadata=meta,
        )

    def verify_response(self, response: str, state: DialogueState,
                        test: EvolutionTest | None = None) -> VerifierResult:
        """Verify using deterministic IFEval constraint checker."""
        vspec = state.verifier_spec
        if vspec.get("type") == "ifeval_constraint_check":
            return verify_ifeval_constraints(
                response,
                vspec.get("instruction_id_list", []),
                vspec.get("kwargs_list", []),
            )
        # Fallback: check all active constraints
        return VerifierResult.fail(["unsupported IFEval verifier specification"])

    def build_canonical_probe(self, suite: EvolvedTestSuite) -> CanonicalProbe:
        """Build single-turn probe presenting the final state directly."""
        final_state = suite.tests[-1].state_after if suite.tests else suite.ebir.initial_state
        prompt = self.render_state_as_prompt(final_state, suite.ebir)
        return CanonicalProbe(
            prompt=prompt,
            expected_behavior="Response satisfying all active constraints",
            verifier_spec=final_state.verifier_spec,
            canonicalization_type="single_turn",
            endpoint_state=final_state.semantic_dict(),
            endpoint_fingerprint=final_state.fingerprint(),
        )

    def recompute_endpoint(self, state: DialogueState) -> tuple[list[str], dict[str, Any]]:
        """Rebuild the constraint checker from the active constraint set."""
        vtype = state.verifier_spec.get("type", "ifeval_constraint_check")
        inst_ids = [c.instruction_id for c in state.active_constraints() if c.instruction_id]
        kwargs_list = [c.kwargs for c in state.active_constraints() if c.instruction_id]
        new_spec = {"type": vtype, "instruction_id_list": inst_ids, "kwargs_list": kwargs_list}
        state.verifier_spec = new_spec
        return [], new_spec

    def execute_revision(self, state: DialogueState,
                         contract: TransitionContract) -> tuple[DialogueState, dict[str, Any]]:
        """Execute a typed revision, updating constraints and lifecycle."""
        from ..core.state_ops import apply_contract
        new_state = apply_contract(state, contract)

        # Update lifecycle for deprecated constraints
        for slot in contract.deprecated_slots:
            new_state.mark_superseded(slot)

        # Add lifecycle entries for new constraints
        for key in contract.updates:
            if key.startswith("constraints"):
                leaf = key.split(".")[-1]
                new_state.lifecycle.append(
                    LifecycleEntry(slot=leaf, label=LifecycleLabel.ACTIVE)
                )

        evidence = {"contract_applied": True, "changes": list(contract.updates.keys())}
        return new_state, evidence
