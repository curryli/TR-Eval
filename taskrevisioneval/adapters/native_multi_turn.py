"""
Native Multi-Turn Adapter: MultiChallenge / MT-Bench benchmark adapter.

Source: MultiChallenge
  Dataset:  https://huggingface.co/datasets/ScaleAI/MultiChallenge

Source: MT-Bench
  GitHub:   https://github.com/lm-sys/FastChat

Paper Table 15: Native multi-turn uses a calibrated, LLM-based, state-grounded
answer verifier for endpoint scoring. State transitions and endpoint equality
remain programmatically checked. Commitments and mutable slots can be added,
replaced, or withdrawn. Reject pragmatic and authority ambiguity.

Verifier: LLM-based state-grounded judge (provide via adapter interface).
Suitable operators: partial_revision, valid_correction, scope_switch, revoke.

Usage:
    from datasets import load_dataset
    ds = load_dataset("ScaleAI/MultiChallenge", split="test")
    seed = NativeMultiTurnAdapter("multichallenge").parse_seed(ds[0])
"""

from __future__ import annotations
import json
from typing import Any
from ..core.ids import stable_id

from .base import BenchmarkAdapter
from ..core.ebir import (
    EBIR, DialogueState, EvolutionTest, CanonicalProbe, EvolvedTestSuite,
    StateConstraint, OutputContract, LifecycleLabel, LifecycleEntry,
    TransitionContract,
)
from ..evaluation.verifier import VerifierResult, LLMJudgeVerifier


class NativeMultiTurnAdapter(BenchmarkAdapter):
    """Native multi-turn benchmark evolution adapter (MultiChallenge, MT-Bench)."""

    family = "native_multi_turn"

    def __init__(self, dataset_name: str = "multichallenge"):
        self.dataset_name = dataset_name
        self._judge: LLMJudgeVerifier | None = None

    @property
    def editable_fields(self) -> list[str]:
        return ["constraints", "facts", "environment"]

    @property
    def protected_fields(self) -> list[str]:
        return ["goal", "output_contract"]

    def set_llm_client(self, llm_client):
        """Inject LLM client for the state-grounded judge."""
        super().set_llm_client(llm_client)
        self._judge = self._llm_judge

    def parse_seed(self, raw_item: dict) -> dict:
        """
        Parse native multi-turn sample into unified seed.
        MultiChallenge format: {"task_id": ..., "dialogue": [...], "expected_outcome": ..., "verifier": ...}
        MT-Bench format: {"question_id": int, "turns": [...], "reference": [...]}
        """
        task_id = str(raw_item.get("task_id",
                         raw_item.get("question_id",
                         stable_id(raw_item))))
        dialogue = raw_item.get("dialogue", raw_item.get("turns", []))
        expected = raw_item.get("expected_outcome", raw_item.get("reference", ""))
        verifier_spec = raw_item.get("verifier", {})

        # Extract the initial user turn as the primary input
        first_user = ""
        for msg in dialogue:
            if msg.get("role") == "user":
                first_user = msg.get("content", "")
                break

        return {
            "task_id": f"nmt_{task_id}",
            "input": first_user,
            "reference_answer": json.dumps(expected) if isinstance(expected, (dict, list)) else str(expected),
            "metadata": {
                "domain": "native_multi_turn",
                "language": "en",
                "dialogue": dialogue,
                "expected_outcome": expected,
                "verifier_spec": verifier_spec,
                "dataset": self.dataset_name,
            },
        }

    def lift_state(self, seed: dict) -> EBIR:
        """Convert native multi-turn seed to EBIR with dialogue history."""
        meta = seed.get("metadata", {})
        dialogue = meta.get("dialogue", [])
        expected = meta.get("expected_outcome", "")

        # Build constraints from the expected outcome
        constraints = []
        if isinstance(expected, dict):
            for key, value in expected.items():
                constraints.append(StateConstraint.condition(
                    key=key,
                    params={"expected": value},
                ))
        elif isinstance(expected, str) and expected:
            constraints.append(StateConstraint.condition(
                key="expected_outcome",
                params={"expected": expected},
            ))

        # Extract facts from dialogue
        facts = []
        for msg in dialogue:
            if msg.get("role") == "user":
                facts.append(msg.get("content", ""))

        lifecycle = [LifecycleEntry(slot=c.key, label=LifecycleLabel.ACTIVE)
                     for c in constraints]

        state = DialogueState(
            goal="Complete the multi-turn dialogue task correctly.",
            facts=facts[:5],  # limit to first 5 user turns
            entities=[],
            constraints=constraints,
            output_contract=OutputContract(format="free_text"),
            verifier_spec={
                "type": "llm_judge",
                "expected_outcome": expected if isinstance(expected, str) else json.dumps(expected),
                "verifier_spec": meta.get("verifier_spec", {}),
            },
            reference={"type": "state_grounded", "outcome": expected},
            lifecycle=lifecycle,
        )

        return EBIR(
            task_id=seed["task_id"],
            domain="native_multi_turn",
            input=seed["input"],
            reference=seed.get("reference_answer", ""),
            initial_state=state,
            mutable_slots=["constraints", "facts", "environment"],
            preserved_slots=["goal", "output_contract"],
            supported_operators=[
                "partial_revision", "valid_correction",
                "scope_switch", "revoke", "constraint_accumulation",
            ],
            metadata=meta,
            seed_dialogue=dialogue,
        )

    def verify_response(self, response: str, state: DialogueState,
                        test: EvolutionTest | None = None) -> VerifierResult:
        """
        Verify response using LLM-based state-grounded judge.
        State transitions and endpoint equality remain programmatically checked.
        """
        vspec = state.verifier_spec
        if vspec.get("type") != "llm_judge":
            return VerifierResult.fail(["no LLM judge verifier configured"])

        expected = vspec.get("expected_outcome", "")

        # Use LLM judge if available
        if self._judge:
            rubric = (
                f"The response must correctly address the task given the following "
                f"expected outcome: {expected}. "
                f"Check that the response satisfies all active constraints: "
                f"{[c.to_dict() for c in state.active_constraints()]}. "
                f"The response must be coherent with the dialogue context."
            )
            return self._judge.judge(response, state, rubric=rubric)

        return VerifierResult.fail(["calibrated answer judge is required"])

    def build_canonical_probe(self, suite: EvolvedTestSuite) -> CanonicalProbe:
        """
        Build canonical probe. For native multi-turn, uses trajectory-preserving
        canonicalization when the task depends on multi-turn protocol.
        """
        final_state = suite.tests[-1].state_after if suite.tests else suite.ebir.initial_state

        # Check if the task requires multi-turn protocol
        if suite.ebir.seed_dialogue and len(suite.ebir.seed_dialogue) > 2:
            # Trajectory-preserving: keep the dialogue structure but reset to S_u
            canonical_dialogue = []
            for msg in suite.ebir.seed_dialogue:
                canonical_dialogue.append(dict(msg))
            # Add revision turns to reach final state
            for test in suite.tests:
                canonical_dialogue.append({"role": "assistant", "content": "[acknowledgment]"})
                canonical_dialogue.append({"role": "user", "content": test.user_utterance})

            return CanonicalProbe(
                prompt=self.render_state_as_prompt(final_state, suite.ebir),
                expected_behavior="Correct multi-turn dialogue completion",
                verifier_spec=final_state.verifier_spec,
                canonicalization_type="trajectory_preserving",
                trajectory=canonical_dialogue,
                burden_profile={"num_turns": len(canonical_dialogue) // 2},
                endpoint_state=final_state.semantic_dict(),
                endpoint_fingerprint=final_state.fingerprint(),
            )
        else:
            # Single-turn: present the final state directly
            return CanonicalProbe(
                prompt=self.render_state_as_prompt(final_state, suite.ebir),
                expected_behavior="Correct response satisfying all active constraints",
                verifier_spec=final_state.verifier_spec,
                canonicalization_type="single_turn",
                endpoint_state=final_state.semantic_dict(),
                endpoint_fingerprint=final_state.fingerprint(),
            )

    def recompute_endpoint(self, state: DialogueState) -> tuple[list[str], dict[str, Any]]:
        """Recompute expected outcome from updated constraints."""
        expected_parts = []
        for c in state.active_constraints():
            if c.key and c.params.get("expected") is not None:
                expected_parts.append(str(c.params["expected"]))

        expected = " | ".join(expected_parts) if expected_parts else ""
        new_spec = {
            "type": "llm_judge",
            "expected_outcome": expected,
            "verifier_spec": state.verifier_spec.get("verifier_spec", {}),
        }
        state.verifier_spec = new_spec
        return [expected], new_spec

    def execute_revision(self, state: DialogueState,
                         contract: TransitionContract) -> tuple[DialogueState, dict[str, Any]]:
        """Execute a typed revision, updating constraints and facts."""
        from ..core.state_ops import apply_contract
        new_state = apply_contract(state, contract)

        for slot in contract.deprecated_slots:
            new_state.mark_superseded(slot)

        # Recompute expected outcome if constraints changed
        if any(k.startswith(("constraints", "facts")) for k in contract.updates):
            answers, vspec = self.recompute_endpoint(new_state)
            new_state.verifier_spec = vspec

        evidence = {"contract_applied": True, "changes": list(contract.updates.keys())}
        return new_state, evidence

    def render_state_as_prompt(self, state: DialogueState, ebir: EBIR) -> str:
        """Render the multi-turn task as a self-contained prompt."""
        lines = []
        if state.goal:
            lines.append(state.goal)
        lines.append("")
        for fact in state.facts:
            lines.append(f"Context: {fact}")
        lines.append("")
        lines.append("Please provide a complete response to the task above.")
        return "\n".join(lines)
