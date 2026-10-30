"""
Math Adapter: Math/Reasoning benchmark adapter (MATH-500, GSM8K).

Source: MATH-500
  Dataset:  https://huggingface.co/datasets/HuggingFaceH4/MATH-500
  Format:   {"problem": str, "solution": str, "answer": str}

Source: GSM8K
  GitHub:   https://github.com/openai/grade-school-math
  Format:   {"question": str, "answer": str}  (answer ends with "#### <number>")

Paper Table 15: Math/reasoning uses symbolic or programmatic solver with
numeric answer recomputation. Reveals, replaces, or withdraws values, facts,
and constraints. Rejects changed solvability and hidden dependency roots.

Verifier: numeric answer extraction and exact-match comparison.
Suitable operators: partial_revision, valid_correction, scope_switch, revoke.

Usage:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    seed = MathAdapter("gsm8k").parse_seed(ds[0])
"""

from __future__ import annotations
import re
from typing import Any
from ..core.ids import stable_id

from .base import BenchmarkAdapter
from ..core.ebir import (
    EBIR, DialogueState, EvolutionTest, CanonicalProbe, EvolvedTestSuite,
    StateConstraint, OutputContract, LifecycleLabel, LifecycleEntry,
    TransitionContract,
)
from ..evaluation.verifier import VerifierResult, verify_numeric_answer


class MathAdapter(BenchmarkAdapter):
    """Math/reasoning benchmark evolution adapter (GSM8K, MATH-500)."""

    family = "math_reasoning"

    def __init__(self, dataset_name: str = "gsm8k"):
        self.dataset_name = dataset_name

    @property
    def editable_fields(self) -> list[str]:
        return ["facts", "constraints", "environment"]

    @property
    def protected_fields(self) -> list[str]:
        return ["goal", "output_contract"]

    def parse_seed(self, raw_item: dict) -> dict:
        """
        Parse math benchmark sample into unified seed.
        GSM8K format: {"question": "...", "answer": "...#### 72"}
        MATH-500 format: {"problem": "...", "solution": "...", "answer": "72"}
        """
        question = raw_item.get("question", raw_item.get("problem", ""))
        answer = raw_item.get("answer", raw_item.get("solution", ""))
        final_answer = self._extract_final_answer(answer)

        return {
            "task_id": f"math_{stable_id(question)}",
            "input": question,
            "reference_answer": str(final_answer) if final_answer is not None else "",
            "metadata": {
                "domain": "math",
                "language": "en",
                "full_answer": answer,
                "final_answer": final_answer,
                "dataset": self.dataset_name,
            },
        }

    def lift_state(self, seed: dict) -> EBIR:
        """Convert math seed to EBIR with numeric answer constraint."""
        meta = seed.get("metadata", {})
        final_answer = meta.get("final_answer")

        facts = self._extract_facts(seed["input"])

        constraints = []
        if final_answer is not None:
            constraints.append(StateConstraint.condition(
                key="answer_value",
                params={"expected": str(final_answer), "tolerance": 0.001},
            ))

        lifecycle = [LifecycleEntry(slot="answer_value", label=LifecycleLabel.ACTIVE)]

        state = DialogueState(
            goal="Solve the math problem and provide the correct numerical answer.",
            facts=facts,
            entities=[],
            constraints=constraints,
            output_contract=OutputContract(format="free_text"),
            verifier_spec={
                "type": "numeric_match",
                "expected": str(final_answer) if final_answer is not None else "",
                "tolerance": 0.001,
            },
            reference={"type": "numeric", "answer": str(final_answer) if final_answer else ""},
            lifecycle=lifecycle,
        )

        return EBIR(
            task_id=seed["task_id"],
            domain="math_reasoning",
            input=seed["input"],
            reference=seed.get("reference_answer", ""),
            initial_state=state,
            mutable_slots=["facts", "constraints", "environment"],
            preserved_slots=["goal", "output_contract"],
            supported_operators=[
                "partial_revision", "valid_correction",
                "scope_switch", "revoke", "recovery_probe",
            ],
            metadata=meta,
        )

    def verify_response(self, response: str, state: DialogueState,
                        test: EvolutionTest | None = None) -> VerifierResult:
        """Verify numeric answer by extraction and comparison."""
        vspec = state.verifier_spec
        if vspec.get("type") == "numeric_match":
            return verify_numeric_answer(
                response, vspec.get("expected", ""),
                tolerance=vspec.get("tolerance", 0.001),
            )
        return VerifierResult.fail(["no numeric verifier configured"])

    def build_canonical_probe(self, suite: EvolvedTestSuite) -> CanonicalProbe:
        """Build single-turn probe with the final problem statement."""
        final_state = suite.tests[-1].state_after if suite.tests else suite.ebir.initial_state
        prompt = self._render_problem_prompt(final_state)
        return CanonicalProbe(
            prompt=prompt,
            expected_behavior="Correct numerical answer for the updated problem",
            verifier_spec=final_state.verifier_spec,
            canonicalization_type="single_turn",
            endpoint_state=final_state.semantic_dict(),
            endpoint_fingerprint=final_state.fingerprint(),
        )

    def recompute_endpoint(self, state: DialogueState) -> tuple[list[str], dict[str, Any]]:
        """Recompute the expected answer from the updated state's constraints."""
        expected = ""
        for c in state.active_constraints():
            if c.key == "answer_value":
                expected = str(c.params.get("expected", ""))
        new_spec = {"type": "numeric_match", "expected": expected, "tolerance": 0.001}
        state.verifier_spec = new_spec
        return [expected], new_spec

    def execute_revision(self, state: DialogueState,
                         contract: TransitionContract) -> tuple[DialogueState, dict[str, Any]]:
        """Execute a typed revision, updating facts and constraints."""
        from ..core.state_ops import apply_contract
        new_state = apply_contract(state, contract)

        for slot in contract.deprecated_slots:
            new_state.mark_superseded(slot)

        # Recompute answer if facts or constraints changed
        if any(k.startswith(("facts", "constraints")) for k in contract.updates):
            answers, vspec = self.recompute_endpoint(new_state)
            new_state.verifier_spec = vspec

        evidence = {"contract_applied": True, "changes": list(contract.updates.keys())}
        return new_state, evidence

    def render_state_as_prompt(self, state: DialogueState, ebir: EBIR) -> str:
        """Render the math problem as a self-contained prompt."""
        return self._render_problem_prompt(state)

    def _render_problem_prompt(self, state: DialogueState) -> str:
        parts = []
        for fact in state.facts:
            parts.append(fact)
        prompt = ". ".join(parts) if parts else ""
        if state.goal:
            prompt = f"{prompt}. {state.goal}" if prompt else state.goal
        return prompt.strip()

    @staticmethod
    def _extract_final_answer(answer: str) -> str | None:
        """Extract the final numeric answer from a solution string."""
        # GSM8K: answer ends with "#### 72"
        match = re.search(r"####\s*([\d,.\-]+)", answer)
        if match:
            return match.group(1).replace(",", "")
        # MATH-500: answer in \boxed{...}
        match = re.search(r"\\boxed\{([^}]+)\}", answer)
        if match:
            return match.group(1).strip()
        # Fallback: last number in the text
        numbers = re.findall(r"-?\d+\.?\d*", answer)
        if numbers:
            return numbers[-1]
        return None

    @staticmethod
    def _extract_facts(question: str) -> list[str]:
        """Extract facts from the question using simple heuristics."""
        sentences = re.split(r"[.!?]+", question)
        facts = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
        return facts[:10]  # limit to 10 facts
