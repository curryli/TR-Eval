"""
Structured QA Adapter: MMLU-Pro / TruthfulQA benchmark adapter.

Source: MMLU-Pro
  GitHub:   https://github.com/TIGER-AI-Lab/MMLU-Pro
  Format:   {"question": str, "options": [str], "answer": str, "subject": str}

Source: TruthfulQA
  GitHub:   https://github.com/sylinrl/TruthfulQA

Paper Table 15: Structured QA regenerates answers, evidence, and option mappings.
Rejects ambiguous evidence and label shortcuts. Evidence, options, and labels
can be introduced, replaced, or withdrawn.

Verifier: multiple-choice answer matching (letter or content extraction).
Suitable operators: partial_revision, scope_switch, valid_correction, revoke.

Usage:
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    seed = StructuredQAAdapter("mmlu_pro").parse_seed(ds[0])
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
from ..evaluation.verifier import VerifierResult


class StructuredQAAdapter(BenchmarkAdapter):
    """Structured QA / MCQ benchmark evolution adapter (MMLU-Pro, TruthfulQA)."""

    family = "structured_qa"

    def __init__(self, dataset_name: str = "mmlu_pro"):
        self.dataset_name = dataset_name

    @property
    def editable_fields(self) -> list[str]:
        return ["facts", "constraints", "environment"]

    @property
    def protected_fields(self) -> list[str]:
        return ["goal", "output_contract"]

    def parse_seed(self, raw_item: dict) -> dict:
        """
        Parse structured QA sample into unified seed.
        MMLU-Pro format: {"question_id": int, "question": "...", "options": [...], "answer": "D", "answer_value": "..."}
        TruthfulQA format: {"qid": int, "question": "...", "mc1_targets": {...}, "mc2_targets": {...}}
        """
        question = raw_item.get("question", "")
        options = raw_item.get("options", [])
        answer = raw_item.get("answer", "")
        answer_value = raw_item.get("answer_value", "")

        return {
            "task_id": f"structqa_{raw_item.get('question_id', raw_item.get('qid', stable_id(question)))}",
            "input": question,
            "reference_answer": answer,
            "metadata": {
                "domain": "structured_qa",
                "language": "en",
                "options": options,
                "answer": answer,
                "answer_value": answer_value,
                "dataset": self.dataset_name,
            },
        }

    def lift_state(self, seed: dict) -> EBIR:
        """Convert structured QA seed to EBIR."""
        meta = seed.get("metadata", {})
        options = meta.get("options", [])
        answer = meta.get("answer", "")

        constraints = []
        if answer:
            constraints.append(StateConstraint.condition(
                key="correct_option",
                params={"expected": answer, "options": options},
            ))

        facts = [seed["input"]]
        if options:
            for i, opt in enumerate(options):
                letter = chr(65 + i)
                facts.append(f"Option {letter}: {opt}")

        lifecycle = [LifecycleEntry(slot="correct_option", label=LifecycleLabel.ACTIVE)]

        state = DialogueState(
            goal="Select the correct answer from the given options.",
            facts=facts,
            entities=[],
            constraints=constraints,
            output_contract=OutputContract(format="short_answer", required_fields=["answer_letter"]),
            verifier_spec={
                "type": "mcq_match",
                "expected": answer,
                "options": options,
            },
            reference={"type": "mcq", "answer": answer},
            lifecycle=lifecycle,
        )

        return EBIR(
            task_id=seed["task_id"],
            domain="structured_qa",
            input=seed["input"],
            reference=seed.get("reference_answer", ""),
            initial_state=state,
            mutable_slots=["facts", "constraints", "environment"],
            preserved_slots=["goal", "output_contract"],
            supported_operators=[
                "partial_revision", "scope_switch",
                "valid_correction", "revoke",
            ],
            metadata=meta,
        )

    def verify_response(self, response: str, state: DialogueState,
                        test: EvolutionTest | None = None) -> VerifierResult:
        """Verify multiple-choice answer by letter or content matching."""
        vspec = state.verifier_spec
        if vspec.get("type") != "mcq_match":
            return VerifierResult.fail(["no MCQ verifier configured"])

        expected = vspec.get("expected", "")
        options = vspec.get("options", [])

        # Extract answer letter from response
        response_stripped = response.strip()

        # Direct letter match (A, B, C, D)
        letter_match = re.search(r'\b([A-Z])\b', response_stripped)
        if letter_match:
            actual = letter_match.group(1)
            passed = actual.upper() == expected.upper()
            return VerifierResult(
                passed=passed,
                reasons=[] if passed else [f"answer '{actual}' != expected '{expected}'"],
                details={"actual": actual, "expected": expected},
            )

        # Content match against option text
        for i, opt in enumerate(options):
            letter = chr(65 + i)
            if opt and opt.strip().lower() in response_stripped.lower():
                passed = letter.upper() == expected.upper()
                return VerifierResult(
                    passed=passed,
                    reasons=[] if passed else [f"matched option {letter} != expected {expected}"],
                    details={"actual": letter, "expected": expected},
                )

        return VerifierResult.fail([f"could not extract answer from response; expected '{expected}'"])

    def build_canonical_probe(self, suite: EvolvedTestSuite) -> CanonicalProbe:
        """Build single-turn probe with the final question and options."""
        final_state = suite.tests[-1].state_after if suite.tests else suite.ebir.initial_state
        prompt = self._render_qa_prompt(final_state)
        return CanonicalProbe(
            prompt=prompt,
            expected_behavior="Correct answer letter matching the updated question",
            verifier_spec=final_state.verifier_spec,
            canonicalization_type="single_turn",
            endpoint_state=final_state.semantic_dict(),
            endpoint_fingerprint=final_state.fingerprint(),
        )

    def recompute_endpoint(self, state: DialogueState) -> tuple[list[str], dict[str, Any]]:
        """Recompute the expected answer from updated constraints."""
        expected = ""
        options = []
        for c in state.active_constraints():
            if c.key == "correct_option":
                expected = str(c.params.get("expected", ""))
                options = c.params.get("options", [])
        new_spec = {"type": "mcq_match", "expected": expected, "options": options}
        state.verifier_spec = new_spec
        return [expected], new_spec

    def execute_revision(self, state: DialogueState,
                         contract: TransitionContract) -> tuple[DialogueState, dict[str, Any]]:
        """Execute a typed revision, updating facts/options and constraints."""
        from ..core.state_ops import apply_contract
        new_state = apply_contract(state, contract)

        for slot in contract.deprecated_slots:
            new_state.mark_superseded(slot)

        # Regenerate answer mapping if options changed
        if any(k.startswith("facts") for k in contract.updates):
            answers, vspec = self.recompute_endpoint(new_state)
            new_state.verifier_spec = vspec

        evidence = {"contract_applied": True, "changes": list(contract.updates.keys())}
        return new_state, evidence

    def render_state_as_prompt(self, state: DialogueState, ebir: EBIR) -> str:
        return self._render_qa_prompt(state)

    def _render_qa_prompt(self, state: DialogueState) -> str:
        """Render the question with all options as a self-contained prompt."""
        lines = []
        for fact in state.facts:
            lines.append(fact)
        lines.append("")
        lines.append("Provide the correct answer letter (A, B, C, D, etc.).")
        return "\n".join(lines)
