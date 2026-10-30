"""
Code Generation Adapter: HumanEval / MBPP / BigCodeBench adapter.

Source: HumanEval
  GitHub:   https://github.com/openai/human-eval
  Format:   {"task_id": str, "prompt": str, "canonical_solution": str, "test": str}

Source: MBPP
  GitHub:   https://github.com/google-research/google-research/tree/master/mbpp
  Format:   {"task_id": int, "text": str, "code": str, "test_list": [str]}

Source: BigCodeBench
  GitHub:   https://github.com/bigcode-project/bigcodebench

Paper Table 15: Code generation derives hidden tests and mutants from the
current contract. Extend, replace, or withdraw contract fields and tests.
Reject weak tests and ambiguous preservation.

Verifier: test-case execution (provide a sandbox via the adapter interface).
Suitable operators: partial_revision, valid_correction, revoke, format_locking.

Usage:
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    seed = CodeAdapter("humaneval").parse_seed(ds[0])
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


class CodeAdapter(BenchmarkAdapter):
    """Code generation benchmark evolution adapter (HumanEval, MBPP, BigCodeBench)."""

    family = "code_generation"

    def __init__(self, dataset_name: str = "humaneval"):
        self.dataset_name = dataset_name

    @property
    def editable_fields(self) -> list[str]:
        return ["constraints", "environment", "facts"]

    @property
    def protected_fields(self) -> list[str]:
        return ["goal"]

    def parse_seed(self, raw_item: dict) -> dict:
        """
        Parse code generation sample into unified seed.
        HumanEval format: {"task_id": "HumanEval/0", "prompt": "...", "canonical_solution": "...", "test": "...", "entry_point": "..."}
        MBPP format: {"task_id": int, "text": "...", "code": "...", "test_list": [...], "test_imports": [...]}
        """
        task_id = raw_item.get("task_id", f"code_{stable_id(raw_item)}")
        prompt = raw_item.get("prompt", raw_item.get("text", ""))
        solution = raw_item.get("canonical_solution", raw_item.get("code", ""))
        test_code = raw_item.get("test", "")
        test_list = raw_item.get("test_list", [])
        entry_point = raw_item.get("entry_point", "")

        return {
            "task_id": str(task_id),
            "input": prompt,
            "reference_answer": solution,
            "metadata": {
                "domain": "code_generation",
                "language": "python",
                "solution": solution,
                "test_code": test_code,
                "test_list": test_list,
                "entry_point": entry_point,
                "dataset": self.dataset_name,
            },
        }

    def lift_state(self, seed: dict) -> EBIR:
        """Convert code seed to EBIR with function contract and test cases."""
        meta = seed.get("metadata", {})
        test_list = meta.get("test_list", [])
        test_code = meta.get("test_code", "")
        entry_point = meta.get("entry_point", "")

        constraints = []
        if test_list:
            for i, tc in enumerate(test_list):
                constraints.append(StateConstraint.condition(
                    key=f"test_case_{i}",
                    params={"test": tc, "index": i},
                ))
        elif test_code:
            constraints.append(StateConstraint.condition(
                key="test_suite",
                params={"test_code": test_code, "entry_point": entry_point},
            ))

        facts = [seed["input"]]

        lifecycle = [LifecycleEntry(slot=f"test_case_{i}", label=LifecycleLabel.ACTIVE)
                     for i in range(len(test_list))] or [LifecycleEntry(slot="test_suite", label=LifecycleLabel.ACTIVE)]

        state = DialogueState(
            goal="Write a Python function that passes all test cases.",
            facts=facts,
            entities=[entry_point] if entry_point else [],
            constraints=constraints,
            output_contract=OutputContract(format="code", required_fields=["function_definition"]),
            verifier_spec={
                "type": "code_test",
                "test_list": test_list,
                "test_code": test_code,
                "entry_point": entry_point,
            },
            reference={"type": "code", "solution": meta.get("solution", "")},
            lifecycle=lifecycle,
        )

        return EBIR(
            task_id=seed["task_id"],
            domain="code_generation",
            input=seed["input"],
            reference=seed.get("reference_answer", ""),
            initial_state=state,
            mutable_slots=["constraints", "environment", "facts"],
            preserved_slots=["goal"],
            supported_operators=[
                "partial_revision", "valid_correction",
                "revoke", "format_locking", "constraint_accumulation",
            ],
            metadata=meta,
        )

    def verify_response(self, response: str, state: DialogueState,
                        test: EvolutionTest | None = None) -> VerifierResult:
        """
        Verify code by running test cases.
        Note: Full execution requires a sandbox environment. This implementation
        performs structural checks and delegates to a sandbox if available.
        """
        vspec = state.verifier_spec
        if vspec.get("type") != "code_test":
            return VerifierResult.fail(["no code verifier configured"])

        # Structural check: response must contain a function definition
        if not re.search(r"def\s+\w+\s*\(", response):
            return VerifierResult.fail(["no function definition found in response"])

        # Try sandbox execution if available
        test_list = vspec.get("test_list", [])
        entry_point = vspec.get("entry_point", "")

        if test_list and entry_point:
            results = self._run_tests(response, test_list, entry_point)
            all_pass = all(r["passed"] for r in results)
            return VerifierResult(
                passed=all_pass,
                reasons=[] if all_pass else [r["error"] for r in results if not r["passed"]],
                constraint_results=results,
            )

        return VerifierResult.fail(["code test cases and an isolated runner are required"])

    def build_canonical_probe(self, suite: EvolvedTestSuite) -> CanonicalProbe:
        """Build single-turn probe with the final function signature and tests."""
        final_state = suite.tests[-1].state_after if suite.tests else suite.ebir.initial_state
        prompt = self._render_code_prompt(final_state)
        return CanonicalProbe(
            prompt=prompt,
            expected_behavior="Python function passing all updated test cases",
            verifier_spec=final_state.verifier_spec,
            canonicalization_type="single_turn",
            endpoint_state=final_state.semantic_dict(),
            endpoint_fingerprint=final_state.fingerprint(),
        )

    def recompute_endpoint(self, state: DialogueState) -> tuple[list[str], dict[str, Any]]:
        """Recompute test cases from the updated state's constraints."""
        test_list = []
        for c in state.active_constraints():
            if c.key and c.key.startswith("test_case_"):
                test_list.append(c.params.get("test", ""))
            elif c.key == "test_suite":
                pass  # test_code stored separately
        new_spec = {
            "type": "code_test",
            "test_list": test_list,
            "test_code": state.verifier_spec.get("test_code", ""),
            "entry_point": state.verifier_spec.get("entry_point", ""),
        }
        state.verifier_spec = new_spec
        return [state.verifier_spec.get("entry_point", "")], new_spec

    def execute_revision(self, state: DialogueState,
                         contract: TransitionContract) -> tuple[DialogueState, dict[str, Any]]:
        """Execute a typed revision, updating test cases and constraints."""
        from ..core.state_ops import apply_contract
        new_state = apply_contract(state, contract)

        for slot in contract.deprecated_slots:
            new_state.mark_superseded(slot)

        # Recompute test cases if constraints changed
        if any(k.startswith("constraints") for k in contract.updates):
            answers, vspec = self.recompute_endpoint(new_state)
            new_state.verifier_spec = vspec

        evidence = {"contract_applied": True, "changes": list(contract.updates.keys())}
        return new_state, evidence

    def render_state_as_prompt(self, state: DialogueState, ebir: EBIR) -> str:
        return self._render_code_prompt(state)

    def _render_code_prompt(self, state: DialogueState) -> str:
        """Render the code task as a self-contained prompt."""
        lines = []
        for fact in state.facts:
            lines.append(fact)
        lines.append("")
        lines.append("Write a Python function that passes all test cases.")
        return "\n".join(lines)

    @staticmethod
    def _run_tests(code: str, test_list: list[str], entry_point: str) -> list[dict]:
        """Refuse local execution until an isolated runner is integrated."""
        raise RuntimeError("No isolated code execution backend configured")
