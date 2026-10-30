"""
Tool Use Adapter: BFCL / ToolSandbox benchmark adapter.

Source: BFCL
  GitHub:   https://github.com/ShishirPatil/gorilla
  Format:   {"id": int, "question": str, "function": [...], "ground_truths": [...]}

Source: ToolSandbox
  GitHub:   https://github.com/apple-aiml-research/ToolSandbox

Paper Table 15: Tool use validates calls and resettable state. Add, replace,
or withdraw schemas, bindings, contracts, or requirements. Reject nondeterminism
and irreversible side effects.

Verifier: tool call parameter validation against expected schema.
Suitable operators: partial_revision, scope_switch, revoke, valid_correction.

Usage:
    from datasets import load_dataset
    ds = load_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard", split="test")
    seed = ToolUseAdapter("bfcl").parse_seed(ds[0])
"""

from __future__ import annotations
import json
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


class ToolUseAdapter(BenchmarkAdapter):
    """Tool use benchmark evolution adapter (BFCL, ToolSandbox)."""

    family = "tool_use"

    def __init__(self, dataset_name: str = "bfcl"):
        self.dataset_name = dataset_name

    @property
    def editable_fields(self) -> list[str]:
        return ["environment", "constraints", "facts"]

    @property
    def protected_fields(self) -> list[str]:
        return ["goal"]

    def parse_seed(self, raw_item: dict) -> dict:
        """
        Parse tool use sample into unified seed.
        BFCL format: {"id": int, "question": "...", "function": [...], "ground_truths": [...]}
        ToolSandbox format: {"task_id": ..., "user_input": ..., "expected_tool_calls": [...], "tools": [...]}
        """
        task_id = str(raw_item.get("id", raw_item.get("task_id", stable_id(raw_item))))
        question = raw_item.get("question", raw_item.get("query", raw_item.get("user_input", "")))
        tools = raw_item.get("function", raw_item.get("tools", []))
        ground_truths = raw_item.get("ground_truths", raw_item.get("expected_tool_calls", []))

        return {
            "task_id": f"tool_{task_id}",
            "input": question,
            "reference_answer": json.dumps(ground_truths) if ground_truths else "",
            "metadata": {
                "domain": "tool_use",
                "language": "en",
                "tools": tools,
                "ground_truths": ground_truths,
                "dataset": self.dataset_name,
            },
        }

    def lift_state(self, seed: dict) -> EBIR:
        """Convert tool use seed to EBIR with tool schemas and expected calls."""
        meta = seed.get("metadata", {})
        tools = meta.get("tools", [])
        ground_truths = meta.get("ground_truths", [])

        constraints = []
        for i, gt in enumerate(ground_truths):
            if isinstance(gt, dict):
                constraints.append(StateConstraint.condition(
                    key=f"expected_call_{i}",
                    params={"tool": gt.get("name", gt.get("function", "")),
                            "arguments": gt.get("arguments", {})},
                ))

        facts = [seed["input"]]

        lifecycle = [LifecycleEntry(slot=f"expected_call_{i}", label=LifecycleLabel.ACTIVE)
                     for i in range(len(ground_truths))]

        state = DialogueState(
            goal="Make the correct tool call(s) to complete the task.",
            facts=facts,
            entities=[],
            constraints=constraints,
            output_contract=OutputContract(format="json", required_fields=["tool_call"]),
            verifier_spec={
                "type": "tool_call_validation",
                "ground_truths": ground_truths,
                "tools": tools,
            },
            reference={"type": "tool_calls", "ground_truths": ground_truths},
            lifecycle=lifecycle,
        )

        return EBIR(
            task_id=seed["task_id"],
            domain="tool_use",
            input=seed["input"],
            reference=seed.get("reference_answer", ""),
            initial_state=state,
            mutable_slots=["environment", "constraints", "facts"],
            preserved_slots=["goal"],
            supported_operators=[
                "partial_revision", "scope_switch",
                "valid_correction", "revoke", "constraint_accumulation",
            ],
            metadata=meta,
        )

    def verify_response(self, response: str, state: DialogueState,
                        test: EvolutionTest | None = None) -> VerifierResult:
        """Verify tool call parameters against expected ground truths."""
        vspec = state.verifier_spec
        if vspec.get("type") != "tool_call_validation":
            return VerifierResult.fail(["no tool call verifier configured"])

        ground_truths = vspec.get("ground_truths", [])
        if not ground_truths:
            return VerifierResult.fail(["no expected tool calls configured"])

        # Extract tool calls from response (JSON or structured format)
        tool_calls = self._extract_tool_calls(response)

        if not tool_calls:
            return VerifierResult.fail(["no tool calls found in response"])

        # Match each ground truth against extracted calls
        results = []
        all_pass = True
        for i, gt in enumerate(ground_truths):
            matched = False
            expected_name = gt.get("name", gt.get("function", ""))
            expected_args = gt.get("arguments", {})

            for tc in tool_calls:
                tc_name = tc.get("name", tc.get("function", ""))
                tc_args = tc.get("arguments", tc.get("parameters", {}))
                if tc_name == expected_name:
                    # Check arguments match
                    if self._args_match(expected_args, tc_args):
                        matched = True
                        break

            results.append({"expected": str(gt), "matched": matched})
            if not matched:
                all_pass = False

        reasons = [] if all_pass else [f"ground truth {i} not matched" for i, r in enumerate(results) if not r["matched"]]
        return VerifierResult(passed=all_pass, reasons=reasons, constraint_results=results)

    def build_canonical_probe(self, suite: EvolvedTestSuite) -> CanonicalProbe:
        """Build single-turn probe with the final task and tool schemas."""
        final_state = suite.tests[-1].state_after if suite.tests else suite.ebir.initial_state
        prompt = self._render_tool_prompt(final_state)
        return CanonicalProbe(
            prompt=prompt,
            expected_behavior="Correct tool call(s) matching expected ground truths",
            verifier_spec=final_state.verifier_spec,
            canonicalization_type="single_turn",
            endpoint_state=final_state.semantic_dict(),
            endpoint_fingerprint=final_state.fingerprint(),
        )

    def recompute_endpoint(self, state: DialogueState) -> tuple[list[str], dict[str, Any]]:
        """Recompute expected tool calls from updated constraints."""
        ground_truths = []
        for c in state.active_constraints():
            if c.key and c.key.startswith("expected_call_"):
                ground_truths.append({
                    "name": c.params.get("tool", ""),
                    "arguments": c.params.get("arguments", {}),
                })
        new_spec = {
            "type": "tool_call_validation",
            "ground_truths": ground_truths,
            "tools": state.verifier_spec.get("tools", []),
        }
        state.verifier_spec = new_spec
        return [json.dumps(ground_truths)], new_spec

    def execute_revision(self, state: DialogueState,
                         contract: TransitionContract) -> tuple[DialogueState, dict[str, Any]]:
        """Execute a typed revision, updating tool schemas and expected calls."""
        from ..core.state_ops import apply_contract
        new_state = apply_contract(state, contract)

        for slot in contract.deprecated_slots:
            new_state.mark_superseded(slot)

        if any(k.startswith(("constraints", "environment")) for k in contract.updates):
            answers, vspec = self.recompute_endpoint(new_state)
            new_state.verifier_spec = vspec

        evidence = {"contract_applied": True, "changes": list(contract.updates.keys())}
        return new_state, evidence

    def render_state_as_prompt(self, state: DialogueState, ebir: EBIR) -> str:
        return self._render_tool_prompt(state)

    def _render_tool_prompt(self, state: DialogueState) -> str:
        """Render the tool-use task as a self-contained prompt."""
        lines = []
        for fact in state.facts:
            lines.append(fact)
        lines.append("")
        lines.append("Make the appropriate tool call(s) to complete the task.")
        return "\n".join(lines)

    @staticmethod
    def _extract_tool_calls(response: str) -> list[dict]:
        """Extract tool calls from a model response."""
        # Try JSON parsing first
        try:
            parsed = json.loads(response.strip())
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict) and "tool_calls" in parsed:
                return parsed["tool_calls"]
            if isinstance(parsed, dict) and ("name" in parsed or "function" in parsed):
                return [parsed]
        except json.JSONDecodeError:
            pass

        # Try to find JSON-like blocks
        matches = re.findall(r'\{[^{}]*"name"[^{}]*\}', response, re.DOTALL)
        if matches:
            results = []
            for m in matches:
                try:
                    results.append(json.loads(m))
                except json.JSONDecodeError:
                    pass
            if results:
                return results

        return []

    @staticmethod
    def _args_match(expected: dict, actual: dict) -> bool:
        """Check if tool call arguments match (subset match)."""
        if not expected:
            return True
        for k, v in expected.items():
            if k not in actual:
                return False
            if isinstance(v, str) and isinstance(actual[k], str):
                if v.casefold() != actual[k].casefold():
                    return False
            elif v != actual[k]:
                return False
        return True
