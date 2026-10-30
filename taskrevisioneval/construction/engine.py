"""
Evolution Engine: orchestrates typed contracts over EBIR states.

Paper Section 4: The construction agent combines LLM-proposed typed revisions
with task-family adapters. The engine executes controlled transitions.
"""

from __future__ import annotations
import json
from typing import Any

from ..core.ebir import (
    DialogueState, EBIR, EvolutionTest, EvolvedTestSuite, CanonicalProbe,
    TransitionContract, LifecycleLabel, StateConstraint,
)
from ..adapters.base import BenchmarkAdapter
from ..core.state_ops import apply_contract, validate_transition
from .operators import get_operator_spec, select_operator, OPERATOR_REGISTRY


class EvolutionEngine:
    """Execute controlled transitions over EBIR states."""

    def __init__(self, llm_client=None, *, require_round_trip: bool = True):
        self._llm = llm_client
        self._require_round_trip = require_round_trip

    def evolve(
        self,
        ebir: EBIR,
        adapter: BenchmarkAdapter,
        num_turns: int = 1,
        strategy: str = "coverage",
        discovery_model: str | None = None,
        model_hint: str | None = None,
        force_operator: str | None = None,
    ) -> EvolvedTestSuite:
        """
        Evolve a seed through atomic, admitted transitions.

        Paper Section 4: The primary protocol is one transition.
        num_turns > 1 is supported as a separate stress track.
        force_operator: if set, use this operator directly (bypass select_operator).
        """
        adapter.set_llm_client(self._llm)
        suite = EvolvedTestSuite(
            task_id=ebir.task_id, ebir=ebir, source=strategy,
            discovery_model=discovery_model or "",
            metadata={"track": "primary" if num_turns == 1 else "multi_step_stress",
                       "requested_turns": num_turns},
        )
        suite.dialogue = self._initial_dialogue(ebir)

        current_state = ebir.initial_state.clone()
        used_operators: list[str] = []

        for turn_index in range(1, max(0, num_turns) + 1):
            if force_operator:
                operator = force_operator
            else:
                available = self._allowed_operators(ebir, adapter, current_state)
                operator = select_operator(current_state, available, strategy=strategy,
                                            model_hint=model_hint, used_operators=used_operators)
            if operator is None:
                suite.metadata.setdefault("generation_rejections", []).append(
                    {"turn": turn_index, "reason": "no applicable operator"})
                break
            used_operators.append(operator)

            test, rejection = self._build_test(
                ebir, adapter, current_state, operator, turn_index
            )
            if rejection:
                suite.metadata.setdefault("generation_rejections", []).append(rejection)
                break
            if test:
                suite.tests.append(test)
                current_state = test.state_after

        # Build canonical probe from the final state
        if suite.tests:
            suite.canonical_probe = adapter.build_canonical_probe(suite)

        return suite

    def _initial_dialogue(self, ebir: EBIR) -> list[dict]:
        if ebir.seed_dialogue:
            return [dict(m) for m in ebir.seed_dialogue]
        return [{"role": "user", "content": ebir.input}]

    def _allowed_operators(self, ebir: EBIR, adapter: BenchmarkAdapter,
                           state: DialogueState) -> list[str]:
        if ebir.supported_operators:
            return ebir.supported_operators
        return list(OPERATOR_REGISTRY.keys())

    def _build_test(
        self, ebir: EBIR, adapter: BenchmarkAdapter, state: DialogueState,
        operator: str, turn_index: int,
    ) -> tuple[EvolutionTest | None, dict | None]:
        spec = get_operator_spec(operator)
        if spec is None:
            return None, {"turn": turn_index, "reason": f"unknown operator: {operator}"}

        # Build transition contract
        contract = self._propose_contract(ebir, adapter, state, spec, turn_index)
        if contract is None:
            return None, {"turn": turn_index, "reason": "contract proposal failed"}

        # Validate contract
        if not contract.is_valid():
            return None, {"turn": turn_index, "reason": "invalid contract"}

        # Execute revision
        try:
            new_state, evidence = adapter.execute_revision(state, contract)
        except Exception as e:
            return None, {"turn": turn_index, "reason": f"execution failed: {e}"}

        # Validate transition
        violations = validate_transition(state, new_state, contract)
        if violations:
            return None, {"turn": turn_index, "reason": f"transition violations: {violations}"}

        # Render user utterance
        test = EvolutionTest(
            operator=operator, turn_index=turn_index,
            state_before=state, state_after=new_state,
            allowed_change={f: True for f in spec.allowed_change_fields},
            must_preserve=spec.must_preserve_fields,
            must_not_preserve=spec.must_not_preserve_fields,
            user_utterance=contract.user_utterance,
            verifier_spec=new_state.verifier_spec,
            target_weakness=spec.target_weakness,
            contract=contract,
        )

        # Update dialogue
        test.user_utterance = adapter.render_user_utterance(test)

        return test, None

    def _propose_contract(
        self, ebir: EBIR, adapter: BenchmarkAdapter, state: DialogueState,
        spec, turn_index: int,
    ) -> TransitionContract | None:
        """
        Propose a transition contract for the given operator.

        If an LLM client is available, use it to generate the contract.
        Otherwise, use a deterministic fallback based on the adapter's editable fields.
        """
        if self._llm:
            return self._llm_propose_contract(ebir, adapter, state, spec, turn_index)
        return self._deterministic_contract(ebir, adapter, state, spec, turn_index)

    def _llm_propose_contract(
        self, ebir: EBIR, adapter: BenchmarkAdapter, state: DialogueState,
        spec, turn_index: int,
    ) -> TransitionContract | None:
        """Use LLM to propose a typed revision contract."""
        prompt = self._build_proposal_prompt(ebir, state, spec)
        try:
            result = self._llm.chat_json(prompt)
            contract = TransitionContract.from_dict(result)
            contract.operator = spec.name
            return contract
        except Exception:
            return self._deterministic_contract(ebir, adapter, state, spec, turn_index)

    def _deterministic_contract(
        self, ebir: EBIR, adapter: BenchmarkAdapter, state: DialogueState,
        spec, turn_index: int,
    ) -> TransitionContract | None:
        """Deterministic fallback: create a valid contract with actual state changes."""
        editable = adapter.editable_fields or spec.allowed_change_fields
        if not editable:
            return None

        target_field = editable[0]
        updates = self._generate_updates(state, target_field, spec, turn_index)
        if not updates:
            return None

        # For deprecated_slots, only include specific lifecycle slot names that
        # are actually being replaced, not generic container field names.
        deprecated = []
        for field in (spec.must_not_preserve_fields or []):
            if field in ("constraints", "facts", "entities", "environment"):
                # Container field: deprecate specific slots that exist in the state
                for entry in state.lifecycle:
                    if entry.label == LifecycleLabel.ACTIVE and entry.slot not in deprecated:
                        deprecated.append(entry.slot)
                        if len(deprecated) >= 1:
                            break
            else:
                deprecated.append(field)

        return TransitionContract(
            operator=spec.name,
            mutable_slots=[target_field],
            preserved_slots=spec.must_preserve_fields or ebir.preserved_slots,
            deprecated_slots=deprecated,
            precondition=f"State has field '{target_field}' that can be modified",
            transition_fn=f"Modify '{target_field}' via {spec.name}",
            oracle_transport="Recompute accepted answers from updated state",
            invariants=[f"Field '{f}' must remain unchanged" for f in (spec.must_preserve_fields or [])],
            user_utterance=f"[{spec.name}] Please update {target_field}.",
            updates=updates,
        )

    def _generate_updates(
        self, state: DialogueState, target_field: str, spec, turn_index: int,
    ) -> dict[str, Any]:
        """Generate actual state updates for the target field based on revision type."""
        if target_field == "facts":
            if spec.revision_type == "replace":
                return {"facts": state.facts + [f"[Updated turn {turn_index}] Revised parameter value."]}
            elif spec.revision_type == "withdraw":
                return {"facts": state.facts[:-1] if len(state.facts) > 1 else state.facts + ["[withdrawn]"]}
            else:
                return {"facts": state.facts + [f"[Added turn {turn_index}] New information."]}
        elif target_field == "constraints":
            new_constraint = {
                "type": "custom",
                "key": f"revised_constraint_{turn_index}",
                "params": {"source": spec.name, "turn": turn_index},
                "instruction_id": "",
                "kwargs": {},
            }
            if spec.revision_type == "replace":
                return {"constraints": [new_constraint] + [c.to_dict() for c in state.constraints[1:]]}
            elif spec.revision_type == "withdraw":
                return {"constraints": [c.to_dict() for c in state.constraints[1:]] if len(state.constraints) > 1 else []}
            else:
                return {"constraints": [c.to_dict() for c in state.constraints] + [new_constraint]}
        elif target_field == "environment":
            return {"environment.modified_param": f"revised_value_{turn_index}"}
        elif target_field == "output_contract":
            return {"output_contract.constraints.revision_tag": f"turn_{turn_index}"}
        return {}

    def _build_proposal_prompt(self, ebir: EBIR, state: DialogueState, spec) -> str:
        return (
            f"Propose a typed revision contract for the following task.\n\n"
            f"Operator: {spec.name} ({spec.description})\n"
            f"Revision type: {spec.revision_type}\n"
            f"Allowed changes: {spec.allowed_change_fields}\n"
            f"Must preserve: {spec.must_preserve_fields}\n"
            f"Must invalidate: {spec.must_not_preserve_fields}\n\n"
            f"Current state:\n{json.dumps(state.semantic_dict(), indent=2, default=str)}\n\n"
            f"Return JSON with fields: mutable_slots, preserved_slots, deprecated_slots, "
            f"precondition, transition_fn, oracle_transport, invariants, user_utterance, updates"
        )
