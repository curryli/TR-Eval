"""
BenchmarkAdapter: Standard interface connecting existing benchmarks to the
construction engine.

Paper Section 4: Each adapter provides:
  1. Seed Parser: read raw benchmark samples
  2. State Lifter: convert seed to EBIR (structured initial state)
  3. Operator Registry: declare available operations
  4. Verifier Adapter: provide answer and state verification
  5. Canonicalizer: construct single-turn equivalent probe
  6. Revision Executor: apply typed revisions and recompute endpoint
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional

from ..core.ebir import (
    EBIR, DialogueState, EvolutionTest, CanonicalProbe, EvolvedTestSuite,
    StateConstraint, TransitionContract,
)
from ..core.state_ops import apply_contract
from ..evaluation.verifier import VerifierResult, LLMJudgeVerifier


class BenchmarkAdapter(ABC):
    """
    Benchmark Evolution Adapter abstract base class.

    New benchmarks only need to inherit this class and implement abstract methods.
    The construction engine only interacts with EBIR, not specific benchmarks.

    Supported verification modes:
      1. Deterministic verifier (rule checks, numeric/text matching) -- preferred
      2. LLM judge verifier (semantic rubric verification) -- supplementary
    """

    family: str = "base"
    _llm_judge: Optional[LLMJudgeVerifier] = None

    def set_llm_client(self, llm_client):
        """Inject LLM client for LLM judge verification."""
        self._llm_judge = LLMJudgeVerifier(llm_client) if llm_client else None

    @abstractmethod
    def parse_seed(self, raw_item: dict) -> dict:
        """Read raw benchmark sample, return unified seed dict."""
        ...

    @abstractmethod
    def lift_state(self, seed: dict) -> EBIR:
        """Convert seed to EBIR (structured initial state)."""
        ...

    @abstractmethod
    def verify_response(
        self,
        response: str,
        state: DialogueState,
        test: EvolutionTest | None = None,
    ) -> VerifierResult:
        """Verify model response satisfies all constraints in current state."""
        ...

    @abstractmethod
    def build_canonical_probe(self, suite: EvolvedTestSuite) -> CanonicalProbe:
        """Compress multi-turn final state into single-turn equivalent question."""
        ...

    # ----- Optional overrides for revision execution -----

    def execute_revision(
        self,
        state: DialogueState,
        contract: TransitionContract,
    ) -> tuple[DialogueState, dict[str, Any]]:
        """
        Execute a typed revision on the state.
        Returns (new_state, execution_evidence).
        Default implementation applies the contract declaratively.
        """
        new_state = apply_contract(state, contract)
        evidence = {"contract_applied": True, "changes": list(contract.updates.keys())}
        return new_state, evidence

    def recompute_endpoint(self, state: DialogueState) -> tuple[list[str], dict[str, Any]]:
        """
        Recompute accepted answers and verifier from the updated state.
        Returns (accepted_answers, verifier_spec).
        """
        return [], state.verifier_spec

    def render_user_utterance(self, test: EvolutionTest) -> str:
        """Render a user utterance for the evolution test."""
        if test.contract and test.contract.user_utterance:
            return test.contract.user_utterance
        if test.user_utterance:
            return test.user_utterance
        return f"[{test.operator}] Please update the task accordingly."

    def render_state_as_prompt(self, state: DialogueState, ebir: EBIR) -> str:
        """Render a dialogue state as a self-contained user prompt."""
        return ebir.input

    @property
    def editable_fields(self) -> list[str]:
        """Fields that can be edited by revision proposals."""
        return []

    @property
    def protected_fields(self) -> list[str]:
        """Fields that must be preserved across revisions."""
        return []
