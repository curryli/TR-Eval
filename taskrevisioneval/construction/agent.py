"""
Budgeted Verifier-Grounded Construction Agent.

Implements paper Algorithm 1: Fixed-budget compilation of paired packets
in the First-valid and Challenge suites from public benchmark seeds.

The agent coordinates four functional blocks:
  1. Task initialization → verified history h (from public seeds via adapters)
  2. Heterogeneous proposal → ordered typed candidates C
  3. Execution, packet verification, and F → ordered auto-pass queue Q
  4. Diagnostic selection → paired packets (u_first, u_chal)

Source benchmarks: IFEval, MATH-500, GSM8K, MMLU-Pro, TruthfulQA,
HumanEval, MBPP, BigCodeBench, BFCL, ToolSandbox, MultiChallenge, MT-Bench.
All seeds are publicly available; see README.md for HuggingFace paths.
"""

from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.ebir import (
    EBIR, DialogueState, EvolutionTest, EvolvedTestSuite, CanonicalProbe,
    TerminalPacket, PresentationPath, TransitionContract,
    RevisionType, RevisionDepth, SelectionLabel,
)
from ..adapters.base import BenchmarkAdapter
from ..core.state_ops import apply_contract, changed_paths
from .operators import get_operator_spec, get_revision_type, is_task_revision, OPERATOR_REGISTRY
from .engine import EvolutionEngine
from .admission import AdmissionGate


@dataclass
class ConstructionResult:
    """Result of compiling one initial instance (paper Algorithm 1)."""
    instance_id: str
    family: str
    packets: list[TerminalPacket] = field(default_factory=list)
    selected_packet: Optional[TerminalPacket] = None   # Challenge suite member
    shadow_packet: Optional[TerminalPacket] = None      # First-valid suite member
    rejection_reason: str = ""
    attempted: int = 0
    admissible: int = 0
    diagnostic_hit: bool = False
    trace: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id, "family": self.family,
            "packets": [p.to_dict() for p in self.packets],
            "selected_packet": self.selected_packet.to_dict() if self.selected_packet else None,
            "shadow_packet": self.shadow_packet.to_dict() if self.shadow_packet else None,
            "rejection_reason": self.rejection_reason,
            "attempted": self.attempted, "admissible": self.admissible,
            "diagnostic_hit": self.diagnostic_hit,
            "trace": self.trace,
        }


class ConstructionAgent:
    """
    Paper Algorithm 1: Budgeted verifier-grounded construction.

    For each initialization source:
      Stage 1: Task initialization and revision proposal
      Stage 2: Execution and verification (budget K candidates)
      Stage 3: Diagnostic selection and provisional retention

    Produces paired packets: u_first (score-independent) and u_chal (Challenge).
    """

    def __init__(
        self,
        adapter: BenchmarkAdapter,
        llm_client=None,
        construction_target_model: str = "",
        k_budget: int = 5,          # total candidate budget across proposers
        alpha: float = 0.5,         # control-success floor
        gamma: float = 0.1,         # min I-R drop for diagnostic hit
        max_multi_revisions: int = 3,
    ):
        self._adapter = adapter
        self._llm = llm_client
        self._construction_target = construction_target_model
        self._k_budget = k_budget
        self._alpha = alpha
        self._gamma = gamma
        self._max_multi_revisions = max_multi_revisions
        self._engine = EvolutionEngine(llm_client)
        self._gate = AdmissionGate(adapter, llm_client)

    @staticmethod
    def prescreen_instance(ebir: EBIR) -> tuple[bool, str]:
        """
        Paper Section 4.2: Instance pre-screening before compilation.
        Filters out seeds unlikely to produce valid revision packets.
        """
        if not ebir.mutable_slots:
            return False, "no_mutable_slots"
        state = ebir.initial_state
        if not state.goal and not ebir.input:
            return False, "empty_goal"
        if not state.constraints and not state.facts and not state.environment:
            return False, "empty_state"
        if not state.reference and not state.verifier_spec:
            return False, "no_reference_or_verifier"
        if len(ebir.input) < 10:
            return False, "trivial_input"
        return True, "passed"

    def construct(
        self,
        ebir: EBIR,
        num_turns: int = 1,
        strategy: str = "coverage",
    ) -> ConstructionResult:
        """
        Execute the full construction pipeline for one initialization source.

        Paper Algorithm 1:
          1. Initialize task and propose revisions (up to K candidates)
          2. Execute each candidate, verify transition, compile packet
          3. Select first Diagnostic-Hit or first admissible as fallback
        """
        result = ConstructionResult(
            instance_id=ebir.task_id, family=self._adapter.family,
        )

        # Pre-screen
        passes, reason = self.prescreen_instance(ebir)
        if not passes:
            result.rejection_reason = f"prescreen: {reason}"
            return result

        # Stage 1: Generate candidate suites
        candidates: list[EvolvedTestSuite] = []
        operators_to_try = self._select_operators(ebir, strategy)

        for op_name in operators_to_try[:self._k_budget]:
            suite = self._engine.evolve(
                ebir, self._adapter,
                num_turns=num_turns,
                strategy="targeted",
                force_operator=op_name,
                discovery_model=self._construction_target,
            )
            if suite.tests:
                # Check if this is a task revision (Replace or Withdraw)
                has_revision = any(is_task_revision(t.operator) for t in suite.tests)
                if has_revision or num_turns > 1:
                    candidates.append(suite)
            result.attempted += 1

        if not candidates:
            result.rejection_reason = "no candidates with revisions generated"
            return result

        # Stage 2: Execute, verify, and compile packets
        packets: list[TerminalPacket] = []
        for suite in candidates:
            # Admission check
            admitted, reasons = self._gate.check(suite)
            if not admitted:
                result.trace.append({"stage": "suite_admission", "reasons": reasons})
                continue

            # Compile packet
            packet = self._compile_packet(suite, ebir)
            if packet is not None:
                # Packet-level admission
                pkt_admitted, pkt_reasons = self._gate.check_packet(packet)
                if pkt_admitted:
                    if self._construction_target:
                        self.score_packet(packet, self._construction_target)
                    packets.append(packet)
                else:
                    result.trace.append({"stage": "packet_admission", "reasons": pkt_reasons})

        result.admissible = len(packets)
        if not packets:
            result.rejection_reason = "no admissible packets after verification"
            return result

        result.packets = packets

        # Stage 3: Diagnostic selection
        # First-valid = first admissible packet (score-independent)
        result.shadow_packet = packets[0]

        # Challenge = first Diagnostic-Hit, or first admissible as fallback
        diagnostic_hit_packet = None
        for pkt in packets:
            if pkt.selection_label == SelectionLabel.DIAGNOSTIC_HIT.value:
                diagnostic_hit_packet = pkt
                break

        result.selected_packet = diagnostic_hit_packet or packets[0]
        result.diagnostic_hit = diagnostic_hit_packet is not None

        return result

    def _select_operators(self, ebir: EBIR, strategy: str) -> list[str]:
        """Select operators to try, prioritizing revision-type operators."""
        available = ebir.supported_operators or list(OPERATOR_REGISTRY.keys())
        # Prioritize Replace and Withdraw (task revisions) over Introduce
        revisions = [op for op in available if is_task_revision(op)]
        introduces = [op for op in available if not is_task_revision(op)]
        if strategy == "coverage":
            return revisions + introduces
        return available

    def _compile_packet(
        self, suite: EvolvedTestSuite, ebir: EBIR,
    ) -> TerminalPacket | None:
        """Compile a TerminalPacket from an evolved test suite."""
        if not suite.tests:
            return None

        final_state = suite.tests[-1].state_after
        revision_types = [get_revision_type(t.operator) for t in suite.tests]
        task_revisions = [rt for rt in revision_types
                          if rt in (RevisionType.REPLACE.value, RevisionType.WITHDRAW.value)]
        num_revisions = len(task_revisions)
        revision_depth = RevisionDepth.MULTI.value if num_revisions >= 2 else RevisionDepth.SINGLE.value
        terminal_revision_type = task_revisions[-1] if task_revisions else RevisionType.INTRODUCE.value

        # Recompute accepted answers and verifier BEFORE building paths
        # (recompute_endpoint may modify final_state.verifier_spec in place)
        accepted_answers, verifier_spec = self._adapter.recompute_endpoint(final_state)

        # Build three paths (uses final_state.semantic_dict() which now has updated verifier_spec)
        paths = self._build_three_paths(suite, final_state, revision_types)

        # Extract stale-answer witness
        stale_witness = self._extract_stale_witness(suite)

        # Determine selection label (will be updated with construction scores)
        selection_label = SelectionLabel.COVERAGE_FALLBACK.value

        # Generate packet ID
        packet_id = hashlib.sha256(
            f"{suite.task_id}:{final_state.fingerprint()}:{[t.user_utterance for t in suite.tests]}".encode()
        ).hexdigest()[:16]

        return TerminalPacket(
            packet_id=packet_id, instance_id=suite.task_id,
            family=self._adapter.family, endpoint_state=final_state,
            paths=paths, accepted_answers=accepted_answers,
            verifier_spec=verifier_spec,
            revision_depth=revision_depth,
            terminal_revision_type=terminal_revision_type,
            num_revisions=num_revisions,
            selection_label=selection_label,
            stale_answer_witness=stale_witness,
            metadata={"source": suite.source, "discovery_model": suite.discovery_model,
                       "admitted": suite.admitted},
        )

    def _build_three_paths(
        self, suite: EvolvedTestSuite, final_state: DialogueState,
        revision_types: list[str],
    ) -> dict[str, PresentationPath]:
        """Build Direct, Introduce-Only, and Revision-Containing paths."""
        # 1. Direct: single-turn presentation of final state
        direct_prompt = ""
        if suite.canonical_probe and suite.canonical_probe.prompt:
            direct_prompt = suite.canonical_probe.prompt
        else:
            direct_prompt = self._adapter.render_state_as_prompt(final_state, suite.ebir)

        direct_path = PresentationPath(
            path_type="direct",
            dialogue=[{"role": "user", "content": direct_prompt}],
            endpoint_state=final_state.semantic_dict(),
            num_turns=1, num_external_calls=0,
            has_revision=False, revision_types=[],
        )

        # 2. Revision-Containing: actual history with revisions
        rev_dialogue: list[dict] = []
        if suite.ebir.seed_dialogue:
            rev_dialogue.extend(dict(m) for m in suite.ebir.seed_dialogue)
        else:
            rev_dialogue.append({"role": "user", "content": suite.ebir.input})

        for test in suite.tests:
            rev_dialogue.append({"role": "assistant", "content": "[acknowledgment]"})
            utterance = test.user_utterance or self._adapter.render_user_utterance(test)
            rev_dialogue.append({"role": "user", "content": utterance})

        has_revisions = any(rt in (RevisionType.REPLACE.value, RevisionType.WITHDRAW.value)
                           for rt in revision_types)

        revision_path = PresentationPath(
            path_type="revision_containing", dialogue=rev_dialogue,
            endpoint_state=final_state.semantic_dict(),
            num_turns=sum(m["role"] == "user" for m in rev_dialogue), num_external_calls=0,
            has_revision=has_revisions, revision_types=revision_types,
        )

        # 3. Introduce-Only: reach same endpoint using only Introduce events
        intro_dialogue: list[dict] = []
        if all(rt == RevisionType.INTRODUCE.value for rt in revision_types):
            # No revisions — introduce-only == revision-containing
            if suite.ebir.seed_dialogue:
                intro_dialogue.extend(dict(m) for m in suite.ebir.seed_dialogue)
            else:
                intro_dialogue.append({"role": "user", "content": suite.ebir.input})
            for test in suite.tests:
                intro_dialogue.append({"role": "assistant", "content": "[acknowledgment]"})
                utterance = test.user_utterance or self._adapter.render_user_utterance(test)
                intro_dialogue.append({"role": "user", "content": utterance})
        else:
            # Reveal terminally active information across the same user turns.
            # A semantically reviewed rendering is still required for publication.
            user_turns = sum(m["role"] == "user" for m in rev_dialogue)
            words = direct_prompt.split()
            if len(words) < user_turns:
                return {}
            for i in range(user_turns):
                start = len(words) * i // user_turns
                end = len(words) * (i + 1) // user_turns
                if i:
                    intro_dialogue.append({"role": "assistant", "content": "[acknowledgment]"})
                intro_dialogue.append({"role": "user", "content": " ".join(words[start:end])})

        actual_turns = sum(1 for m in intro_dialogue if m["role"] == "user")

        introduce_path = PresentationPath(
            path_type="introduce_only", dialogue=intro_dialogue,
            endpoint_state=final_state.semantic_dict(),
            num_turns=actual_turns, num_external_calls=0,
            has_revision=False, revision_types=[RevisionType.INTRODUCE.value] * actual_turns,
        )

        return {
            "direct": direct_path,
            "introduce_only": introduce_path,
            "revision_containing": revision_path,
        }

    def _extract_stale_witness(self, suite: EvolvedTestSuite) -> str:
        """
        Extract the stale-answer witness: answer valid before revision, now rejected.
        Only return if it differs from the post-revision accepted answers.
        """
        if len(suite.tests) < 1:
            return ""
        # The answer from the state before the final revision
        pre_revision_state = suite.tests[-1].state_before
        stale_answers, _ = self._adapter.recompute_endpoint(pre_revision_state)
        if not stale_answers:
            return ""
        # Check if the stale answer differs from the post-revision accepted answers
        final_state = suite.tests[-1].state_after
        final_answers, _ = self._adapter.recompute_endpoint(final_state)
        if stale_answers[0] in final_answers:
            return ""  # No stale answer to test (answer didn't change)
        return stale_answers[0]

    def score_packet(
        self, packet: TerminalPacket, construction_model: str = "",
    ) -> dict[str, float]:
        """
        Score a packet using the construction target model.
        Paper Equation 7: s_{p,j} = V_j(ot_{p,j}^tgt) in {0,1}
        """
        scores = {"d_score": 0.0, "i_score": 0.0, "r_score": 0.0}
        if not construction_model or not self._llm:
            return scores

        for ptype, pkey in [("direct", "d_score"), ("introduce_only", "i_score"),
                           ("revision_containing", "r_score")]:
            path = packet.paths.get(ptype)
            if not path or not path.dialogue:
                continue
            try:
                # Run construction target on this path
                response = self._llm.chat(
                    [m for m in path.dialogue if m["role"] in ("user", "assistant")],
                    model=construction_model,
                )
                # Verify with the packet's verifier
                result = self._adapter.verify_response(response, packet.endpoint_state)
                scores[pkey] = 1.0 if result.passed else 0.0
            except Exception:
                pass

        # Update selection label based on scores (paper Eq. 7-8)
        d, i, r = scores["d_score"], scores["i_score"], scores["r_score"]
        if d >= self._alpha and i >= self._alpha and (i - r) >= self._gamma:
            packet.selection_label = SelectionLabel.DIAGNOSTIC_HIT.value
        else:
            packet.selection_label = SelectionLabel.COVERAGE_FALLBACK.value

        packet.construction_scores = scores
        return scores
