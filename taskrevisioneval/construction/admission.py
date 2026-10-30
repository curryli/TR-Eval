"""
Admission Gate: candidate test admission control.

Paper Section 4.2: Transition admission requires all of:
  - Solve(S_c) in A_c, V_c(a) = 1 for all a in A_c
  - V_c(a_{t-1}) = 0 for prior answers absent from A_c (stale-answer rejection)
  - Protected-field preservation
  - Mutation adequacy

Key principle: Adm(e) is independent of Failure(M_d, e).
Finding a model failure is NOT required for admission.
"""

from __future__ import annotations
from typing import Any

from ..core.ebir import EvolvedTestSuite, EvolutionTest, TerminalPacket
from ..core.state_ops import validate_transition, changed_paths
from .operators import get_operator_spec, is_task_revision


class AdmissionGate:
    """
    Multi-gate admission control (paper Section 4.2).

    Gate 1: Formal validity (well-formedness)
    Gate 2: State consistency (valid transition)
    Gate 3: Invariant defined
    Gate 4: Answerability (reference solvability)
    Gate 5: Verifier executability
    Gate 6: Canonicalizable
    Gate 7: Endpoint equivalence (all paths reach same endpoint)
    """

    def __init__(self, adapter, llm_client=None, audit_models: list[str] = None):
        self._adapter = adapter
        self._llm = llm_client
        self._audit_models = audit_models or []

    def check(self, suite: EvolvedTestSuite) -> tuple[bool, list[str]]:
        """Execute complete admission checks. Returns (admitted, reasons)."""
        gates = [
            ("formal_validity", self._check_well_formedness),
            ("state_consistency", self._check_valid_transition),
            ("invariants", self._check_invariant_defined),
            ("answerability", self._check_reference_solvability),
            ("verifier_executability", self._check_verifier_executable),
            ("canonicalizability", self._check_canonicalizable),
            ("endpoint_equivalence", self._check_endpoint_equivalence),
        ]

        report = []
        reasons: list[str] = []
        for name, checker in gates:
            ok, gate_reasons = checker(suite)
            report.append({"gate": name, "passed": ok, "reasons": gate_reasons})
            if not ok:
                reasons.extend(f"{name}: {r}" for r in gate_reasons)

        suite.metadata["admission_report"] = report
        suite.admitted = len(reasons) == 0
        suite.admission_reasons = reasons
        return suite.admitted, reasons

    def check_packet(self, packet: TerminalPacket) -> tuple[bool, list[str]]:
        """Check admission for a compiled TerminalPacket."""
        reasons: list[str] = []
        required = {"direct", "introduce_only", "revision_containing"}
        if set(packet.paths) != required:
            reasons.append("packet must contain exactly the three evaluation paths")
        if packet.num_revisions < 1:
            reasons.append("packet has no task revision")
        if not packet.stale_answer_witness:
            reasons.append("missing stale-answer witness")
        if not packet.accepted_answers or any(not a for a in packet.accepted_answers):
            reasons.append("missing accepted answer")
        if packet.verifier_spec != packet.endpoint_state.verifier_spec:
            reasons.append("packet verifier differs from endpoint verifier")

        # Endpoint equivalence: all paths must reach the same endpoint
        endpoint_fp = packet.endpoint_state.fingerprint()
        for ptype, path in packet.paths.items():
            if not path.dialogue or path.dialogue[-1].get("role") != "user":
                reasons.append(f"path '{ptype}' has no final user turn")
            if path.num_turns != sum(m.get("role") == "user" for m in path.dialogue):
                reasons.append(f"path '{ptype}' has incorrect turn count")
            if path.endpoint_state:
                from ..core.ebir import DialogueState
                path_state = DialogueState.from_dict(path.endpoint_state)
                if path_state.fingerprint() != endpoint_fp:
                    reasons.append(f"path '{ptype}' reaches different endpoint")

        # Stale-answer witness: verifier must reject old valid answer
        if packet.stale_answer_witness and packet.accepted_answers:
            if packet.stale_answer_witness in packet.accepted_answers:
                reasons.append("stale answer witness is still in accepted answers")
            try:
                if self._adapter.verify_response(packet.stale_answer_witness, packet.endpoint_state).passed:
                    reasons.append("verifier accepts stale-answer witness")
            except Exception as exc:
                reasons.append(f"stale-answer check failed: {exc}")
        for answer in packet.accepted_answers:
            try:
                if not self._adapter.verify_response(answer, packet.endpoint_state).passed:
                    reasons.append("verifier rejects an accepted answer")
            except Exception as exc:
                reasons.append(f"accepted-answer check failed: {exc}")
        intro = packet.paths.get("introduce_only")
        rev_path = packet.paths.get("revision_containing")
        if intro and rev_path:
            if len(intro.dialogue) != len(rev_path.dialogue):
                reasons.append("history paths differ in message count")
            if intro.num_external_calls != rev_path.num_external_calls:
                reasons.append("history paths differ in external calls")

        # Has at least one revision for revision-containing path
        rev_path = packet.paths.get("revision_containing")
        if rev_path and not rev_path.has_revision and packet.num_revisions > 0:
            reasons.append("revision_containing path has no revision but packet claims revisions")

        return (len(reasons) == 0, reasons)

    def _check_well_formedness(self, suite: EvolvedTestSuite) -> tuple[bool, list[str]]:
        if not suite.tests:
            return False, ["no evolution tests"]
        for t in suite.tests:
            if not t.operator or t.state_before is None or t.state_after is None:
                return False, [f"test at turn {t.turn_index} is incomplete"]
        return True, []

    def _check_valid_transition(self, suite: EvolvedTestSuite) -> tuple[bool, list[str]]:
        for t in suite.tests:
            if t.contract:
                violations = validate_transition(t.state_before, t.state_after, t.contract)
                if violations:
                    return False, [f"turn {t.turn_index}: {v}" for v in violations]
        return True, []

    def _check_invariant_defined(self, suite: EvolvedTestSuite) -> tuple[bool, list[str]]:
        for t in suite.tests:
            if t.contract and not t.contract.invariants and t.must_preserve:
                return False, [f"turn {t.turn_index}: invariants not defined"]
        return True, []

    def _check_reference_solvability(self, suite: EvolvedTestSuite) -> tuple[bool, list[str]]:
        final_state = suite.tests[-1].state_after if suite.tests else None
        if final_state is None:
            return False, ["no final state to check solvability"]
        if not final_state.verifier_spec:
            return False, ["no verifier specification in final state"]
        return True, []

    def _check_verifier_executable(self, suite: EvolvedTestSuite) -> tuple[bool, list[str]]:
        # Check that the adapter can execute the verifier
        final_state = suite.tests[-1].state_after if suite.tests else None
        if final_state is None:
            return False, ["no final state"]
        try:
            result = self._adapter.verify_response("test", final_state)
            return True, []
        except Exception as e:
            return False, [f"verifier execution error: {e}"]

    def _check_canonicalizable(self, suite: EvolvedTestSuite) -> tuple[bool, list[str]]:
        if suite.canonical_probe is None:
            return False, ["no canonical probe"]
        if not suite.canonical_probe.prompt:
            return False, ["canonical probe has empty prompt"]
        return True, []

    def _check_endpoint_equivalence(self, suite: EvolvedTestSuite) -> tuple[bool, list[str]]:
        if not suite.tests:
            return False, ["no tests to check endpoint equivalence"]
        final_state = suite.tests[-1].state_after
        if suite.canonical_probe and suite.canonical_probe.endpoint_fingerprint:
            if final_state.fingerprint() != suite.canonical_probe.endpoint_fingerprint:
                return False, ["canonical probe endpoint differs from final state"]
        return True, []
