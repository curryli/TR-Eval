"""
EBIR: Evolvable Benchmark Intermediate Representation.

Core data structures for TaskRevisionEval, aligned with paper Section 3.

Key abstractions:
  - DialogueState S_t = (G, F, E, C, O, Omega, R, V, H)
  - TransitionContract T_Delta = (P, M, K, D, Phi, Gamma, I)
  - TerminalPacket u = (S_u, P_u, A_u, V_u)
  - PresentationPath: Direct / Introduce-Only / Revision-Containing
  - PathOutcome: (D, I, R) binary outcomes
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum
import hashlib
import json


# ============================================================
# Enums (Paper Section 3.1)
# ============================================================

class RevisionType(str, Enum):
    """Three semantic types of task-state change (paper Section 3.1)."""
    INTRODUCE = "introduce"   # New value/requirement becomes active; nothing invalidated
    REPLACE = "replace"       # New value replaces an active one; old value no longer active
    WITHDRAW = "withdraw"     # Active requirement revoked without replacement


class RevisionDepth(str, Enum):
    """Stratum assignment by number of task revisions."""
    SINGLE = "single"   # Exactly one Replace or Withdraw
    MULTI = "multi"      # Two or more Replace/Withdraw events


class SelectionLabel(str, Enum):
    """Construction-side selection label (paper Section 4.3)."""
    DIAGNOSTIC_HIT = "diagnostic_hit"
    COVERAGE_FALLBACK = "coverage_fallback"


class LifecycleLabel(str, Enum):
    """
    State slot lifecycle labels (paper Section 3.1).
    History that exists != state that is currently valid.
    """
    ACTIVE = "active"        # Currently valid, affects the answer
    SUPERSEDED = "superseded"  # Replaced by a new value; old value must not affect answer
    RETRACTED = "retracted"    # Explicitly revoked by user; no longer in effect
    PRESERVED = "preserved"    # Must remain unchanged across transitions (invariant)


# ============================================================
# Core State Structures
# ============================================================

@dataclass
class OutputContract:
    """Defines the expected output format and structure."""
    format: str = "free_text"  # free_text, json, markdown, short_answer, code, etc.
    required_fields: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"format": self.format, "required_fields": self.required_fields, "constraints": self.constraints}

    @classmethod
    def from_dict(cls, d: dict) -> "OutputContract":
        return cls(format=d.get("format", "free_text"), required_fields=d.get("required_fields", []),
                   constraints=d.get("constraints", {}))


@dataclass
class StateConstraint:
    """
    Generic state constraint supporting multiple types:
      - instruction: IFEval-style (instruction_id + kwargs)
      - condition: math/logic (condition_str + expected_value)
      - schema: structured output (schema_spec)
      - format: format constraint (format_type + params)
      - custom: adapter-defined (type + params)
    """
    type: str = "instruction"
    key: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    instruction_id: str = ""
    kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"type": self.type, "key": self.key, "params": self.params,
                "instruction_id": self.instruction_id, "kwargs": self.kwargs}

    @classmethod
    def from_dict(cls, d: dict) -> "StateConstraint":
        return cls(type=d.get("type", "instruction"), key=d.get("key", ""),
                   params=d.get("params", {}), instruction_id=d.get("instruction_id", ""),
                   kwargs=d.get("kwargs", {}))

    @classmethod
    def instruction(cls, instruction_id: str, kwargs: dict = None) -> "StateConstraint":
        return cls(type="instruction", key=instruction_id,
                   instruction_id=instruction_id, kwargs=kwargs or {})

    @classmethod
    def condition(cls, key: str, params: dict = None) -> "StateConstraint":
        return cls(type="condition", key=key, params=params or {})

    @classmethod
    def schema(cls, key: str, params: dict = None) -> "StateConstraint":
        return cls(type="schema", key=key, params=params or {})


# Backward-compatible alias
InstructionConstraint = StateConstraint


@dataclass
class LifecycleEntry:
    """Lifecycle record for a single state slot."""
    slot: str
    label: LifecycleLabel = LifecycleLabel.ACTIVE
    value: Any = None
    superseded_by: str = ""
    history_id: str = ""

    def to_dict(self) -> dict:
        return {"slot": self.slot, "label": self.label.value, "value": self.value,
                "superseded_by": self.superseded_by, "history_id": self.history_id}

    @classmethod
    def from_dict(cls, d: dict) -> "LifecycleEntry":
        try:
            label = LifecycleLabel(d.get("label", "active"))
        except ValueError:
            label = LifecycleLabel.ACTIVE
        return cls(slot=d.get("slot", ""), label=label, value=d.get("value"),
                    superseded_by=d.get("superseded_by", ""), history_id=d.get("history_id", ""))


@dataclass
class DialogueState:
    """
    Executable task state S_t = (G, F, E, C, O, Omega, R, V, H).

    Paper Section 3: A task state records the active request and its executable checks.
      G: goal           -- task goal
      F: facts          -- facts and premises
      E: entities       -- entities and relations
      C: constraints    -- active constraints
      O: output_contract -- output format requirements
      Omega: environment -- tool params, API schema, execution context
      R: reference      -- reference answer / solver
      V: verifier_spec  -- verifier specification
      H: lifecycle       -- provenance and lifecycle labels
    """
    goal: str = ""
    facts: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    constraints: list[StateConstraint] = field(default_factory=list)
    output_contract: OutputContract = field(default_factory=OutputContract)
    verifier_spec: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    reference: dict[str, Any] = field(default_factory=dict)
    lifecycle: list[LifecycleEntry] = field(default_factory=list)
    superseded: list[dict] = field(default_factory=list)
    model_commitments: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "goal": self.goal, "facts": self.facts, "entities": self.entities,
            "constraints": [c.to_dict() for c in self.constraints],
            "output_contract": self.output_contract.to_dict(),
            "verifier_spec": self.verifier_spec, "environment": self.environment,
            "reference": self.reference,
            "lifecycle": [e.to_dict() for e in self.lifecycle],
            "superseded": self.superseded, "model_commitments": self.model_commitments,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DialogueState":
        return cls(
            goal=d.get("goal", ""), facts=d.get("facts", []), entities=d.get("entities", []),
            constraints=[StateConstraint.from_dict(c) for c in d.get("constraints", [])],
            output_contract=OutputContract.from_dict(d.get("output_contract", {})),
            verifier_spec=d.get("verifier_spec", {}), environment=d.get("environment", {}),
            reference=d.get("reference", {}),
            lifecycle=[LifecycleEntry.from_dict(e) for e in d.get("lifecycle", [])],
            superseded=d.get("superseded", []), model_commitments=d.get("model_commitments", []),
        )

    def clone(self) -> "DialogueState":
        return DialogueState.from_dict(self.to_dict())

    def semantic_dict(self) -> dict[str, Any]:
        """Return the operational endpoint, excluding provenance-only history."""
        active_lifecycle = [e.to_dict() for e in self.lifecycle
                            if e.label in (LifecycleLabel.ACTIVE, LifecycleLabel.PRESERVED)]
        return {
            "goal": self.goal, "facts": self.facts, "entities": self.entities,
            "constraints": [c.to_dict() for c in self.active_constraints()],
            "output_contract": self.output_contract.to_dict(),
            "verifier_spec": self.verifier_spec, "environment": self.environment,
            "reference": self.reference, "lifecycle": active_lifecycle,
        }

    def fingerprint(self) -> str:
        """Stable digest for endpoint comparison (paper Equation 2)."""
        payload = json.dumps(self.semantic_dict(), ensure_ascii=False,
                              sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def active_constraints(self) -> list[StateConstraint]:
        """Return currently active (ACTIVE or PRESERVED) constraint subset."""
        active_slots = {e.slot for e in self.lifecycle
                        if e.label in (LifecycleLabel.ACTIVE, LifecycleLabel.PRESERVED)}
        if not self.lifecycle:
            return list(self.constraints)
        return [c for c in self.constraints if c.key in active_slots or not c.key]

    def mark_superseded(self, slot: str, new_slot: str = "") -> None:
        for entry in self.lifecycle:
            if entry.slot == slot and entry.label == LifecycleLabel.ACTIVE:
                entry.label = LifecycleLabel.SUPERSEDED
                entry.superseded_by = new_slot
        self.superseded.append({"slot": slot, "superseded_by": new_slot})

    def mark_retracted(self, slot: str) -> None:
        for entry in self.lifecycle:
            if entry.slot == slot and entry.label in (LifecycleLabel.ACTIVE, LifecycleLabel.PRESERVED):
                entry.label = LifecycleLabel.RETRACTED


# ============================================================
# EBIR: Unified Intermediate Representation
# ============================================================

@dataclass
class EBIR:
    """Evolvable Benchmark Intermediate Representation: Lift(s) for all adapters."""
    task_id: str
    domain: str
    input: str
    reference: dict[str, Any] = field(default_factory=dict)
    initial_state: DialogueState = field(default_factory=DialogueState)
    mutable_slots: list[str] = field(default_factory=list)
    preserved_slots: list[str] = field(default_factory=list)
    supported_operators: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    seed_dialogue: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "domain": self.domain, "input": self.input,
            "reference": self.reference, "initial_state": self.initial_state.to_dict(),
            "mutable_slots": self.mutable_slots, "preserved_slots": self.preserved_slots,
            "supported_operators": self.supported_operators, "metadata": self.metadata,
            "seed_dialogue": self.seed_dialogue,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EBIR":
        return cls(
            task_id=d["task_id"], domain=d["domain"], input=d["input"],
            reference=d.get("reference", {}),
            initial_state=DialogueState.from_dict(d.get("initial_state", {})),
            mutable_slots=d.get("mutable_slots", []),
            preserved_slots=d.get("preserved_slots", []),
            supported_operators=d.get("supported_operators", []),
            metadata=d.get("metadata", {}), seed_dialogue=d.get("seed_dialogue", []),
        )


# ============================================================
# Transition Contract (Paper Section 3.1)
# ============================================================

@dataclass
class TransitionContract:
    """
    Typed state-transition contract T_Delta = (P, M, K, D, Phi, Gamma, I).

    Paper Section 3.1: The transition contract is the key technical object that
    converts natural-language follow-ups into executable, auditable state operations.

      P: precondition         -- conditions for the update to be valid
      M: mutable_slots        -- state slots allowed to change
      K: preserved_slots      -- slots that must remain unchanged
      D: deprecated_slots     -- old information that must become invalid after update
      Phi: transition_fn      -- declarative state transition function
      Gamma: oracle_transport  -- oracle/verifier synchronization rule
      I: invariants           -- invariants that must hold before and after

    Validity constraint: M ∩ K = empty, D ∩ K = empty
    """
    operator: str = ""
    mutable_slots: list[str] = field(default_factory=list)
    preserved_slots: list[str] = field(default_factory=list)
    deprecated_slots: list[str] = field(default_factory=list)
    precondition: str = ""
    transition_fn: str = ""
    oracle_transport: str = ""
    invariants: list[str] = field(default_factory=list)
    user_utterance: str = ""
    verifier_spec: dict[str, Any] = field(default_factory=dict)
    updates: dict[str, Any] = field(default_factory=dict)
    recovered_contract: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check structural and set-theoretic validity."""
        m, k, d, u = (set(self.mutable_slots), set(self.preserved_slots),
                      set(self.deprecated_slots), set(self.updates))
        # Check that each update key's top-level prefix is in mutable_slots
        update_prefixes_ok = all(
            any(key == slot or key.startswith(slot + ".") for slot in m)
            for key in u
        ) if u and m else True
        return (bool(self.operator) and bool(m or k)
                and not (m & k) and not (d & k)
                and (not u or update_prefixes_ok))

    def to_dict(self) -> dict:
        return {
            "operator": self.operator, "mutable_slots": self.mutable_slots,
            "preserved_slots": self.preserved_slots, "deprecated_slots": self.deprecated_slots,
            "precondition": self.precondition, "transition_fn": self.transition_fn,
            "oracle_transport": self.oracle_transport, "invariants": self.invariants,
            "user_utterance": self.user_utterance, "verifier_spec": self.verifier_spec,
            "updates": self.updates, "recovered_contract": self.recovered_contract,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TransitionContract":
        def _norm(slot: str) -> str:
            return slot[len("state."):] if slot.startswith("state.") else slot
        raw_updates = d.get("updates", {})
        return cls(
            operator=d.get("operator", ""),
            mutable_slots=[_norm(s) for s in d.get("mutable_slots", [])],
            preserved_slots=[_norm(s) for s in d.get("preserved_slots", [])],
            deprecated_slots=[_norm(s) for s in d.get("deprecated_slots", [])],
            precondition=d.get("precondition", ""), transition_fn=d.get("transition_fn", ""),
            oracle_transport=d.get("oracle_transport", ""), invariants=d.get("invariants", []),
            user_utterance=d.get("user_utterance", ""), verifier_spec=d.get("verifier_spec", {}),
            updates={_norm(k): v for k, v in raw_updates.items()},
            recovered_contract=d.get("recovered_contract", {}), metadata=d.get("metadata", {}),
        )


# ============================================================
# Evolution Test and Suite
# ============================================================

@dataclass
class EvolutionTest:
    """One round of evolution: e = (S_t, Delta_S, I, V)."""
    operator: str
    turn_index: int
    state_before: DialogueState
    state_after: DialogueState
    allowed_change: dict[str, Any] = field(default_factory=dict)
    must_preserve: list[str] = field(default_factory=list)
    must_not_preserve: list[str] = field(default_factory=list)
    user_utterance: str = ""
    verifier_spec: dict[str, Any] = field(default_factory=dict)
    target_weakness: str = ""
    contract: Optional[TransitionContract] = None

    def to_dict(self) -> dict:
        return {
            "operator": self.operator, "turn_index": self.turn_index,
            "state_before": self.state_before.to_dict(),
            "state_after": self.state_after.to_dict(),
            "allowed_change": self.allowed_change, "must_preserve": self.must_preserve,
            "must_not_preserve": self.must_not_preserve, "user_utterance": self.user_utterance,
            "verifier_spec": self.verifier_spec, "target_weakness": self.target_weakness,
            "contract": self.contract.to_dict() if self.contract else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvolutionTest":
        cd = d.get("contract")
        return cls(
            operator=d.get("operator", ""), turn_index=d.get("turn_index", 0),
            state_before=DialogueState.from_dict(d.get("state_before", {})),
            state_after=DialogueState.from_dict(d.get("state_after", {})),
            allowed_change=d.get("allowed_change", {}), must_preserve=d.get("must_preserve", []),
            must_not_preserve=d.get("must_not_preserve", []),
            user_utterance=d.get("user_utterance", ""),
            verifier_spec=d.get("verifier_spec", {}),
            target_weakness=d.get("target_weakness", ""),
            contract=TransitionContract.from_dict(cd) if cd else None,
        )


@dataclass
class CanonicalProbe:
    """
    Same-endpoint control probe c_1 = Canonicalize(S_1; b_0).
    Paper Section 3.2: single-turn or trajectory-preserving canonicalization.
    """
    prompt: str = ""
    expected_behavior: str = ""
    verifier_spec: dict[str, Any] = field(default_factory=dict)
    canonicalization_type: str = "single_turn"
    trajectory: list[dict] = field(default_factory=list)
    burden_profile: dict[str, Any] = field(default_factory=dict)
    endpoint_state: dict[str, Any] = field(default_factory=dict)
    endpoint_fingerprint: str = ""

    def to_dict(self) -> dict:
        return {"prompt": self.prompt, "expected_behavior": self.expected_behavior,
                "verifier_spec": self.verifier_spec,
                "canonicalization_type": self.canonicalization_type,
                "trajectory": self.trajectory, "burden_profile": self.burden_profile,
                "endpoint_state": self.endpoint_state, "endpoint_fingerprint": self.endpoint_fingerprint}

    @classmethod
    def from_dict(cls, d: dict) -> "CanonicalProbe":
        return cls(prompt=d.get("prompt", ""), expected_behavior=d.get("expected_behavior", ""),
                   verifier_spec=d.get("verifier_spec", {}),
                   canonicalization_type=d.get("canonicalization_type", "single_turn"),
                   trajectory=d.get("trajectory", []), burden_profile=d.get("burden_profile", {}),
                   endpoint_state=d.get("endpoint_state", {}),
                   endpoint_fingerprint=d.get("endpoint_fingerprint", ""))


@dataclass
class EvolvedTestSuite:
    """A complete evolution test suite: multi-turn dialogue + canonical probe."""
    task_id: str
    ebir: EBIR
    tests: list[EvolutionTest] = field(default_factory=list)
    canonical_probe: Optional[CanonicalProbe] = None
    dialogue: list[dict] = field(default_factory=list)
    source: str = "model_guided"
    discovery_model: str = ""
    admitted: bool = False
    admission_reasons: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "ebir": self.ebir.to_dict(),
            "tests": [t.to_dict() for t in self.tests],
            "canonical_probe": self.canonical_probe.to_dict() if self.canonical_probe else None,
            "dialogue": self.dialogue, "source": self.source,
            "discovery_model": self.discovery_model, "admitted": self.admitted,
            "admission_reasons": self.admission_reasons, "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvolvedTestSuite":
        return cls(
            task_id=d.get("task_id", ""), ebir=EBIR.from_dict(d.get("ebir", {})),
            tests=[EvolutionTest.from_dict(t) for t in d.get("tests", [])],
            canonical_probe=CanonicalProbe.from_dict(d["canonical_probe"]) if d.get("canonical_probe") else None,
            dialogue=d.get("dialogue", []), source=d.get("source", "model_guided"),
            discovery_model=d.get("discovery_model", ""), admitted=d.get("admitted", False),
            admission_reasons=d.get("admission_reasons", []), metadata=d.get("metadata", {}),
        )


# ============================================================
# Three-Path Packet Structures (Paper Section 3.2)
# ============================================================

@dataclass
class PresentationPath:
    """
    One of three presentation paths for a terminal packet.
    Paper Section 3.2: Direct, Introduce-Only, or Revision-Containing.
    Each path is a dialogue that reaches the same endpoint S_u.
    """
    path_type: str  # "direct", "introduce_only", "revision_containing"
    dialogue: list[dict] = field(default_factory=list)
    endpoint_state: dict[str, Any] = field(default_factory=dict)
    num_turns: int = 0
    num_external_calls: int = 0
    has_revision: bool = False
    revision_types: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"path_type": self.path_type, "dialogue": self.dialogue,
                "endpoint_state": self.endpoint_state, "num_turns": self.num_turns,
                "num_external_calls": self.num_external_calls,
                "has_revision": self.has_revision, "revision_types": self.revision_types}

    @classmethod
    def from_dict(cls, d: dict) -> "PresentationPath":
        return cls(path_type=d.get("path_type", ""), dialogue=d.get("dialogue", []),
                   endpoint_state=d.get("endpoint_state", {}), num_turns=d.get("num_turns", 0),
                   num_external_calls=d.get("num_external_calls", 0),
                   has_revision=d.get("has_revision", False),
                   revision_types=d.get("revision_types", []))


@dataclass
class TerminalPacket:
    """
    Paper Section 3.2: one retained terminal packet u = (S_u, P_u, A_u, V_u).
    Contains three presentation paths sharing the same endpoint, accepted answers,
    and executable verifier.
    """
    packet_id: str
    instance_id: str
    family: str
    endpoint_state: DialogueState = field(default_factory=DialogueState)
    paths: dict[str, PresentationPath] = field(default_factory=dict)
    accepted_answers: list[str] = field(default_factory=list)
    verifier_spec: dict[str, Any] = field(default_factory=dict)
    revision_depth: str = "single"
    terminal_revision_type: str = "replace"
    num_revisions: int = 1
    selection_label: str = "coverage_fallback"
    construction_scores: dict[str, float] = field(default_factory=dict)
    stale_answer_witness: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "packet_id": self.packet_id, "instance_id": self.instance_id, "family": self.family,
            "endpoint_state": self.endpoint_state.to_dict(),
            "paths": {k: v.to_dict() for k, v in self.paths.items()},
            "accepted_answers": self.accepted_answers, "verifier_spec": self.verifier_spec,
            "revision_depth": self.revision_depth,
            "terminal_revision_type": self.terminal_revision_type,
            "num_revisions": self.num_revisions, "selection_label": self.selection_label,
            "construction_scores": self.construction_scores,
            "stale_answer_witness": self.stale_answer_witness, "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TerminalPacket":
        return cls(
            packet_id=d.get("packet_id", ""), instance_id=d.get("instance_id", ""),
            family=d.get("family", ""),
            endpoint_state=DialogueState.from_dict(d.get("endpoint_state", {})),
            paths={k: PresentationPath.from_dict(v) for k, v in d.get("paths", {}).items()},
            accepted_answers=d.get("accepted_answers", []),
            verifier_spec=d.get("verifier_spec", {}),
            revision_depth=d.get("revision_depth", "single"),
            terminal_revision_type=d.get("terminal_revision_type", "replace"),
            num_revisions=d.get("num_revisions", 1),
            selection_label=d.get("selection_label", "coverage_fallback"),
            construction_scores=d.get("construction_scores", {}),
            stale_answer_witness=d.get("stale_answer_witness", ""),
            metadata=d.get("metadata", {}),
        )


# ============================================================
# Evaluation Outcome Structures (Paper Section 3.2)
# ============================================================

@dataclass
class PathOutcome:
    """Binary outcome for one path of one packet for one model (paper Eq. 3)."""
    direct: Optional[bool] = None           # D_u
    introduce_only: Optional[bool] = None  # I_u
    revision_containing: Optional[bool] = None  # R_u

    @property
    def is_rcf(self) -> bool:
        """Paper Eq. 3: RCF = D=1, I=1, R=0 (strict Revision-Context Failure)."""
        return self.direct is True and self.introduce_only is True and self.revision_containing is False

    @property
    def is_all_pass(self) -> bool:
        """All-path success: D=1, I=1, R=1."""
        return self.direct is True and self.introduce_only is True and self.revision_containing is True

    @property
    def diagnostic_category(self) -> str:
        """Paper Section 3.2: four-way hierarchical routing under D->I->R ordering."""
        if self.direct is None or self.direct is False:
            return "end_fail"
        if self.introduce_only is None or self.introduce_only is False:
            return "inc_fail"
        if self.revision_containing is False:
            return "rcf"
        return "all_pass"

    def to_dict(self) -> dict:
        return {"direct": self.direct, "introduce_only": self.introduce_only,
                "revision_containing": self.revision_containing,
                "is_rcf": self.is_rcf, "diagnostic_category": self.diagnostic_category}


@dataclass
class PacketEvaluationResult:
    """Complete evaluation result for one packet across all models."""
    packet_id: str
    model: str
    outcomes: PathOutcome = field(default_factory=PathOutcome)
    responses: dict[str, str] = field(default_factory=dict)
    verifier_details: dict[str, dict] = field(default_factory=dict)
    error_labels: list[str] = field(default_factory=list)
    failure_type: str = ""

    def to_dict(self) -> dict:
        return {"packet_id": self.packet_id, "model": self.model,
                "outcomes": self.outcomes.to_dict(), "responses": self.responses,
                "verifier_details": self.verifier_details, "error_labels": self.error_labels,
                "failure_type": self.failure_type}
