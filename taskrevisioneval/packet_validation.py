"""Structural and answer-verifier checks for constructed packets.

These checks verify structural endpoint fingerprints, stale-answer
rejection, and verifier executability. Human review is required for semantic
endpoint equivalence. Adapters for all six task families are included;
code-generation, tool-use, and native-multi-turn families require their
respective execution backends (sandbox, tool runtime, LLM judge) which
callers should provide via the adapter interface.
"""

from __future__ import annotations

from .adapters.math import MathAdapter
from .adapters.structured_qa import StructuredQAAdapter
from .adapters.ifeval import IFEvalAdapter
from .adapters.code import CodeAdapter
from .adapters.tool_use import ToolUseAdapter
from .adapters.native_multi_turn import NativeMultiTurnAdapter
from .construction.admission import AdmissionGate
from .core.ebir import TerminalPacket


ADAPTERS = {
    cls.family: cls for cls in (
        MathAdapter, StructuredQAAdapter, IFEvalAdapter,
        CodeAdapter, ToolUseAdapter, NativeMultiTurnAdapter,
    )
}
SUPPORTED_EVALUATION_FAMILIES = {
    "math_reasoning", "structured_qa", "instruction_following",
}


def validate_packet(packet: TerminalPacket) -> list[str]:
    """Return all detected hard defects; an empty list is a structural pass."""
    errors: list[str] = []
    if not packet.packet_id or not packet.instance_id:
        errors.append("missing packet or instance ID")
    adapter_class = ADAPTERS.get(packet.family)
    if adapter_class is None:
        return errors + [f"unsupported family: {packet.family}"]
    if packet.family not in SUPPORTED_EVALUATION_FAMILIES:
        errors.append(f"{packet.family} requires its execution backend (sandbox/tool-runtime/LLM-judge); provide it via the adapter interface")
    adapter = adapter_class()
    _, gate_errors = AdmissionGate(adapter).check_packet(packet)
    errors.extend(gate_errors)

    direct = packet.paths.get("direct")
    intro = packet.paths.get("introduce_only")
    revision = packet.paths.get("revision_containing")
    if direct and direct.num_turns != 1:
        errors.append("Direct must have one user turn")
    if intro and intro.has_revision:
        errors.append("Introduce-Only is marked as containing a revision")
    if revision and not revision.has_revision:
        errors.append("Revision-Containing is not marked as containing a revision")
    if intro and revision:
        roles_i = [m.get("role") for m in intro.dialogue]
        roles_r = [m.get("role") for m in revision.dialogue]
        if roles_i != roles_r:
            errors.append("history paths differ in response opportunities")
    if packet.family == "instruction_following":
        if not packet.endpoint_state.verifier_spec.get("instruction_id_list"):
            errors.append("IFEval endpoint has no active checks")
    return errors
