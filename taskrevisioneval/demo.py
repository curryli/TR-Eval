"""Hand-authored example packets demonstrating the three-path structure.

These packets illustrate how a public benchmark seed (e.g., GSM8K, MMLU-Pro)
is transformed into a revision-sensitive TerminalPacket after construction.
Use `python -m taskrevisioneval demo` to export them as JSON.
"""

from __future__ import annotations

from .core.ebir import (
    DialogueState, OutputContract, PresentationPath, StateConstraint,
    TerminalPacket,
)


def _path(kind: str, dialogue: list[dict], state: DialogueState,
          has_revision: bool) -> PresentationPath:
    return PresentationPath(
        path_type=kind, dialogue=dialogue,
        endpoint_state=state.semantic_dict(),
        num_turns=sum(m["role"] == "user" for m in dialogue),
        has_revision=has_revision,
        revision_types=["replace"] if has_revision else [],
    )


def _messages(first: str, last: str) -> list[dict]:
    return [
        {"role": "user", "content": first},
        {"role": "assistant", "content": "Understood."},
        {"role": "user", "content": last},
    ]


def build_demo_packets() -> list[TerminalPacket]:
    math_prompt = "Janet starts with 7 apples and buys 3 more. How many apples does she have?"
    math_state = DialogueState(
        goal="Give the final number of apples.",
        facts=[math_prompt],
        constraints=[StateConstraint.condition("answer_value", {"expected": "10"})],
        verifier_spec={"type": "numeric_match", "expected": "10", "tolerance": 0.001},
        output_contract=OutputContract(format="short_answer"),
    )
    math_packet = TerminalPacket(
        packet_id="demo-math-001", instance_id="demo-math-001",
        family="math_reasoning", endpoint_state=math_state,
        accepted_answers=["10"], verifier_spec=math_state.verifier_spec,
        stale_answer_witness="8", num_revisions=1,
        metadata={"source": "hand_authored_demo", "semantic_review": "hand_checked"},
        paths={
            "direct": _path("direct", [{"role": "user", "content": math_prompt}], math_state, False),
            "introduce_only": _path("introduce_only", _messages(
                "Janet starts with 7 apples.",
                "She buys 3 more. How many apples does she have?"), math_state, False),
            "revision_containing": _path("revision_containing", _messages(
                "Janet starts with 5 apples and buys 3 more. How many apples does she have?",
                "Correction: Janet started with 7 apples, not 5. How many does she have after buying 3?"),
                math_state, True),
        },
    )

    options = ["Earth", "Jupiter", "Mars", "Saturn"]
    final_question = "Which is the largest planet?"
    final_qa = final_question + " Options: A. Earth B. Jupiter C. Mars D. Saturn. Reply with the letter only."
    qa_state = DialogueState(
        goal="Answer the multiple-choice question with one letter.",
        facts=[final_question],
        constraints=[StateConstraint.condition("correct_option", {"expected": "B", "options": options})],
        verifier_spec={"type": "mcq_match", "expected": "B", "options": options},
        output_contract=OutputContract(format="short_answer"),
    )
    qa_packet = TerminalPacket(
        packet_id="demo-qa-001", instance_id="demo-qa-001",
        family="structured_qa", endpoint_state=qa_state,
        accepted_answers=["B"], verifier_spec=qa_state.verifier_spec,
        stale_answer_witness="C", num_revisions=1,
        metadata={"source": "hand_authored_demo", "semantic_review": "hand_checked"},
        paths={
            "direct": _path("direct", [{"role": "user", "content": final_qa}], qa_state, False),
            "introduce_only": _path("introduce_only", _messages(
                "I will ask a multiple-choice astronomy question. Answer with one letter.", final_qa),
                qa_state, False),
            "revision_containing": _path("revision_containing", _messages(
                "Which planet is closest to the Sun? Options: A. Venus B. Earth C. Mercury D. Mars. Reply with the letter only.",
                "Change the question: " + final_qa), qa_state, True),
        },
    )
    return [math_packet, qa_packet]
