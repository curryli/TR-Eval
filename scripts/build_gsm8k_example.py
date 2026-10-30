"""Reproduce a numeric task revision from a credited GSM8K source row."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from taskrevisioneval.adapters.math import MathAdapter
from taskrevisioneval.core.ebir import (
    DialogueState, OutputContract, PresentationPath, StateConstraint,
    TerminalPacket,
)
from taskrevisioneval.packet_validation import validate_packet


def build_packet(source: dict) -> TerminalPacket:
    original = source["question"]
    count_match = re.search(r"lay (\d+) eggs per day", original)
    if count_match is None or int(count_match.group(1)) != 16 or source["answer"] != "18":
        raise ValueError("This worked recipe expects GSM8K test row 0")

    adapter = MathAdapter("gsm8k")
    seed = adapter.parse_seed({"question": original, "answer": f"#### {source['answer']}"})
    initial_state = adapter.lift_state(seed).initial_state
    stale_answer = seed["reference_answer"]
    if initial_state.verifier_spec.get("expected") != stale_answer:
        raise ValueError("The source answer did not survive adapter normalization")

    old_count = int(count_match.group(1))
    new_count = old_count + 2
    revised = original.replace(f"{old_count} eggs per day", f"{new_count} eggs per day", 1)
    # The credited source row fixes 3 eggs eaten, 4 baked, and $2 per egg.
    answer = str((new_count - 3 - 4) * 2)
    state = DialogueState(
        goal="Calculate Janet's daily revenue from selling the remaining eggs.",
        facts=[revised],
        constraints=[StateConstraint.condition("answer_value", {"expected": answer})],
        verifier_spec={"type": "numeric_match", "expected": answer, "tolerance": 0.001},
        output_contract=OutputContract(format="short_answer"),
    )

    def path(kind: str, dialogue: list[dict], revised_path: bool) -> PresentationPath:
        return PresentationPath(
            path_type=kind, dialogue=dialogue, endpoint_state=state.semantic_dict(),
            num_turns=sum(m["role"] == "user" for m in dialogue),
            has_revision=revised_path,
            revision_types=["replace"] if revised_path else [],
        )

    packet = TerminalPacket(
        packet_id="gsm8k-test-0000-replace-eggs-16-to-18",
        instance_id="gsm8k-test-0000", family="math_reasoning",
        endpoint_state=state, accepted_answers=[answer],
        verifier_spec=state.verifier_spec, stale_answer_witness=stale_answer,
        num_revisions=1,
        metadata={
            "source_dataset": source["source_dataset"],
            "source_url": source["source_url"],
            "source_commit": source["source_commit"],
            "source_split": source["split"],
            "source_row_index": source["row_index"],
            "construction": f"replace daily egg count {old_count} with {new_count}; recompute revenue",
            "semantic_review": "worked example reviewed for endpoint equivalence",
        },
        paths={
            "direct": path("direct", [{"role": "user", "content": revised}], False),
            "introduce_only": path("introduce_only", [
                {"role": "user", "content": "I have a daily egg sales word problem. Please calculate the revenue."},
                {"role": "assistant", "content": "Understood."},
                {"role": "user", "content": revised},
            ], False),
            "revision_containing": path("revision_containing", [
                {"role": "user", "content": original},
                {"role": "assistant", "content": "Understood."},
                {"role": "user", "content": f"Correction: Janet's ducks lay {new_count} eggs per day, not {old_count}. Keep all other details the same. How much in dollars does she make every day at the farmers' market?"},
            ], True),
        },
    )
    errors = validate_packet(packet)
    if errors:
        raise ValueError("Invalid constructed packet: " + "; ".join(errors))
    return packet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/examples/gsm8k_source.json"))
    parser.add_argument("--output", type=Path, default=Path("data/examples/gsm8k_constructed_packet.json"))
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    packet = build_packet(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps([packet.to_dict()], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
