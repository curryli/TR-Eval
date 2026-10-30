# Data Provenance and Construction Records

The paper's 12 public benchmark sources and their official project links are listed in the [README](README.md). To reproduce a construction run, retrieve the original items from the corresponding upstream project, pin the dataset version and split, and record each item ID before applying this project's adapters, revision components, and verifiers. Each upstream project's license and access terms continue to apply.

## Provenance of the GSM8K example

- **Original item:** item 0 of the official [GSM8K test split](https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/test.jsonl).
- **Pinned upstream commit:** `3101c7d5072418e28b9008a6636bde82a006892c`.
- **Bundled source record:** [gsm8k_source.json](data/examples/gsm8k_source.json).
- **Transformation rule:** [build_gsm8k_example.py](scripts/build_gsm8k_example.py) replaces the daily egg count from 16 to 18 and recomputes the answer from the unchanged conditions.
- **Constructed result:** [gsm8k_constructed_packet.json](data/examples/gsm8k_constructed_packet.json) records the source, endpoint state, three presentation paths, accepted answer, stale-answer witness, and verifier.
- **License and attribution:** [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The separate `example_packets.json` file contains two hand-authored packets for learning the packet format.

## Extending the workflow to other public sources

Keep the following information with each constructed packet so the path from upstream item to final evaluation task can be inspected and rerun:

1. Source project, split, immutable version or commit, item ID, license, and original answer.
2. Sampling and deduplication rules, random seeds, candidate revisions, and rejection reasons.
3. The basis for recomputing the final answer, verifier configuration, and stale-answer witness.
4. Human semantic review of the three paths and the final packet hash.
5. Construction and evaluation model IDs, request settings, and scripts used to produce reported tables.

Evaluation reports may contain raw model responses. Review their contents before sharing them.
