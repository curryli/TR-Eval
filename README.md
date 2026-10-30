# TaskRevisionEval: Dynamic Task Revisions from Open Benchmarks

TaskRevisionEval turns public benchmark questions into controlled tests of whether a language model follows the **currently valid task** after a user changes it. Each source item is mapped to a structured task state. A construction step introduces, replaces, or withdraws facts and constraints, then recomputes the final answer. The resulting task is presented through three paths that share one endpoint and one verifier:

- **Direct:** present the final task in one turn.
- **Introduce-Only:** reveal currently valid information incrementally.
- **Revision-Containing:** present an earlier task, then revise it.

Comparing the three outcomes helps isolate failures caused by revision history from failures on the final task itself. The repository provides [task-family adapters](taskrevisioneval/adapters), [construction components](taskrevisioneval/construction), [packet validation](taskrevisioneval/packet_validation.py), a [three-path evaluation CLI](taskrevisioneval/cli), diagnostic metrics, and a reproducible construction example.

## Start with public benchmark data

The paper uses the following upstream projects. Their original datasets are available from the linked project pages. For each construction run, record the source project, split, version or commit, item ID, and original answer.

| Task family | Public sources | Examples of revisable elements |
| --- | --- | --- |
| Instruction following | [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) | Output format, keywords, verifiable constraints |
| Math reasoning | [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), [GSM8K](https://github.com/openai/grade-school-math) | Numbers, conditions, target quantity |
| Structured QA | [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro), [TruthfulQA](https://github.com/sylinrl/TruthfulQA) | Question scope, choices, answer mapping |
| Code generation | [HumanEval](https://github.com/openai/human-eval), [MBPP](https://github.com/google-research/google-research/tree/master/mbpp), [BigCodeBench](https://github.com/bigcode-project/bigcodebench) | Interfaces, function requirements, test constraints |
| Tool use | [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard), [ToolSandbox](https://github.com/apple-aiml-research/ToolSandbox) | Tool arguments, call objectives, environment state |
| Native multi-turn | [MultiChallenge](https://huggingface.co/datasets/ScaleAI/MultiChallenge), [MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) | Accumulated requirements, later revisions, commitments |

The adapters map supported source formats to initial task states. Construction operators describe how those states change. Validation checks the new answer, the superseded-answer witness, and the three paths against a shared endpoint. See [DATA.md](DATA.md) for provenance and license records.

## Follow one question from source to constructed packet

The bundled [source record](data/examples/gsm8k_source.json) is item 0 of the official [GSM8K test split](https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/test.jsonl), pinned to a specific upstream commit. In the original question, Janet's ducks lay **16** eggs per day. She uses 3 for breakfast and 4 for baking, then sells the rest for $2 each. The original answer is `(16 - 3 - 4) × 2 = 18` dollars.

The [construction script](scripts/build_gsm8k_example.py) first normalizes the source item with `MathAdapter.parse_seed()` and `lift_state()`. It then replaces the daily egg count with **18**, keeps the other conditions, and recomputes the answer as `(18 - 3 - 4) × 2 = 22` dollars. The [constructed packet](data/examples/gsm8k_constructed_packet.json) presents that same final task in three ways:

| Path | Presentation | Final answer |
| --- | --- | ---: |
| Direct | Ask the revised 18-egg question immediately | 22 |
| Introduce-Only | Introduce the word problem, then ask the revised 18-egg question | 22 |
| Revision-Containing | Ask the original 16-egg question, then correct the count to 18 | 22 |

The original answer, **18**, becomes `stale_answer_witness`; **22** is the `accepted_answers` value for all three paths. If a model still answers 18 after the correction, the numeric verifier rejects that response. This excerpt shows the key fields; the [full JSON file](data/examples/gsm8k_constructed_packet.json) also contains the task state, dialogue messages, provenance, and verifier configuration.

```json
{
  "packet_id": "gsm8k-test-0000-replace-eggs-16-to-18",
  "accepted_answers": ["22"],
  "stale_answer_witness": "18",
  "paths": {
    "direct": {"num_turns": 1},
    "introduce_only": {"num_turns": 2},
    "revision_containing": {"num_turns": 2, "has_revision": true}
  }
}
```

Run the following commands from the repository root to regenerate and validate the example:

```bash
python -m pip install -e '.[llm]'
python -m scripts.build_gsm8k_example
taskrevisioneval validate --packets data/examples/gsm8k_constructed_packet.json
```

Two additional hand-authored packets demonstrate the same interface:

```bash
taskrevisioneval demo --output demo_packets.json
taskrevisioneval validate --packets demo_packets.json
```

## From source item to evaluation result

1. **Load a source item.** Retrieve a specific upstream release. `parse_seed()` and `lift_state()` map its question, answer, and constraints to an initial state.
2. **Apply a revision.** Choose an introduction, replacement, or withdrawal appropriate to the task family, and record the changed field and previous value. The GSM8K example applies the numeric replacement `16 → 18`.
3. **Recompute and compile.** Establish the final answer, a stale-answer witness, and a verifier. Compile Direct, Introduce-Only, and Revision-Containing paths to the same endpoint.
4. **Validate the packet.** `taskrevisioneval validate` checks structure and executable answer verification. Human semantic review confirms that all three dialogues express the declared final task.
5. **Evaluate a model.** Run the paths independently, score each response with the shared verifier, and compute revision-sensitivity diagnostics.

To evaluate an OpenAI-compatible Chat Completions model, supply its model ID and API key at runtime:

```bash
taskrevisioneval evaluate \
  --packets data/examples/gsm8k_constructed_packet.json \
  --models YOUR_MODEL_ID \
  --output data/results/gsm8k_eval.json
```

The CLI reads `OPENAI_API_KEY` by default. Use `--base-url` and `--api-key-env` for another compatible service. Reports include raw model responses, per-path verifier results, and diagnostic metrics. `data/results/` is ignored by Git; review responses before sharing a report. The current CLI directly evaluates math, structured QA, and supported instruction-following packets. Other task families can be connected to their execution or judging backends.

## Development and license

```bash
python -m unittest discover -s tests -v
python -m scripts.check_release
python -m pip wheel . --no-deps -w dist
```

Project code is licensed under [MIT](LICENSE). Upstream datasets retain their own terms. The GSM8K example's attribution is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); contribution and release checks are in [CONTRIBUTING.md](CONTRIBUTING.md).
