# Contributing

Use Python 3.10 or later. Before submitting a change, run:

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
python -m scripts.check_release
python -m scripts.build_gsm8k_example --output /tmp/gsm8k_packet.json
taskrevisioneval validate --packets /tmp/gsm8k_packet.json
python -m pip wheel . --no-deps -w /tmp/taskrevisioneval-wheels
```

For new source adapters or packets, include source attribution, original
license, a deterministic reconstruction step, an executable verifier, and a
human review of endpoint equivalence. Do not submit credentials, private
endpoints, raw model logs, or personal data. Store API keys in the runtime
environment, not in tracked files.
