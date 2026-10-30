"""Public command-line interface for frozen packet validation and evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..core.ebir import PathOutcome, TerminalPacket
from ..demo import build_demo_packets
from ..evaluation.evaluator import PathEvaluator
from ..evaluation.metrics import compute_model_diagnostic
from ..packet_validation import ADAPTERS, validate_packet


def _load_packets(path: Path) -> list[TerminalPacket]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("packet file must contain a nonempty JSON array")
    return [TerminalPacket.from_dict(item) for item in data]


def _validate_all(packets: list[TerminalPacket]) -> list[str]:
    errors = []
    seen = set()
    seen_instances = set()
    for packet in packets:
        if packet.packet_id in seen:
            errors.append(f"{packet.packet_id}: duplicate packet ID")
        seen.add(packet.packet_id)
        if packet.instance_id in seen_instances:
            errors.append(f"{packet.packet_id}: duplicate initialization source ID")
        seen_instances.add(packet.instance_id)
        errors.extend(f"{packet.packet_id}: {error}" for error in validate_packet(packet))
    return errors


class _ChatClient:
    def __init__(self, api_key_env: str, base_url: str | None, max_tokens: int):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the llm extra: pip install '.[llm]'") from exc
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"API credential environment variable {api_key_env} is not set")
        options = {"api_key": key, "timeout": 120}
        if base_url:
            options["base_url"] = base_url
        self._client = OpenAI(**options)
        self._max_tokens = max_tokens

    def chat(self, messages: list[dict], model: str = "") -> str:
        response = self._client.chat.completions.create(
            model=model, messages=messages, max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content or ""


def _evaluate(packets: list[TerminalPacket], models: list[str],
              replicates: int, client: _ChatClient) -> dict:
    records = []
    primary = {model: [] for model in models}
    for packet in packets:
        evaluator = PathEvaluator(ADAPTERS[packet.family](), client)
        for model in models:
            for replicate in range(replicates):
                path_results = {}
                for path in ("direct", "introduce_only", "revision_containing"):
                    passed, response, details = evaluator.evaluate_path(packet, path, model)
                    path_results[path] = {
                        "passed": passed, "response": response, "details": details,
                    }
                outcome = PathOutcome(
                    direct=path_results["direct"]["passed"],
                    introduce_only=path_results["introduce_only"]["passed"],
                    revision_containing=path_results["revision_containing"]["passed"],
                )
                records.append({
                    "packet_id": packet.packet_id, "instance_id": packet.instance_id,
                    "family": packet.family, "model": model,
                    "replicate": replicate + 1,
                    "paths": path_results, "category": outcome.diagnostic_category,
                })
                if replicate == 0:
                    primary[model].append(outcome)
    return {
        "schema_version": 1,
        "packet_count": len(packets),
        "replicates": replicates,
        "primary_replicate": 1,
        "models": models,
        "primary_diagnostics": {
            model: compute_model_diagnostic(model, outcomes).to_dict()
            for model, outcomes in primary.items()
        },
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="taskrevisioneval")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="write two hand-authored example packets")
    demo.add_argument("--output", type=Path, default=Path("demo_packets.json"))
    validate = sub.add_parser("validate", help="check frozen packet structure and verifiers")
    validate.add_argument("--packets", type=Path, required=True)
    evaluate = sub.add_parser("evaluate", help="run three-path evaluation on frozen packets")
    evaluate.add_argument("--packets", type=Path, required=True)
    evaluate.add_argument("--models", nargs="+", required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--replicates", type=int, default=3)
    evaluate.add_argument("--api-key-env", default="OPENAI_API_KEY")
    evaluate.add_argument("--base-url", default=None)
    evaluate.add_argument("--max-tokens", type=int, default=2000)
    args = parser.parse_args(argv)

    if args.command == "demo":
        packets = build_demo_packets()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps([p.to_dict() for p in packets], indent=2) + "\n",
                               encoding="utf-8")
        print(f"Wrote {len(packets)} example packets to {args.output}")
        return 0

    try:
        packets = _load_packets(args.packets)
        errors = _validate_all(packets)
        if errors:
            print("Packet validation failed:")
            for error in errors:
                print(f"  - {error}")
            return 1
        if args.command == "validate":
            print(f"Validated {len(packets)} packets structurally; semantic review remains required.")
            return 0
        if args.replicates < 1 or args.max_tokens < 1:
            raise ValueError("replicates and max-tokens must be positive")
        client = _ChatClient(args.api_key_env, args.base_url, args.max_tokens)
        report = _evaluate(packets, args.models, args.replicates, client)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {len(report['records'])} packet/model/replicate records to {args.output}")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
