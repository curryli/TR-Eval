"""Checks for the public package's runnable examples and release interface."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.build_gsm8k_example import build_packet
from taskrevisioneval.cli import _evaluate, _validate_all
from taskrevisioneval.core.ebir import TerminalPacket
from taskrevisioneval.demo import build_demo_packets
from taskrevisioneval.packet_validation import validate_packet


ROOT = Path(__file__).resolve().parents[1]


class FakeClient:
    def chat(self, messages, model=""):
        return "22"


class FailingClient:
    def chat(self, messages, model=""):
        raise RuntimeError("sensitive-marker")


class ReleaseTests(unittest.TestCase):
    def test_public_source_builds_checked_in_packet(self):
        source = json.loads((ROOT / "data/examples/gsm8k_source.json").read_text())
        packet = build_packet(source)
        checked_in = json.loads((ROOT / "data/examples/gsm8k_constructed_packet.json").read_text())[0]
        self.assertEqual(packet.to_dict(), checked_in)
        self.assertEqual(packet.accepted_answers, ["22"])
        self.assertEqual(packet.stale_answer_witness, "18")
        self.assertEqual(validate_packet(packet), [])

    def test_demo_packets_validate(self):
        self.assertEqual(_validate_all(build_demo_packets()), [])

    def test_example_evaluates_on_three_paths(self):
        source = json.loads((ROOT / "data/examples/gsm8k_source.json").read_text())
        report = _evaluate([build_packet(source)], ["fake"], 1, FakeClient())
        self.assertEqual(report["primary_diagnostics"]["fake"]["p_all"], 1.0)
        self.assertEqual(len(report["records"][0]["paths"]), 3)

    def test_unsupported_family_fails_closed(self):
        packet = TerminalPacket.from_dict(build_demo_packets()[0].to_dict())
        packet.family = "tool_use"
        self.assertTrue(validate_packet(packet))

    def test_provider_exception_text_is_not_written_to_report(self):
        source = json.loads((ROOT / "data/examples/gsm8k_source.json").read_text())
        report = _evaluate([build_packet(source)], ["fake"], 1, FailingClient())
        details = report["records"][0]["paths"]["direct"]["details"]
        self.assertNotIn("sensitive-marker", json.dumps(details))
        self.assertIn("RuntimeError", details["error"])


if __name__ == "__main__":
    unittest.main()
