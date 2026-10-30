"""
Model Evaluator: runs models on three paths and collects outcomes.

Paper Algorithm 2: Controlled three-path execution and RCF labeling.
For each frozen packet, run the model on Direct, Introduce-Only, and
Revision-Containing in independently reset sessions, then score with V_u.
"""

from __future__ import annotations
import json
from typing import Any, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.ebir import (
    TerminalPacket, PresentationPath, PathOutcome, PacketEvaluationResult,
)
from .verifier import VerifierResult
from .metrics import compute_model_diagnostic, ModelDiagnostic, SuiteSummary, compute_suite_summary

if TYPE_CHECKING:
    from ..adapters.base import BenchmarkAdapter


class PathEvaluator:
    """
    Evaluate a single model on a single path of a single packet.

    Paper Algorithm 2, lines 2-4: Run isolated session and score endpoint.
    """

    def __init__(self, adapter: BenchmarkAdapter, llm_client=None):
        self._adapter = adapter
        self._llm = llm_client

    def evaluate_path(
        self,
        packet: TerminalPacket,
        path_type: str,
        model: str = "",
    ) -> tuple[bool, str, dict]:
        """
        Run model on one path and score the endpoint response.

        Returns: (passed, response, verifier_details)
        """
        path = packet.paths.get(path_type)
        if not path or not path.dialogue:
            return False, "", {"error": "path not found or empty"}

        # Run model on the path dialogue
        messages = [{"role": m["role"], "content": m["content"]} for m in path.dialogue]
        try:
            if self._llm and model:
                response = self._llm.chat(messages, model=model)
            elif self._llm:
                response = self._llm.chat(messages)
            else:
                response = "[no LLM client available]"
        except Exception as e:
            return False, "", {"error": f"model call failed: {type(e).__name__}"}

        # Score the endpoint response with the packet's verifier
        try:
            result = self._adapter.verify_response(response, packet.endpoint_state)
            return result.passed, response, {"verifier": result.to_dict()}
        except Exception as e:
            return False, response, {"error": f"verifier error: {type(e).__name__}"}


class SuiteEvaluator:
    """
    Evaluate models across an entire suite of packets.

    Paper Algorithm 2: For each packet, evaluate all three paths for each model.
    Supports batch evaluation with thread-level parallelism.
    """

    def __init__(
        self,
        adapter: BenchmarkAdapter,
        llm_client=None,
        max_workers: int = 4,
        replicates: int = 1,
    ):
        self._adapter = adapter
        self._llm = llm_client
        self._max_workers = max_workers
        self._replicates = replicates
        self._path_eval = PathEvaluator(adapter, llm_client)

    def evaluate_packet(
        self,
        packet: TerminalPacket,
        models: list[str],
    ) -> list[PacketEvaluationResult]:
        """Evaluate all models on all three paths of one packet."""
        results = []
        for model in models:
            for rep in range(self._replicates):
                d_pass, d_resp, d_det = self._path_eval.evaluate_path(
                    packet, "direct", model if model != "construction_target" else "")
                i_pass, i_resp, i_det = self._path_eval.evaluate_path(
                    packet, "introduce_only", model if model != "construction_target" else "")
                r_pass, r_resp, r_det = self._path_eval.evaluate_path(
                    packet, "revision_containing", model if model != "construction_target" else "")

                outcome = PathOutcome(direct=d_pass, introduce_only=i_pass,
                                       revision_containing=r_pass)
                result = PacketEvaluationResult(
                    packet_id=packet.packet_id, model=model,
                    outcomes=outcome,
                    responses={"direct": d_resp, "introduce_only": i_resp,
                               "revision_containing": r_resp},
                    verifier_details={"direct": d_det, "introduce_only": i_det,
                                      "revision_containing": r_det},
                    failure_type=outcome.diagnostic_category,
                )
                results.append(result)
        return results

    def evaluate_suite(
        self,
        packets: list[TerminalPacket],
        models: list[str],
        suite_name: str = "first_valid",
    ) -> SuiteSummary:
        """Evaluate all packets in a suite for all models."""
        all_results: list[PacketEvaluationResult] = []

        for packet in packets:
            results = self.evaluate_packet(packet, models)
            all_results.extend(results)

        return compute_suite_summary(suite_name, len(packets), all_results)

    def evaluate_suite_parallel(
        self,
        packets: list[TerminalPacket],
        models: list[str],
        suite_name: str = "first_valid",
    ) -> SuiteSummary:
        """Evaluate suite with thread-level parallelism across packets."""
        all_results: list[PacketEvaluationResult] = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(self.evaluate_packet, pkt, models): pkt for pkt in packets}
            for future in as_completed(futures):
                try:
                    all_results.extend(future.result())
                except Exception:
                    pass

        return compute_suite_summary(suite_name, len(packets), all_results)


def generate_primary_table(
    first_valid: SuiteSummary,
    challenge: SuiteSummary,
) -> list[dict]:
    """
    Generate the primary results table (paper Table 3).

    Columns: Model, First-valid P_all/P_rev/G, Challenge P_all/P_rev/G, Delta P_rev/G
    """
    rows = []
    for model in challenge.model_diagnostics:
        ch = challenge.model_diagnostics[model]
        fv = first_valid.model_diagnostics.get(model)
        if fv is None:
            continue
        rows.append({
            "model": model,
            "fv_p_all": round(fv.p_all * 100, 1),
            "fv_p_rev": round(fv.p_rev * 100, 1),
            "fv_gap": round(fv.gap_ir_given_d * 100, 1),
            "ch_p_all": round(ch.p_all * 100, 1),
            "ch_p_rev": round(ch.p_rev * 100, 1),
            "ch_gap": round(ch.gap_ir_given_d * 100, 1),
            "delta_p_rev": round((ch.p_rev - fv.p_rev) * 100, 1),
            "delta_gap": round((ch.gap_ir_given_d - fv.gap_ir_given_d) * 100, 1),
        })
    return rows


def generate_contingency_table(
    summary: SuiteSummary,
) -> list[dict]:
    """Generate the full (D,I,R) contingency table (paper Table 22)."""
    rows = []
    for model, diag in summary.model_diagnostics.items():
        row = {"model": model}
        for key in ["D=F,I=F,R=F", "D=F,I=F,R=P", "D=F,I=P,R=F", "D=F,I=P,R=P",
                     "D=P,I=F,R=F", "D=P,I=F,R=P", "D=P,I=P,R=F", "D=P,I=P,R=P"]:
            row[key] = diag.contingency.get(key, 0)
        rows.append(row)
    return rows
