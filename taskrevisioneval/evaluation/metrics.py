"""
TaskRevisionEval Diagnostic Metrics.

Paper Section 3.2 (Equations 4-5, 11):
  - Four-way hierarchical outcome distribution: P_dir, P_inc, P_rev, P_all
  - Within-model controlled gap: G_{I-R|D}
  - Supplementary RCFR
  - Raw R-failure decomposition
  - Suite-level summaries and paired suite contrasts
"""

from __future__ import annotations
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..core.ebir import PathOutcome, PacketEvaluationResult


# ============================================================
# Per-model diagnostic (paper Eq. 4-5, 11)
# ============================================================

@dataclass
class ModelDiagnostic:
    """All diagnostic summaries for one model on one suite."""
    model: str
    num_packets: int = 0
    # Path success rates (unconditioned)
    direct_rate: float = 0.0
    introduce_rate: float = 0.0
    revision_rate: float = 0.0
    # Four-way hierarchical outcome distribution (paper Eq. 4)
    p_dir: float = 0.0   # P(D=0) — direct failure
    p_inc: float = 0.0   # P(D=1, I=0) — incremental failure after direct success
    p_rev: float = 0.0   # P(D=1, I=1, R=0) — strict RCF mass
    p_all: float = 0.0   # P(D=1, I=1, R=1) — all-path success
    # Controlled gap G_{I-R|D} (paper Eq. 5)
    gap_ir_given_d: float = 0.0
    n_d: int = 0  # denominator: count of D=1
    # Supplementary RCFR (paper Eq. 11)
    rcfr: float = 0.0
    n_di: int = 0  # denominator: count of D=1 and I=1
    # Raw R-failure decomposition
    r_fail_total: int = 0
    r_fail_end: int = 0     # R=0 and D=0
    r_fail_inc: int = 0     # R=0 and D=1, I=0
    r_fail_rcf: int = 0     # R=0 and D=1, I=1 (strict RCF)
    # Eight-cell contingency (paper Appendix Table 22)
    contingency: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model": self.model, "num_packets": self.num_packets,
            "direct_rate": round(self.direct_rate, 4),
            "introduce_rate": round(self.introduce_rate, 4),
            "revision_rate": round(self.revision_rate, 4),
            "p_dir": round(self.p_dir, 4), "p_inc": round(self.p_inc, 4),
            "p_rev": round(self.p_rev, 4), "p_all": round(self.p_all, 4),
            "gap_ir_given_d": round(self.gap_ir_given_d, 4),
            "n_d": self.n_d, "rcfr": round(self.rcfr, 4), "n_di": self.n_di,
            "r_fail_total": self.r_fail_total, "r_fail_end": self.r_fail_end,
            "r_fail_inc": self.r_fail_inc, "r_fail_rcf": self.r_fail_rcf,
            "contingency": self.contingency,
        }


def compute_model_diagnostic(
    model: str,
    outcomes: list[PathOutcome],
) -> ModelDiagnostic:
    """
    Compute all diagnostic summaries for one model (paper Eq. 4-5, 11).

    Args:
        model: model name
        outcomes: list of PathOutcome, one per packet
    """
    diag = ModelDiagnostic(model=model, num_packets=len(outcomes))
    if not outcomes:
        return diag

    n = len(outcomes)
    d_count = sum(1 for o in outcomes if o.direct is True)
    i_count = sum(1 for o in outcomes if o.introduce_only is True)
    r_count = sum(1 for o in outcomes if o.revision_containing is True)

    diag.direct_rate = d_count / n
    diag.introduce_rate = i_count / n
    diag.revision_rate = r_count / n
    diag.n_d = d_count

    # Four-way hierarchical distribution (paper Eq. 4)
    p_dir = sum(1 for o in outcomes if o.direct is not True) / n
    p_inc = sum(1 for o in outcomes if o.direct is True and o.introduce_only is not True) / n
    p_rev = sum(1 for o in outcomes if o.is_rcf) / n
    p_all = sum(1 for o in outcomes if o.is_all_pass) / n

    diag.p_dir = p_dir
    diag.p_inc = p_inc
    diag.p_rev = p_rev
    diag.p_all = p_all

    # Controlled gap G_{I-R|D} (paper Eq. 5)
    if d_count > 0:
        sum_i = sum(1 for o in outcomes if o.direct is True and o.introduce_only is True)
        sum_r = sum(1 for o in outcomes if o.direct is True and o.revision_containing is True)
        diag.gap_ir_given_d = (sum_i - sum_r) / d_count

    # Supplementary RCFR (paper Eq. 11)
    n_di = sum(1 for o in outcomes if o.direct is True and o.introduce_only is True)
    diag.n_di = n_di
    if n_di > 0:
        n_rcf = sum(1 for o in outcomes if o.is_rcf)
        diag.rcfr = n_rcf / n_di

    # Raw R-failure decomposition
    diag.r_fail_total = sum(1 for o in outcomes if o.revision_containing is not True)
    diag.r_fail_end = sum(1 for o in outcomes if o.revision_containing is not True and o.direct is not True)
    diag.r_fail_inc = sum(1 for o in outcomes if o.revision_containing is not True
                          and o.direct is True and o.introduce_only is not True)
    diag.r_fail_rcf = sum(1 for o in outcomes if o.is_rcf)

    # Eight-cell contingency
    for o in outcomes:
        d_val = "P" if o.direct is True else "F"
        i_val = "P" if o.introduce_only is True else "F"
        r_val = "P" if o.revision_containing is True else "F"
        key = f"D={d_val},I={i_val},R={r_val}"
        diag.contingency[key] = diag.contingency.get(key, 0) + 1

    return diag


# ============================================================
# Suite-level summary
# ============================================================

@dataclass
class SuiteSummary:
    """Summary statistics for one suite across all models."""
    suite_name: str
    num_packets: int = 0
    model_diagnostics: dict[str, ModelDiagnostic] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "suite_name": self.suite_name, "num_packets": self.num_packets,
            "model_diagnostics": {k: v.to_dict() for k, v in self.model_diagnostics.items()},
        }


def compute_suite_summary(
    suite_name: str,
    num_packets: int,
    results: list[PacketEvaluationResult],
) -> SuiteSummary:
    """Compute suite-level summary from per-packet evaluation results."""
    summary = SuiteSummary(suite_name=suite_name, num_packets=num_packets)

    # Group outcomes by model
    model_outcomes: dict[str, list[PathOutcome]] = {}
    for r in results:
        if r.model not in model_outcomes:
            model_outcomes[r.model] = []
        model_outcomes[r.model].append(r.outcomes)

    for model, outcomes in model_outcomes.items():
        summary.model_diagnostics[model] = compute_model_diagnostic(model, outcomes)

    return summary


# ============================================================
# Paired suite contrast (paper Section 6.2, Eq. 12)
# ============================================================

@dataclass
class SelectionTransfer:
    """Challenge-minus-First-valid contrast (paper Section 6.2)."""
    model: str
    delta_p_rev: float = 0.0
    delta_gap: float = 0.0
    delta_p_all: float = 0.0

    def to_dict(self) -> dict:
        return {"model": self.model, "delta_p_rev": round(self.delta_p_rev, 4),
                "delta_gap": round(self.delta_gap, 4),
                "delta_p_all": round(self.delta_p_all, 4)}


def compute_selection_transfer(
    first_valid: SuiteSummary,
    challenge: SuiteSummary,
) -> list[SelectionTransfer]:
    """Compute Challenge-minus-First-valid contrasts for all shared models."""
    transfers = []
    for model in challenge.model_diagnostics:
        if model in first_valid.model_diagnostics:
            ch = challenge.model_diagnostics[model]
            fv = first_valid.model_diagnostics[model]
            transfers.append(SelectionTransfer(
                model=model,
                delta_p_rev=ch.p_rev - fv.p_rev,
                delta_gap=ch.gap_ir_given_d - fv.gap_ir_given_d,
                delta_p_all=ch.p_all - fv.p_all,
            ))
    return transfers


# ============================================================
# Confidence intervals (paper Section 5.3)
# ============================================================

def bootstrap_gap_ci(
    outcomes: list[PathOutcome],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for G_{I-R|D}."""
    import random
    rng = random.Random(seed)
    n = len(outcomes)
    if n == 0:
        return (0.0, 0.0)

    gaps = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(outcomes) for _ in range(n)]
        d_count = sum(1 for o in sample if o.direct is True)
        if d_count == 0:
            gaps.append(0.0)
            continue
        sum_i = sum(1 for o in sample if o.direct is True and o.introduce_only is True)
        sum_r = sum(1 for o in sample if o.direct is True and o.revision_containing is True)
        gaps.append((sum_i - sum_r) / d_count)

    gaps.sort()
    alpha = (1 - confidence) / 2
    lower = gaps[int(alpha * n_bootstrap)]
    upper = gaps[int((1 - alpha) * n_bootstrap)]
    return (lower, upper)


def paired_difference_se(
    n_pf: int, n_fp: int, n_d: int, g: float,
) -> float:
    """
    Paired-difference standard error for G (paper Section 5.3).

    SE = sqrt((n_PF + n_FP)/n_D - (G/100)^2) / n_D
    """
    if n_d == 0:
        return 0.0
    variance = (n_pf + n_fp) / n_d - (g / 100.0) ** 2
    variance = max(variance, 0.0)
    return math.sqrt(variance / n_d)


# ============================================================
# Common-support analysis (paper Appendix D.1, Eq. 9)
# ============================================================

def compute_common_support_gap(
    all_model_outcomes: dict[str, list[PathOutcome]],
) -> dict[str, float]:
    """
    Paper Eq. 9: G_common computed on the intersection of packets
    that every evaluated model passes on Direct.
    """
    # Find packets where all models pass Direct
    n_models = len(all_model_outcomes)
    if n_models == 0:
        return {}

    model_names = list(all_model_outcomes.keys())
    n_packets = len(all_model_outcomes[model_names[0]])

    common_indices = []
    for i in range(n_packets):
        if all(all_model_outcomes[m][i].direct is True for m in model_names):
            common_indices.append(i)

    if not common_indices:
        return {}

    result = {}
    for model in model_names:
        common_outcomes = [all_model_outcomes[model][i] for i in common_indices]
        n_common = len(common_outcomes)
        sum_i = sum(1 for o in common_outcomes if o.introduce_only is True)
        sum_r = sum(1 for o in common_outcomes if o.revision_containing is True)
        result[model] = (sum_i - sum_r) / n_common

    return result
