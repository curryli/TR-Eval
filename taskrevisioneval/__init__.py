"""
TaskRevisionEval: Diagnosing Revision Sensitivity in LLMs with Shared-Endpoint Controls.

This package provides research components for constructing,
validating, and evaluating revision-sensitive benchmark packets from public
benchmark seeds (IFEval, MATH-500, GSM8K, MMLU-Pro, TruthfulQA, HumanEval,
MBPP, BigCodeBench, BFCL, ToolSandbox, MultiChallenge, MT-Bench).

Components:
  - EBIR: Evolvable Benchmark Intermediate Representation
  - 6 task-family adapters: parse public seeds → typed revisions → verifiers
  - Construction agent: budgeted verifier-grounded construction (Algorithm 1)
  - Three-path evaluator: Direct / Introduce-Only / Revision-Containing (Algorithm 2)
  - Metrics: P_dir, P_inc, P_rev, P_all, G, RCFR (Equations 4–5, 11)
  - CLI: demo, validate, evaluate

See README.md for the full list of public benchmark sources and reproduction guide.
"""

__version__ = "0.1.0"

from .core.ebir import (
    EBIR, DialogueState, EvolutionTest, EvolvedTestSuite, CanonicalProbe,
    StateConstraint, InstructionConstraint, TransitionContract, LifecycleLabel,
    LifecycleEntry, OutputContract,
    RevisionType, RevisionDepth, SelectionLabel,
    PresentationPath, TerminalPacket, PathOutcome, PacketEvaluationResult,
)
from .adapters.base import BenchmarkAdapter
from .construction.engine import EvolutionEngine
from .construction.operators import OPERATOR_REGISTRY, OPERATOR_SPECS, OperatorSpec
from .construction.admission import AdmissionGate
from .construction.agent import ConstructionAgent, ConstructionResult
from .evaluation.verifier import VerifierResult, verify_ifeval_constraints, verify_numeric_answer
from .evaluation.metrics import (
    ModelDiagnostic, compute_model_diagnostic,
    SuiteSummary, compute_suite_summary,
    SelectionTransfer, compute_selection_transfer,
    bootstrap_gap_ci, paired_difference_se, compute_common_support_gap,
)
from .evaluation.evaluator import PathEvaluator, SuiteEvaluator, generate_primary_table, generate_contingency_table
