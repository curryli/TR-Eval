"""
Verifier system: determines whether model responses satisfy state constraints.

Priority: deterministic > executable > structured judge > general LLM judge.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import re


@dataclass
class VerifierResult:
    """Verifier result."""
    passed: bool
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    constraint_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "reasons": self.reasons,
            "details": self.details,
            "constraint_results": self.constraint_results,
        }

    @classmethod
    def pass_(cls, details: dict | None = None) -> "VerifierResult":
        return cls(passed=True, details=details or {})

    @classmethod
    def fail(cls, reasons: list[str], details: dict | None = None) -> "VerifierResult":
        return cls(passed=False, reasons=reasons, details=details or {})


# ============================================================
# Deterministic constraint checkers (IFEval-style)
# ============================================================

def _check_relation(actual: int, expected: int, relation: str) -> bool:
    if relation == "at least":
        return actual >= expected
    if relation == "at most":
        return actual <= expected
    if relation == "exactly":
        return actual == expected
    if relation == "less than":
        return actual < expected
    if relation == "more than":
        return actual > expected
    return False


def _check_length_constraints(response: str, kwargs: dict) -> VerifierResult:
    """Length constraint checks (word count, sentence count, paragraph count)."""
    constraint_results = []
    reasons = []

    if "num_words" in kwargs and kwargs["num_words"] is not None:
        num_words = kwargs["num_words"]
        relation = kwargs.get("relation", "at least")
        actual = len(response.split())
        passed = _check_relation(actual, num_words, relation)
        constraint_results.append({"type": "number_words", "expected": f"{relation} {num_words}",
                                   "actual": actual, "passed": passed})
        if not passed:
            reasons.append(f"word count {actual} != {relation} {num_words}")

    if "num_sentences" in kwargs and kwargs["num_sentences"] is not None:
        num_sentences = kwargs["num_sentences"]
        relation = kwargs.get("relation", "at least")
        actual = len(re.findall(r"[.!?]+", response))
        passed = _check_relation(actual, num_sentences, relation)
        constraint_results.append({"type": "number_sentences", "expected": f"{relation} {num_sentences}",
                                   "actual": actual, "passed": passed})
        if not passed:
            reasons.append(f"sentence count {actual} != {relation} {num_sentences}")

    if "num_paragraphs" in kwargs and kwargs["num_paragraphs"] is not None:
        num_paragraphs = kwargs["num_paragraphs"]
        actual = len([p for p in response.split("\n\n") if p.strip()])
        passed = actual == num_paragraphs
        constraint_results.append({"type": "number_paragraphs", "expected": num_paragraphs,
                                   "actual": actual, "passed": passed})
        if not passed:
            reasons.append(f"paragraph count {actual} != {num_paragraphs}")

    return VerifierResult(passed=not reasons, reasons=reasons,
                          constraint_results=constraint_results)


def _check_keyword_constraints(response: str, kwargs: dict) -> VerifierResult:
    """Keyword constraint checks (contain/forbid keywords)."""
    constraint_results = []
    reasons = []

    if "keyword" in kwargs and kwargs["keyword"]:
        keyword = kwargs["keyword"]
        passed = keyword.lower() in response.lower()
        constraint_results.append({"type": "contain_keyword", "keyword": keyword, "passed": passed})
        if not passed:
            reasons.append(f"keyword '{keyword}' not found")

    if "forbidden_words" in kwargs and kwargs["forbidden_words"]:
        forbidden = kwargs["forbidden_words"]
        found = [w for w in forbidden if w.lower() in response.lower()]
        passed = not found
        constraint_results.append({"type": "forbid_keywords", "forbidden": forbidden,
                                   "found": found, "passed": passed})
        if not passed:
            reasons.append(f"forbidden words found: {found}")

    if "first_word" in kwargs and kwargs["first_word"]:
        first_word = kwargs["first_word"]
        actual_first = response.strip().split()[0] if response.strip() else ""
        passed = actual_first.lower() == first_word.lower()
        constraint_results.append({"type": "first_word", "expected": first_word,
                                   "actual": actual_first, "passed": passed})
        if not passed:
            reasons.append(f"first word '{actual_first}' != '{first_word}'")

    return VerifierResult(passed=not reasons, reasons=reasons,
                          constraint_results=constraint_results)


def _check_format_constraints(response: str, kwargs: dict) -> VerifierResult:
    """Format constraint checks (JSON, markdown, etc.)."""
    constraint_results = []
    reasons = []

    fmt = kwargs.get("format", "")
    if fmt == "json":
        import json as json_mod
        try:
            json_mod.loads(response.strip())
            constraint_results.append({"type": "json", "passed": True})
        except Exception:
            constraint_results.append({"type": "json", "passed": False})
            reasons.append("response is not valid JSON")

    if "case" in kwargs and kwargs["case"]:
        case_type = kwargs["case"]
        if case_type == "uppercase":
            passed = response.isupper()
            constraint_results.append({"type": "case", "expected": "uppercase", "passed": passed})
            if not passed: reasons.append("response is not all uppercase")
        elif case_type == "lowercase":
            passed = response.islower()
            constraint_results.append({"type": "case", "expected": "lowercase", "passed": passed})
            if not passed: reasons.append("response is not all lowercase")

    return VerifierResult(passed=not reasons, reasons=reasons,
                          constraint_results=constraint_results)


def verify_ifeval_constraints(response: str, instruction_id_list: list[str],
                                kwargs_list: list[dict]) -> VerifierResult:
    """Verify IFEval-style constraints deterministically."""
    if not instruction_id_list or len(instruction_id_list) != len(kwargs_list):
        return VerifierResult.fail(["missing or misaligned instruction checks"])
    supported = {
        "length_constraints:number_words": "num_words",
        "length_constraints:number_sentences": "num_sentences",
        "length_constraints:number_paragraphs": "num_paragraphs",
        "keywords:contain": "keyword",
        "keywords:forbid": "forbidden_words",
        "keywords:first_word": "first_word",
        "format:json": "format",
        "format:case": "case",
    }
    all_results = []
    all_reasons = []

    for inst_id, kwargs in zip(instruction_id_list, kwargs_list):
        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        required_key = supported.get(inst_id)
        if required_key is None or required_key not in clean_kwargs:
            all_reasons.append(f"[{inst_id}] unsupported or incomplete instruction")
            continue
        if clean_kwargs[required_key] in ("", [], None):
            all_reasons.append(f"[{inst_id}] empty instruction value")
            continue
        if inst_id == "format:json" and clean_kwargs["format"] != "json":
            all_reasons.append(f"[{inst_id}] unsupported format")
            continue
        if inst_id == "format:case" and clean_kwargs["case"] not in ("uppercase", "lowercase"):
            all_reasons.append(f"[{inst_id}] unsupported case rule")
            continue

        # A single instruction may contain more than one independently checked rule.
        checks: list[VerifierResult] = []
        if any(k in clean_kwargs for k in ("num_words", "num_sentences", "num_paragraphs")):
            checks.append(_check_length_constraints(response, clean_kwargs))
        if any(k in clean_kwargs for k in ("keyword", "forbidden_words", "first_word")):
            checks.append(_check_keyword_constraints(response, clean_kwargs))
        if any(k in clean_kwargs for k in ("format", "case")):
            checks.append(_check_format_constraints(response, clean_kwargs))
        if not checks:
            checks.append(VerifierResult.fail([f"unsupported instruction: {inst_id}"]))

        for result in checks:
            all_results.extend(result.constraint_results)
            if not result.passed:
                all_reasons.extend(f"[{inst_id}] {r}" for r in result.reasons)

    return VerifierResult(passed=not all_reasons, reasons=all_reasons,
                           constraint_results=all_results)


def verify_numeric_answer(response: str, expected: str, tolerance: float = 0.001) -> VerifierResult:
    """Verify a numeric answer by extracting and comparing numbers."""
    numbers = re.findall(r"-?\d+\.?\d*", response)
    if not numbers:
        return VerifierResult.fail(["no numeric answer found in response"])

    actual = float(numbers[-1])
    try:
        exp = float(expected)
    except ValueError:
        return VerifierResult.fail([f"expected answer '{expected}' is not numeric"])

    passed = abs(actual - exp) <= tolerance
    return VerifierResult(
        passed=passed,
        reasons=[] if passed else [f"numeric answer {actual} != expected {exp}"],
        details={"actual": actual, "expected": exp, "tolerance": tolerance},
    )


def verify_exact_text(response: str, expected: str, case_sensitive: bool = False) -> VerifierResult:
    """Verify exact text match (after normalization)."""
    r = response.strip() if not case_sensitive else response.strip()
    e = expected.strip() if not case_sensitive else expected.strip()
    if not case_sensitive:
        r, e = r.lower(), e.lower()
    passed = e in r
    return VerifierResult(
        passed=passed,
        reasons=[] if passed else [f"expected text '{expected}' not found in response"],
        details={"expected": expected},
    )


def verify_code(response: str, test_cases: list[dict]) -> VerifierResult:
    """Refuse code scoring without an isolated execution backend."""
    return VerifierResult.fail(["code execution backend is not configured"])


# ============================================================
# LLM Judge Verifier (supplementary)
# ============================================================

class LLMJudgeVerifier:
    """LLM-based semantic judge for tasks without deterministic verifiers."""

    def __init__(self, llm_client):
        self._llm = llm_client

    def judge(self, response: str, state: DialogueState, rubric: str = "") -> VerifierResult:
        """Judge response using LLM with a rubric."""
        if not self._llm:
            return VerifierResult.fail(["no LLM client available for judging"])

        prompt = self._build_judge_prompt(response, state, rubric)
        try:
            result = self._llm.chat_json(prompt, model=self._llm.model)
            passed = result.get("verdict", "").lower() in ("pass", "correct", "yes", True)
            reasons = result.get("reasons", [])
            return VerifierResult(passed=passed, reasons=reasons, details=result)
        except Exception as e:
            return VerifierResult.fail([f"LLM judge error: {e}"])

    def _build_judge_prompt(self, response: str, state: DialogueState, rubric: str) -> str:
        return (
            f"You are an expert judge. Evaluate whether the response satisfies all "
            f"active constraints.\n\n"
            f"Goal: {state.goal}\n"
            f"Active constraints: {[c.to_dict() for c in state.active_constraints()]}\n"
            f"Output contract: {state.output_contract.to_dict()}\n"
            f"Rubric: {rubric}\n\n"
            f"Response to evaluate:\n{response}\n\n"
            f"Return JSON: {{\"verdict\": \"pass\"|\"fail\", \"reasons\": [...]}}"
        )
