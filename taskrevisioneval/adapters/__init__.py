"""Task-family adapters for TaskRevisionEval."""
from .base import BenchmarkAdapter
from .ifeval import IFEvalAdapter
from .math import MathAdapter
from .structured_qa import StructuredQAAdapter
from .code import CodeAdapter
from .tool_use import ToolUseAdapter
from .native_multi_turn import NativeMultiTurnAdapter

__all__ = [
    "BenchmarkAdapter",
    "IFEvalAdapter",
    "MathAdapter",
    "StructuredQAAdapter",
    "CodeAdapter",
    "ToolUseAdapter",
    "NativeMultiTurnAdapter",
]
