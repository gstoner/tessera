"""
tessera.testing — test utilities for Tessera Phase 1.

Exports:
    MockRankGroup — thread-based fake multi-rank group for testing
                    index_launch and collective ops without NCCL/MPI.
"""

from .mock_collective import MockRankGroup, MockRank, MockCollectiveError
from .qa import (
    ChaosEvent,
    HealthSnapshot,
    PerformanceExpectation,
    RegressionBaseline,
    ReplayManifest,
    assert_close_to_reference,
    assert_deterministic,
    assert_finite,
)
from .compiler import CompilerHarnessResult, compile_and_maybe_launch
from .compiler_examples import (
    COMPILER_EXAMPLE_MANIFEST,
    COMPILER_STAGES,
    FOUNDATION_TARGETS,
    CompilerExample,
    CompilerExampleResult,
    qualify_compiler_example,
    qualify_compiler_examples,
)

__all__ = [
    "MockRankGroup",
    "MockRank",
    "MockCollectiveError",
    "ChaosEvent",
    "HealthSnapshot",
    "PerformanceExpectation",
    "RegressionBaseline",
    "ReplayManifest",
    "assert_close_to_reference",
    "assert_deterministic",
    "assert_finite",
    "CompilerHarnessResult",
    "compile_and_maybe_launch",
    "COMPILER_EXAMPLE_MANIFEST",
    "COMPILER_STAGES",
    "FOUNDATION_TARGETS",
    "CompilerExample",
    "CompilerExampleResult",
    "qualify_compiler_example",
    "qualify_compiler_examples",
]
