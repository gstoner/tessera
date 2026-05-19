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
# Re-export the lightweight names directly; ``COMPILER_EXAMPLE_MANIFEST``
# is deliberately lazy because it pulls in ``examples.conformance.*``,
# which requires ``examples/`` to be on ``sys.path``.  Callers reach
# the manifest via ``tessera.testing.COMPILER_EXAMPLE_MANIFEST`` (lazy
# via the PEP 562 ``__getattr__`` below) or directly through
# ``from tessera.testing.compiler_examples import COMPILER_EXAMPLE_MANIFEST``.
from .compiler_examples import (
    COMPILER_STAGES,
    FOUNDATION_TARGETS,
    CompilerExample,
    CompilerExampleResult,
    qualify_compiler_example,
    qualify_compiler_examples,
)


def __getattr__(name):  # noqa: F811 (PEP 562 module __getattr__)
    if name == "COMPILER_EXAMPLE_MANIFEST":
        from . import compiler_examples
        return compiler_examples.COMPILER_EXAMPLE_MANIFEST
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
