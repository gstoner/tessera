"""M3 follow-up — reference vs. optimized split metadata in benchmark JSON.

Closes the residual M3 gap: every :class:`BenchmarkRow` from
``benchmarks.common`` now carries an :class:`ExecutionKind` axis
that's independent of :class:`CompilerPath` (the path through
the compiler) and :class:`RuntimeStatus` (whether the runtime
executed).  The new axis records *which kind of execution
produced the numbers* — naive numpy reference, optimized native
backend, IR-only, or unknown (legacy rows).

These tests are fast and CPU-only so they run in the default
``-m "not slow"`` sweep (unlike ``test_benchmark_compiler_contract.py``
which is marked slow at the module level).
"""

from __future__ import annotations

import pytest

from benchmarks.common import (
    ArtifactLevels,
    BenchmarkOperator,
    BenchmarkRow,
    CompilerPath,
    ExecutionKind,
    RuntimeStatus,
)
from benchmarks.common.artifact_schema import telemetry_for_row


def _make_row(execution_kind: ExecutionKind = ExecutionKind.UNKNOWN) -> BenchmarkRow:
    return BenchmarkRow(
        operator=BenchmarkOperator("gemm", "f32", "16x16x16"),
        compiler_path=CompilerPath.TESSERA_JIT_CPU,
        runtime_status=RuntimeStatus.EXECUTABLE,
        artifact_levels=ArtifactLevels(graph=True, artifact_hash="abc"),
        execution_kind=execution_kind,
    )


def test_execution_kind_enum_values_are_canonical() -> None:
    """Adding a new ExecutionKind value is a deliberate decision."""
    assert {k.value for k in ExecutionKind} == {
        "reference",
        "optimized_native",
        "artifact_only",
        "unknown",
    }


def test_execution_kind_default_is_unknown() -> None:
    """Pre-existing rows that don't pass execution_kind read as
    ``unknown`` (legacy)."""
    row = BenchmarkRow(
        operator=BenchmarkOperator("gemm", "f32", "16x16x16"),
        compiler_path=CompilerPath.GRAPH_IR_ONLY,
        runtime_status=RuntimeStatus.ARTIFACT_ONLY,
    )
    assert row.execution_kind == ExecutionKind.UNKNOWN


def test_to_dict_surfaces_execution_kind() -> None:
    row = _make_row(ExecutionKind.OPTIMIZED_NATIVE)
    assert row.to_dict()["execution_kind"] == "optimized_native"


def test_flat_dict_surfaces_execution_kind() -> None:
    row = _make_row(ExecutionKind.REFERENCE)
    assert row.flat_dict()["execution_kind"] == "reference"


def test_telemetry_event_carries_execution_kind() -> None:
    row = _make_row(ExecutionKind.OPTIMIZED_NATIVE)
    event = telemetry_for_row(row)
    assert event["metadata"]["execution_kind"] == "optimized_native"


def test_reference_and_optimized_are_distinct_axes() -> None:
    """A numpy-reference path and an Accelerate-optimized path
    must serialize differently even when other fields match —
    the whole point of this axis."""
    ref = _make_row(ExecutionKind.REFERENCE)
    opt = _make_row(ExecutionKind.OPTIMIZED_NATIVE)
    assert ref.to_dict()["execution_kind"] != opt.to_dict()["execution_kind"]
    assert ref.flat_dict()["execution_kind"] != opt.flat_dict()["execution_kind"]


def test_execution_kind_is_orthogonal_to_compiler_path() -> None:
    """ExecutionKind labels *which numbers* were produced; CompilerPath
    labels *how the runtime got there*.  They are independent axes —
    e.g., a row can be ``TESSERA_JIT_CPU`` + ``REFERENCE`` (JIT'd a
    program but ran the numpy reference), or ``REFERENCE`` +
    ``OPTIMIZED_NATIVE`` (untracked legacy benchmark whose backend
    happens to be Accelerate)."""
    row = BenchmarkRow(
        operator=BenchmarkOperator("gemm", "f32", "16x16x16"),
        compiler_path=CompilerPath.TESSERA_JIT_CPU,
        runtime_status=RuntimeStatus.EXECUTABLE,
        execution_kind=ExecutionKind.REFERENCE,   # cross-axis combo
    )
    d = row.to_dict()
    assert d["compiler_path"] == "tessera_jit_cpu"
    assert d["execution_kind"] == "reference"


def test_artifact_only_row_uses_artifact_only_execution_kind() -> None:
    """Artifact-only paths (IR built, no runtime executed) should
    declare it on both axes — runtime_status and execution_kind."""
    row = BenchmarkRow(
        operator=BenchmarkOperator("gemm", "f32", "16x16x16"),
        compiler_path=CompilerPath.GRAPH_IR_ONLY,
        runtime_status=RuntimeStatus.ARTIFACT_ONLY,
        execution_kind=ExecutionKind.ARTIFACT_ONLY,
    )
    d = row.flat_dict()
    assert d["runtime_status"] == "artifact_only"
    assert d["execution_kind"] == "artifact_only"
