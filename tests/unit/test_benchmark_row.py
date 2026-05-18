"""M5 — canonical benchmark-row schema + CompileReport bridge.

Locks the contract that every Tessera benchmark harness emits the
same field set, that the schema validator rejects unproven native
claims, and that the bridge between :class:`BenchmarkRow` and
:class:`CompileReport` round-trips losslessly.
"""

from __future__ import annotations

import pytest

from tessera.compiler.benchmark_row import (
    BenchmarkRow,
    NATIVE_MODES,
    NON_NATIVE_BACKENDS,
    OPTIONAL_BENCHMARK_FIELDS,
    REQUIRED_BENCHMARK_FIELDS,
    validate_benchmark_row,
)
from tessera.compiler.canonical import (
    matmul_softmax_matmul,
    rotor_sandwich_norm,
)


# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------

def test_required_fields_are_canonical_ten() -> None:
    """Adding a required field is a deliberate decision."""
    assert REQUIRED_BENCHMARK_FIELDS == frozenset({
        "namespace", "op", "backend", "shape", "dtype", "mode",
        "ok", "latency_ms", "device", "tessera_version",
    })


def test_native_modes_and_non_native_backends_are_closed_sets() -> None:
    """If a harness uses a new mode/backend string we need to update
    the validator first."""
    assert NATIVE_MODES == frozenset({
        "fused", "fused_chain", "jit_compiled", "native",
    })
    assert NON_NATIVE_BACKENDS == frozenset({
        "python_ref", "reference", "numpy",
    })


def test_required_and_optional_fields_are_disjoint() -> None:
    """The two sets must not overlap — otherwise the validator
    rejects rows with intentional duplicates."""
    assert not (REQUIRED_BENCHMARK_FIELDS & OPTIONAL_BENCHMARK_FIELDS)


# ---------------------------------------------------------------------------
# validate_benchmark_row
# ---------------------------------------------------------------------------

def _minimal_python_ref_row() -> dict:
    return {
        "namespace": "test", "op": "x", "backend": "python_ref",
        "shape": "M=4,N=4", "dtype": "fp32",
        "mode": "reference", "ok": True, "latency_ms": 0.1,
        "device": "cpu", "tessera_version": "0.1.0",
    }


def test_validate_accepts_minimal_reference_row() -> None:
    validate_benchmark_row(_minimal_python_ref_row())


def test_validate_rejects_missing_required_field() -> None:
    row = _minimal_python_ref_row()
    del row["latency_ms"]
    with pytest.raises(ValueError, match="missing required fields"):
        validate_benchmark_row(row)


def test_validate_rejects_unknown_field() -> None:
    row = _minimal_python_ref_row()
    row["bogus_field"] = 0
    with pytest.raises(ValueError, match="unknown fields"):
        validate_benchmark_row(row)


def test_validate_rejects_native_claim_without_proof() -> None:
    """The M5 no-silent-native rule: claiming a non-reference
    backend in a native mode requires a route / artifact / plan
    hash / symbol proof."""
    row = _minimal_python_ref_row()
    row["backend"] = "apple_gpu"
    row["mode"] = "fused"
    with pytest.raises(ValueError, match="claims native execution"):
        validate_benchmark_row(row)


def test_validate_accepts_native_claim_with_symbols_proof() -> None:
    row = _minimal_python_ref_row()
    row["backend"] = "apple_gpu"
    row["mode"] = "fused"
    row["symbols"] = ["tessera_apple_gpu_matmul_f32"]
    validate_benchmark_row(row)


def test_validate_accepts_native_claim_with_plan_hash_proof() -> None:
    row = _minimal_python_ref_row()
    row["backend"] = "apple_gpu"
    row["mode"] = "jit_compiled"
    row["plan_hash"] = "abc123"
    validate_benchmark_row(row)


# ---------------------------------------------------------------------------
# BenchmarkRow.from_compile_report — the M1 → M5 bridge
# ---------------------------------------------------------------------------

def test_from_compile_report_round_trips_canonical_program() -> None:
    """A canonical program's report builds a valid benchmark row."""
    report = matmul_softmax_matmul.run()
    row = BenchmarkRow.from_compile_report(
        report,
        namespace="canonical",
        shape="M=32,N=32,K=32",
        device="apple_silicon_metal",
        reps=1,
    )
    d = row.as_dict()
    # Schema-validates without raising — i.e., it has all required
    # fields and (where it claims native) carries the proof.
    validate_benchmark_row(d)
    # Carries the CompileReport hash for cross-reference.
    assert row.report_hash == report.report_hash()


def test_from_compile_report_preserves_correctness_envelope() -> None:
    report = rotor_sandwich_norm.run()
    row = BenchmarkRow.from_compile_report(
        report, namespace="canonical", shape="B=8/Cl(3,0)",
        device="apple_silicon_metal",
    )
    if report.correctness is not None:
        assert row.max_abs_err == report.correctness["max_abs_err"]
        assert row.tolerance == report.correctness["tolerance"]


def test_from_compile_report_does_not_claim_native_for_cpu_target() -> None:
    """The bridge must NOT silently flip a CPU-targeted report into a
    native claim — that's the M5 invariant the validator enforces."""
    import sys
    if sys.platform == "darwin":
        pytest.skip("CPU-target path only exercised on non-Darwin")
    report = matmul_softmax_matmul.run()
    row = BenchmarkRow.from_compile_report(
        report, namespace="canonical", shape="M=32,N=32,K=32",
        device="cpu",
    )
    assert row.backend == "python_ref"
    assert row.mode == "reference"


# ---------------------------------------------------------------------------
# Round-trip: row → report → row
# ---------------------------------------------------------------------------

def test_to_compile_report_round_trip_preserves_essentials() -> None:
    report = matmul_softmax_matmul.run()
    row = BenchmarkRow.from_compile_report(
        report, namespace="canonical", shape="M=32,N=32,K=32",
    )
    reloaded = row.to_compile_report()
    assert reloaded.program_id == report.program_id
    assert reloaded.frontend == report.frontend
    assert reloaded.value_kind == report.value_kind
    assert reloaded.target == report.target
    assert reloaded.ir_hashes == report.ir_hashes
    assert reloaded.target_decision == report.target_decision


# ---------------------------------------------------------------------------
# Symbol provenance from proof routes
# ---------------------------------------------------------------------------

def test_from_compile_report_extracts_symbols_from_routes() -> None:
    """When a CompileReport carries bridge routes (Darwin), the
    row's ``symbols`` list pulls the manifest-resolved C ABI names
    so the benchmark JSON is self-describing."""
    import sys
    if sys.platform != "darwin":
        pytest.skip("Bridge routes only populate on Darwin")
    report = rotor_sandwich_norm.run()
    row = BenchmarkRow.from_compile_report(
        report, namespace="canonical", shape="B=8/Cl(3,0)",
        device="apple_silicon_metal",
    )
    symbols = set(row.symbols)
    assert "tessera_apple_gpu_clifford_rotor_sandwich_cl30_f32" in symbols
    assert "tessera_apple_gpu_clifford_norm_cl30_f32" in symbols
