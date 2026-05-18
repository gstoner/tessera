"""M1 — CompileReport schema + canonical-program drivers.

The schema is the single inspectable envelope every frontend lane
(``@tessera.jit`` / textual / ``@clifford_jit``) emits.  These tests
lock:

  1. The frontend / value-kind / IR-layer enumerations stay normative.
  2. The dataclass refuses invalid values at construction.
  3. JSON serialization round-trips and is deterministic.
  4. ``report_hash()`` is stable across runs (excludes timing /
     version fields by design).
  5. The two shipped canonical programs build a valid report
     CPU-only and produce a deterministic hash.

M1.5 is covered separately in :mod:`test_canonical_program_registry`.
"""

from __future__ import annotations

import json
import sys

import pytest

from tessera.compiler import compile_report as cr
from tessera.compiler.canonical import (
    matmul_softmax_matmul,
    rotor_sandwich_norm,
)


# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------

def test_valid_frontends_is_exactly_three() -> None:
    """One lane per frontend.  Adding a new lane is a deliberate
    decision; this test catches accidental drift."""
    assert cr.VALID_FRONTENDS == frozenset({
        "tessera.jit", "textual", "clifford_jit",
    })


def test_valid_value_kinds_is_three_per_decision_15a() -> None:
    """Decision #15a — Multivector is a sibling value kind, not a 7th
    tensor attribute.  ``mixed`` is the explicit boundary case."""
    assert cr.VALID_VALUE_KINDS == frozenset({
        "tensor", "multivector", "mixed",
    })


def test_ir_layers_is_canonical_four() -> None:
    assert cr.IR_LAYERS == (
        "graph_ir", "schedule_ir", "tile_ir", "target_ir",
    )


def test_rejects_unknown_frontend() -> None:
    with pytest.raises(ValueError, match="frontend"):
        cr.CompileReport(
            program_id="x", source="src", frontend="bogus",
            value_kind="tensor", target="cpu",
        )


def test_rejects_unknown_value_kind() -> None:
    with pytest.raises(ValueError, match="value_kind"):
        cr.CompileReport(
            program_id="x", source="src", frontend="tessera.jit",
            value_kind="quaternion", target="cpu",
        )


def test_rejects_unknown_ir_layer() -> None:
    with pytest.raises(ValueError, match="ir_hashes"):
        cr.CompileReport(
            program_id="x", source="src", frontend="tessera.jit",
            value_kind="tensor", target="cpu",
            ir_hashes={"target_ir": "x", "bogus_layer": "y"},
        )


def test_as_dict_round_trips_through_json() -> None:
    r = cr.CompileReport(
        program_id="demo", source="t.run", frontend="tessera.jit",
        value_kind="tensor", target="cpu",
        ir_hashes={"graph_ir": "deadbeef", "target_ir": "cafebabe"},
        target_decision={"cpu": "numpy reference"},
        diagnostics=("note: cpu-only",),
        fallback_reason="non-darwin host",
        correctness={"max_abs_err": 0.0, "tolerance": 1e-6},
    )
    blob = r.as_json()
    reloaded = json.loads(blob)
    assert reloaded["program_id"] == "demo"
    assert reloaded["value_kind"] == "tensor"
    assert reloaded["ir_hashes"] == {"graph_ir": "deadbeef", "target_ir": "cafebabe"}
    assert reloaded["target_decision"] == {"cpu": "numpy reference"}


def test_report_hash_is_deterministic_for_same_content() -> None:
    """Same content ⇒ same hash, twice."""
    kwargs = dict(
        program_id="demo", source="t.run", frontend="tessera.jit",
        value_kind="tensor", target="cpu",
        ir_hashes={"graph_ir": "deadbeef"},
        correctness={"max_abs_err": 0.0, "tolerance": 1e-6},
    )
    a = cr.CompileReport(**kwargs)
    b = cr.CompileReport(**kwargs)
    assert a.report_hash() == b.report_hash()
    # Hash is a 16-char hex prefix.
    assert len(a.report_hash()) == 16


def test_report_hash_excludes_timing_and_version() -> None:
    """Timing + tessera_version are deliberately not part of the
    hash so reproducibility tests don't flake on wall-clock noise."""
    base = dict(
        program_id="demo", source="t.run", frontend="tessera.jit",
        value_kind="tensor", target="cpu",
    )
    a = cr.CompileReport(**base, timing_ms={"end_to_end": 1.0},
                          tessera_version="0.1.0")
    b = cr.CompileReport(**base, timing_ms={"end_to_end": 99.7},
                          tessera_version="0.2.99")
    assert a.report_hash() == b.report_hash()


def test_report_hash_responds_to_meaningful_change() -> None:
    """Changing correctness, ir_hashes, or target should change the
    hash; that's the property the drift gate relies on."""
    base = dict(
        program_id="demo", source="t.run", frontend="tessera.jit",
        value_kind="tensor", target="cpu",
    )
    h0 = cr.CompileReport(**base).report_hash()
    h1 = cr.CompileReport(**{**base, "target": "apple_gpu"}).report_hash()
    h2 = cr.CompileReport(
        **base, ir_hashes={"graph_ir": "x"},
    ).report_hash()
    assert len({h0, h1, h2}) == 3


# ---------------------------------------------------------------------------
# Canonical program — rotor_sandwich → norm
# ---------------------------------------------------------------------------

def test_rotor_sandwich_norm_runs_and_is_correct() -> None:
    """The GA vertical slice runs CPU-only (non-Darwin) or via
    @clifford_jit on Apple GPU.  Either way correctness must clear
    the report's stated tolerance."""
    report = rotor_sandwich_norm.run()
    assert report.program_id == "rotor_sandwich_norm"
    assert report.frontend == "clifford_jit"
    assert report.value_kind == "multivector"
    err = report.correctness["max_abs_err"]
    tol = report.correctness["tolerance"]
    assert err <= tol, f"max_abs_err={err} > tolerance={tol}"


def test_rotor_sandwich_norm_hash_is_deterministic() -> None:
    """Two back-to-back runs of the same program produce the same
    report hash — timing fields excluded."""
    h_a = rotor_sandwich_norm.run().report_hash()
    h_b = rotor_sandwich_norm.run().report_hash()
    assert h_a == h_b


def test_rotor_sandwich_norm_carries_ir_hash() -> None:
    """The IR digest must be populated — the report's whole point is
    to make IR identity visible."""
    report = rotor_sandwich_norm.run()
    assert "graph_ir" in report.ir_hashes
    assert len(report.ir_hashes["graph_ir"]) == 16


def test_rotor_sandwich_norm_proof_routes_on_darwin() -> None:
    """On Darwin the bridge trace must show clifford_rotor_sandwich
    + clifford_norm; on non-Darwin routes are empty (the fallback
    reason explains why)."""
    report = rotor_sandwich_norm.run()
    if sys.platform == "darwin":
        op_names = [r.op_name for r in report.proof_routes]
        assert op_names == ["clifford_rotor_sandwich", "clifford_norm"]
        assert report.fallback_reason is None
    else:
        assert report.proof_routes == ()
        assert report.fallback_reason is not None


# ---------------------------------------------------------------------------
# Canonical program — matmul → softmax → matmul
# ---------------------------------------------------------------------------

def test_matmul_softmax_matmul_runs_and_is_correct() -> None:
    """Tensor-side canonical program runs CPU-only and matches its
    own numpy reference within tolerance."""
    report = matmul_softmax_matmul.run()
    assert report.program_id == "matmul_softmax_matmul"
    assert report.frontend == "tessera.jit"
    assert report.value_kind == "tensor"
    err = report.correctness["max_abs_err"]
    tol = report.correctness["tolerance"]
    assert err <= tol, f"max_abs_err={err} > tolerance={tol}"


def test_matmul_softmax_matmul_hash_is_deterministic() -> None:
    h_a = matmul_softmax_matmul.run().report_hash()
    h_b = matmul_softmax_matmul.run().report_hash()
    assert h_a == h_b


def test_matmul_softmax_matmul_target_decision_includes_apple_gpu_on_darwin() -> None:
    """On Darwin the decision row must mention the fused 3-op MSL
    kernel symbol so reports show exactly which fast path the
    compiler intended."""
    report = matmul_softmax_matmul.run()
    if sys.platform == "darwin":
        assert "apple_gpu" in report.target_decision
        assert "matmul_softmax_matmul_f32" in report.target_decision["apple_gpu"]
        assert report.fallback_reason is None
    else:
        assert "cpu" in report.target_decision
        assert report.fallback_reason is not None


def test_matmul_softmax_matmul_carries_ir_hash() -> None:
    report = matmul_softmax_matmul.run()
    assert "graph_ir" in report.ir_hashes
    assert len(report.ir_hashes["graph_ir"]) == 16


# ---------------------------------------------------------------------------
# Cross-program — both shipped canonicals must be CPU-runnable
# ---------------------------------------------------------------------------

def test_both_shipped_canonicals_produce_json_round_trip() -> None:
    for runner in (rotor_sandwich_norm.run, matmul_softmax_matmul.run):
        report = runner()
        blob = report.as_json()
        reloaded = json.loads(blob)
        assert reloaded["program_id"] == report.program_id
        assert reloaded["frontend"] in cr.VALID_FRONTENDS
        assert reloaded["value_kind"] in cr.VALID_VALUE_KINDS
