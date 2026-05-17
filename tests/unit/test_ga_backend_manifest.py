"""GA9 — Clifford backend kernel manifest.

Sprint: GA9.
Roadmap: docs/audit/ga_ebm_roadmap.md § GA9

Covers the GA9 v1 Python acceptance:
  - All 17 GA primitives have backend manifest entries under the
    `clifford_*` namespace.
  - x86 + apple_cpu carry reference status (the Python GA ops are the
    v1 execution path on these targets).
  - Apple GPU is planned with fp32/fp16/bf16 for the two headline ops
    (geometric_product + rotor_sandwich) and fp32 baseline for the rest.
  - NVIDIA/ROCm planned-only entries gate on Phase G/H.
  - `audit_backend_dtypes()` reports 0 unknown / 0 alias across the new
    slots (planned/gated dtypes are not used by the GA manifest).
  - `primitive_coverage.py` auto-picks up the new manifest entries via
    the existing `_manifest_for_name` wiring.

The "end-to-end @tessera.jit(target='apple_cpu') on a Cl(3,0)
rotor-sandwich" claim from the roadmap requires the C++ Apple CPU
backend to actually emit a native kernel — that's GA9 follow-on work.
For v1 we verify the manifest declares the path; the actual native
execution is verified by GA10 conformance (which already runs the GA
ops end-to-end via the Python reference).
"""

from __future__ import annotations

import pytest

from tessera.compiler import backend_manifest as bm
from tessera.compiler import primitive_coverage as pc


# ---------------------------------------------------------------------------
# Manifest registry coverage
# ---------------------------------------------------------------------------

EXPECTED_CLIFFORD_OPS = {
    # GA3 core (12)
    "clifford_geometric_product",
    "clifford_grade_projection",
    "clifford_wedge",
    "clifford_left_contraction",
    "clifford_inner",
    "clifford_reverse",
    "clifford_grade_involution",
    "clifford_conjugate",
    "clifford_norm",
    "clifford_exp",
    "clifford_log",
    "clifford_rotor_sandwich",
    # GA5 differential-form (5)
    "clifford_hodge_star",
    "clifford_ext_deriv",
    "clifford_codiff",
    "clifford_vec_deriv",
    "clifford_integral",
}

HEADLINE_OPS = {"clifford_geometric_product", "clifford_rotor_sandwich"}


def test_seventeen_clifford_ops_have_manifest_entries() -> None:
    assert set(bm._CLIFFORD_PRIMITIVES) == EXPECTED_CLIFFORD_OPS


@pytest.mark.parametrize("op_name", sorted(EXPECTED_CLIFFORD_OPS))
def test_each_clifford_op_has_five_backend_entries(op_name: str) -> None:
    """Every GA primitive ships with entries for x86 / apple_cpu /
    apple_gpu / nvidia_sm90 / rocm — 5 slots."""
    manifest = bm.clifford_manifest_for(op_name)
    targets = sorted(e.target for e in manifest)
    assert targets == ["apple_cpu", "apple_gpu", "nvidia_sm90", "rocm", "x86"]


# ---------------------------------------------------------------------------
# Per-target status checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op_name", sorted(EXPECTED_CLIFFORD_OPS))
def test_cpu_targets_carry_reference_status(op_name: str) -> None:
    """x86 and apple_cpu run the Python GA reference path — v1 execution."""
    manifest = bm.clifford_manifest_for(op_name)
    by_target = {e.target: e for e in manifest}
    assert by_target["x86"].status == "reference"
    assert by_target["apple_cpu"].status == "reference"


@pytest.mark.parametrize("op_name", sorted(EXPECTED_CLIFFORD_OPS - HEADLINE_OPS))
def test_non_headline_apple_gpu_status_is_planned(op_name: str) -> None:
    """Non-headline ops remain planned on Apple GPU — MSL coverage is
    scheduled for GA10 follow-on."""
    manifest = bm.clifford_manifest_for(op_name)
    by_target = {e.target: e for e in manifest}
    assert by_target["apple_gpu"].status == "planned"


@pytest.mark.parametrize("op_name", sorted(HEADLINE_OPS))
def test_headline_apple_gpu_status_is_fused(op_name: str) -> None:
    """As of 2026-05-17, the two headline GA ops ship native MSL kernels
    (Cl(3,0) f32) in apple_gpu_runtime.mm — see test_apple_gpu_clifford_msl.py."""
    manifest = bm.clifford_manifest_for(op_name)
    by_target = {e.target: e for e in manifest}
    assert by_target["apple_gpu"].status == "fused"
    # fp32 native kernel; fp16/bf16 ports pending.
    assert by_target["apple_gpu"].dtypes == ("fp32",)
    assert "MSL" in by_target["apple_gpu"].notes


@pytest.mark.parametrize("op_name", sorted(EXPECTED_CLIFFORD_OPS))
def test_nvidia_and_rocm_remain_planned(op_name: str) -> None:
    """NVIDIA / ROCm gated on Phase G/H."""
    manifest = bm.clifford_manifest_for(op_name)
    by_target = {e.target: e for e in manifest}
    assert by_target["nvidia_sm90"].status == "planned"
    assert by_target["rocm"].status == "planned"


# ---------------------------------------------------------------------------
# Headline-op dtype coverage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op_name", sorted(HEADLINE_OPS))
def test_headline_ops_ship_fused_msl_kernel_on_apple_gpu(op_name: str) -> None:
    """geo_product + rotor_sandwich shipped as fused MSL kernels
    (Cl(3,0) f32) on 2026-05-17 — see test_apple_gpu_clifford_msl.py
    for the bitwise GPU-vs-numpy verification."""
    manifest = bm.clifford_manifest_for(op_name)
    apple_gpu = next(e for e in manifest if e.target == "apple_gpu")
    assert apple_gpu.status == "fused"
    assert apple_gpu.dtypes == ("fp32",)
    # Feature flags name the actual stack components.
    assert "msl" in apple_gpu.feature_flags
    assert "metal" in apple_gpu.feature_flags


def test_non_headline_ops_get_apple_gpu_fp32_only() -> None:
    """Non-headline ops carry fp32 baseline on Apple GPU; MSL ports for
    fp16/bf16 are GA10/GA11 conformance follow-on."""
    for op_name in EXPECTED_CLIFFORD_OPS - HEADLINE_OPS:
        manifest = bm.clifford_manifest_for(op_name)
        apple_gpu = next(e for e in manifest if e.target == "apple_gpu")
        assert apple_gpu.dtypes == ("fp32",), (
            f"{op_name} should have fp32-only apple_gpu coverage; got {apple_gpu.dtypes}"
        )


@pytest.mark.parametrize("op_name", sorted(EXPECTED_CLIFFORD_OPS))
def test_cpu_targets_carry_fp32_fp64(op_name: str) -> None:
    """The Python GA ops accept fp32 + fp64 — manifest mirrors that."""
    manifest = bm.clifford_manifest_for(op_name)
    by_target = {e.target: e for e in manifest}
    assert by_target["x86"].dtypes == ("fp32", "fp64")
    assert by_target["apple_cpu"].dtypes == ("fp32", "fp64")


# ---------------------------------------------------------------------------
# manifest_for() dispatch routes clifford_* names to clifford_manifest_for
# ---------------------------------------------------------------------------

def test_manifest_for_dispatches_clifford_names() -> None:
    direct = bm.clifford_manifest_for("clifford_geometric_product")
    via_dispatch = bm.manifest_for("clifford_geometric_product")
    # Same entries (compare by target+status+dtypes; the BackendKernelEntry
    # dataclass is frozen so __eq__ is structural).
    assert direct == via_dispatch


def test_manifest_for_returns_empty_for_unknown_clifford_name() -> None:
    assert bm.manifest_for("clifford_does_not_exist") == []


def test_manifest_for_does_not_disturb_existing_op_lookup() -> None:
    """The clifford_ prefix dispatch must not break tensor-op manifests."""
    matmul = bm.manifest_for("matmul")
    assert matmul, "tensor matmul manifest should still exist"
    targets = {e.target for e in matmul}
    assert "x86" in targets
    assert "apple_cpu" in targets


# ---------------------------------------------------------------------------
# audit_backend_dtypes — no unknown / no alias in the new slots
# ---------------------------------------------------------------------------

def test_audit_backend_dtypes_finds_no_unknown_or_alias_in_clifford_slots() -> None:
    audit = bm.audit_backend_dtypes()
    unknown = audit.get("unknown", [])
    alias = audit.get("alias", [])
    # Filter to entries originating in clifford manifests.
    unknown_clifford = [e for e in unknown if e[0].startswith("clifford_")]
    alias_clifford = [e for e in alias if e[0].startswith("clifford_")]
    assert not unknown_clifford, f"unknown dtypes in clifford manifest: {unknown_clifford}"
    assert not alias_clifford, f"alias dtypes in clifford manifest: {alias_clifford}"


def test_audit_backend_dtypes_finds_no_planned_gated_in_clifford_slots() -> None:
    """GA9 v1 doesn't use any planned/gated dtype family (mxfp / bfp / etc.)."""
    audit = bm.audit_backend_dtypes()
    pg = audit.get("planned_gated", [])
    pg_clifford = [e for e in pg if e[0].startswith("clifford_")]
    assert not pg_clifford, f"clifford manifest must not use planned/gated dtypes: {pg_clifford}"


# ---------------------------------------------------------------------------
# primitive_coverage.py automatically picks up the new manifests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op_name", sorted(EXPECTED_CLIFFORD_OPS))
def test_primitive_coverage_attaches_manifest_to_clifford_entry(op_name: str) -> None:
    """The existing `_manifest_for_name` wiring routes clifford_* names
    through the GA9 manifest, so the GA4 coverage entries automatically
    pick up the backend matrix."""
    entries = pc.all_primitive_coverages()
    entry = entries[op_name]
    md = entry.metadata
    assert "backend_kernel_manifest" in md, (
        f"{op_name} coverage entry missing `backend_kernel_manifest` metadata"
    )
    manifest = md["backend_kernel_manifest"]
    targets = {item["target"] for item in manifest}
    assert {"x86", "apple_cpu", "apple_gpu"}.issubset(targets), (
        f"{op_name} manifest missing expected v1 CPU/GPU targets; got {targets}"
    )


# ---------------------------------------------------------------------------
# all_manifests() includes clifford entries
# ---------------------------------------------------------------------------

def test_all_manifests_includes_clifford_primitives() -> None:
    all_m = bm.all_manifests()
    for op_name in EXPECTED_CLIFFORD_OPS:
        assert op_name in all_m, f"all_manifests missing {op_name}"


def test_all_manifests_total_count_grew_by_17() -> None:
    """Baseline manifest count + 17 clifford ops = new total."""
    all_m = bm.all_manifests()
    clifford = [k for k in all_m if k.startswith("clifford_")]
    assert len(clifford) == 17


# ---------------------------------------------------------------------------
# Manifest entry shape sanity
# ---------------------------------------------------------------------------

def test_clifford_entries_have_feature_flags() -> None:
    """Every clifford manifest entry must carry the `clifford_dialect`
    feature flag so a runtime can filter for GA-capable backends."""
    for op_name in EXPECTED_CLIFFORD_OPS:
        manifest = bm.clifford_manifest_for(op_name)
        for entry in manifest:
            assert "clifford_dialect" in entry.feature_flags, (
                f"{op_name} entry for {entry.target} missing clifford_dialect flag"
            )


def test_headline_apple_gpu_entry_documents_fused_msl_kernel() -> None:
    """The notes now reference the actual exported C ABI symbol so
    consumers can grep for it directly."""
    manifest = bm.clifford_manifest_for("clifford_geometric_product")
    apple_gpu = next(e for e in manifest if e.target == "apple_gpu")
    assert "Fused MSL kernel" in apple_gpu.notes
    assert "tessera_apple_gpu_clifford_geo_product_cl30_f32" in apple_gpu.notes
    assert "apple_gpu_runtime.mm" in apple_gpu.notes


def test_cpu_entries_document_python_reference_path() -> None:
    manifest = bm.clifford_manifest_for("clifford_geometric_product")
    x86 = next(e for e in manifest if e.target == "x86")
    assert "Python GA reference" in x86.notes
    assert "GA8" in x86.notes  # references the GA8 lowering pass


def test_nvidia_entry_documents_phase_g_gating() -> None:
    manifest = bm.clifford_manifest_for("clifford_geometric_product")
    nv = next(e for e in manifest if e.target == "nvidia_sm90")
    assert "Phase G" in nv.notes


def test_rocm_entry_documents_phase_h_gating() -> None:
    manifest = bm.clifford_manifest_for("clifford_geometric_product")
    rocm = next(e for e in manifest if e.target == "rocm")
    assert "Phase H" in rocm.notes


# ---------------------------------------------------------------------------
# End-to-end: the GA Python reference path still runs (the v1 execution
# story on x86 + apple_cpu).  GA10 conformance already exercises this
# elsewhere; we re-run a tiny smoke here so a GA9 regression would
# surface in the manifest test file too.
# ---------------------------------------------------------------------------

def test_python_reference_path_executes_rotor_sandwich_on_cl30() -> None:
    """Demonstrate the v1 execution story: the Python GA reference
    runs end-to-end and matches a numpy SO(3) reference. The
    `target="apple_cpu"` JIT path would dispatch through this same
    code path until a fused MSL kernel exists (GA9 followup)."""
    import math
    import numpy as np
    from tessera.ga import (
        Cl,
        Multivector,
        geometric_product,
        reverse,
        rotor_from_axis,
        rotor_sandwich,
    )

    a = Cl(3, 0)
    bivec = Multivector.from_blade(a.blade("e12"), a, dtype=np.float64)
    R = rotor_from_axis(bivec, math.pi / 4)
    v = Multivector.from_vector([1.0, 0.0, 0.0], a, dtype=np.float64)

    rotated = rotor_sandwich(R, v)
    # SO(3) reference: rotating (1, 0, 0) by π/4 around e12 (=> z-axis-like)
    # gives (cos(π/4), sin(π/4), 0).
    cos = math.cos(math.pi / 4)
    sin = math.sin(math.pi / 4)
    e1_idx = a.blade("e1").mask
    e2_idx = a.blade("e2").mask
    e3_idx = a.blade("e3").mask
    assert rotated.coefficients[e1_idx] == pytest.approx(cos, abs=1e-7)
    assert rotated.coefficients[e2_idx] == pytest.approx(sin, abs=1e-7)
    assert abs(rotated.coefficients[e3_idx]) < 1e-12
