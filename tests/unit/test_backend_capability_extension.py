"""Arch-3 (2026-05-22) — backend capability matrix extension drift gate.

Pins the new schema fields + hardware_verified contract:

  * BackendKernelEntry accepts shape_envelope / runtime_symbol /
    lit_fixture / execute_compare_fixture / benchmark_json (all
    optional, all None by default — preserves backward compatibility).
  * status="hardware_verified" requires runtime_symbol AND
    execute_compare_fixture (validator catches under-evidenced claims).
  * primitive_is_complete() computes registry's backend_kernel="complete"
    from the full target row set.
  * No entry today claims hardware_verified — flipping the first one
    requires real GPU hardware proof.  This is the honest baseline; a
    Phase G / H / I sprint lighting up the first NVIDIA / ROCm /
    Metalium proof updates this expectation.
"""

from __future__ import annotations

import pytest

from tessera.compiler.backend_manifest import (
    BackendKernelEntry,
    all_manifests,
    primitive_is_complete,
)


# ─────────────────────────────────────────────────────────────────────────
# Schema: the new fields exist and default to None
# ─────────────────────────────────────────────────────────────────────────


def test_new_fields_default_none() -> None:
    """Arch-3 fields default to None so existing callers don't break."""
    entry = BackendKernelEntry(target="apple_gpu", status="fused")
    assert entry.shape_envelope is None
    assert entry.runtime_symbol is None
    assert entry.lit_fixture is None
    assert entry.execute_compare_fixture is None
    assert entry.benchmark_json is None
    assert entry.is_hardware_verified is False


def test_new_fields_accept_strings() -> None:
    entry = BackendKernelEntry(
        target="apple_gpu",
        status="fused",
        runtime_symbol="tessera_apple_gpu_matmul_softmax_matmul_f32",
        shape_envelope="M*N*K <= 2**24",
        lit_fixture="tests/tessera-ir/phase8/apple_gpu_lowering.mlir",
        execute_compare_fixture="tests/unit/test_apple_gpu_mla_e2e.py",
        benchmark_json="benchmarks/apple_gpu/benchmark_fusion.json",
    )
    assert entry.runtime_symbol == "tessera_apple_gpu_matmul_softmax_matmul_f32"
    assert entry.shape_envelope == "M*N*K <= 2**24"
    assert entry.lit_fixture.endswith(".mlir")
    assert entry.execute_compare_fixture.endswith(".py")
    assert entry.benchmark_json.endswith(".json")


def test_as_dict_emits_new_fields_when_set() -> None:
    entry = BackendKernelEntry(
        target="apple_gpu",
        status="fused",
        runtime_symbol="tessera_apple_gpu_matmul_f32",
        shape_envelope="rank=2",
    )
    d = entry.as_dict()
    assert d["runtime_symbol"] == "tessera_apple_gpu_matmul_f32"
    assert d["shape_envelope"] == "rank=2"
    # Unset fields are NOT in the dict (keeps JSON compact).
    assert "execute_compare_fixture" not in d
    assert "benchmark_json" not in d


def test_as_dict_omits_new_fields_when_unset() -> None:
    """Backward-compat: a pre-Arch-3 entry's as_dict() output shouldn't
    grow new keys until those fields are actually set."""
    entry = BackendKernelEntry(target="apple_gpu", status="fused")
    d = entry.as_dict()
    for key in (
        "shape_envelope", "runtime_symbol", "lit_fixture",
        "execute_compare_fixture", "benchmark_json",
    ):
        assert key not in d, f"{key} leaked into dict when unset"


# ─────────────────────────────────────────────────────────────────────────
# hardware_verified status: the new top rung of the ladder
# ─────────────────────────────────────────────────────────────────────────


def test_hardware_verified_status_accepted() -> None:
    """The new status is in _VALID_STATUSES — entries that meet the
    contract can use it."""
    entry = BackendKernelEntry(
        target="apple_gpu",
        status="hardware_verified",
        dtypes=("fp32",),
        runtime_symbol="tessera_apple_gpu_matmul_softmax_matmul_f32",
        execute_compare_fixture="tests/unit/test_apple_gpu_mla_e2e.py",
    )
    assert entry.status == "hardware_verified"
    assert entry.is_hardware_verified is True


def test_hardware_verified_requires_runtime_symbol() -> None:
    """The contract: no runtime symbol = no claim to hardware proof."""
    with pytest.raises(ValueError, match="requires runtime_symbol"):
        BackendKernelEntry(
            target="apple_gpu",
            status="hardware_verified",
            execute_compare_fixture="tests/unit/test_x.py",
        )


def test_hardware_verified_requires_execute_compare_fixture() -> None:
    """The contract: no test fixture = no proof = no claim."""
    with pytest.raises(ValueError, match="requires.*execute_compare_fixture"):
        BackendKernelEntry(
            target="apple_gpu",
            status="hardware_verified",
            runtime_symbol="tessera_apple_gpu_matmul_f32",
        )


def test_unknown_status_still_rejected() -> None:
    """Adding hardware_verified didn't accidentally loosen status validation."""
    with pytest.raises(ValueError, match="status must be one of"):
        BackendKernelEntry(target="apple_gpu", status="totally_fake_status")


# ─────────────────────────────────────────────────────────────────────────
# primitive_is_complete() — computed backend_kernel = "complete"
# ─────────────────────────────────────────────────────────────────────────


def test_primitive_is_complete_requires_every_target_verified() -> None:
    """All declared targets must be hardware_verified for a primitive
    to qualify as backend_kernel = complete."""
    apple = BackendKernelEntry(
        target="apple_gpu",
        status="hardware_verified",
        runtime_symbol="sym_a",
        execute_compare_fixture="tests/unit/test_a.py",
    )
    nvidia_partial = BackendKernelEntry(
        target="nvidia_sm90",
        status="planned",
    )
    # Single fully-verified target → complete.
    assert primitive_is_complete((apple,)) is True
    # One target still planned → not complete.
    assert primitive_is_complete((apple, nvidia_partial)) is False
    # Empty target set → not complete (vacuous).
    assert primitive_is_complete(()) is False


def test_primitive_is_complete_rejects_fused_only() -> None:
    """'fused' is the second-highest rung but not hardware-verified —
    a fused kernel without an execute_compare proof doesn't qualify."""
    fused_only = BackendKernelEntry(target="apple_gpu", status="fused")
    assert primitive_is_complete((fused_only,)) is False


# ─────────────────────────────────────────────────────────────────────────
# Honest baseline: nothing claims hardware_verified yet
# ─────────────────────────────────────────────────────────────────────────


def test_no_entry_claims_hardware_verified_today() -> None:
    """As of 2026-05-22 zero backend manifest entries claim
    hardware_verified — by registry design this requires real
    GPU hardware proof that isn't available on this Mac.  The Phase
    G / H / I frontier audit doc tracks the gap honestly.

    When the first NVIDIA / ROCm / Metalium proof lands (and an
    execute_compare_fixture gets checked in), update this test to
    expect that one entry."""
    by_op = all_manifests()
    hw_verified: list[tuple[str, BackendKernelEntry]] = [
        (op, entry)
        for op, entries in by_op.items()
        for entry in entries
        if entry.status == "hardware_verified"
    ]
    assert len(hw_verified) == 0, (
        "Unexpected hardware_verified claim — every such claim requires "
        "a checked-in execute_compare_fixture demonstrating numerical "
        "correctness on real hardware.  Found: "
        + ", ".join(
            f"{op}/{e.target} via {e.runtime_symbol}"
            for op, e in hw_verified
        )
    )
