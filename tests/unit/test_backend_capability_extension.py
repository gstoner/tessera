"""Arch-3 (2026-05-22) — backend capability matrix extension drift gate.

Pins the new schema fields + hardware_verified contract:

  * BackendKernelEntry accepts shape_envelope / runtime_symbol /
    lit_fixture / execute_compare_fixture / benchmark_json (all
    optional, all None by default — preserves backward compatibility).
  * status="hardware_verified" requires runtime_symbol AND
    execute_compare_fixture (validator catches under-evidenced claims).
  * primitive_is_complete() computes registry's backend_kernel="complete"
    from the full target row set.
  * hardware_verified is earned per (op, target) once a real on-silicon
    kernel + checked-in numerical fixture land: Apple GPU encode-session
    ops (Project 3 / Sprint A, 2026-06-01), Strix Halo ROCm WMMA ops
    (2026-06-22), and sm_120 NVIDIA mma.sync ops on the RTX 5070 Ti
    (2026-06-24). Each new on-silicon proof updates this guard; any other
    target/op claiming it is under-evidenced and fails.
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


# Project 3 / Sprint A (2026-06-01) — the Apple GPU encode-session ops
# that are legitimately ``hardware_verified``: each runs a real Metal /
# MPSGraph kernel on this Mac's GPU AND ships a checked-in
# ``execute_compare_fixture`` that numerically validates it against
# numpy. Apple Silicon GPU IS real hardware (unlike the discrete
# NVIDIA / ROCm accelerators, which are genuinely
# unavailable here), so these claims are honest. This is the
# "first proof landed → update the test" event the original guard's
# docstring anticipated.
_APPLE_GPU_HARDWARE_VERIFIED_OPS = frozenset({
    "softmax", "softmax_safe", "gelu", "rope", "flash_attn",
    "rmsnorm", "layer_norm", "silu", "bmm", "conv2d",
    "quantized_matmul",
})

# Strix Halo bring-up — the ROCm ops whose ``hardware_verified`` claim is
# honestly earned: each ships a real C-ABI ``runtime_symbol`` that runs an RDNA
# WMMA kernel on the AMD GPU AND a checked-in ``execute_compare_fixture`` that
# numerically validates it (and skips clean — no false pass — on a host without
# an AMD GPU). The fixture only *executes* on a ROCm box; the manifest claim
# rests on that on-silicon run (see docs/audit/backend/rocm/STRIX_HALO_EXECUTION_PLAN.md).
#   - matmul  (2026-06-22): libtessera_rocm_gemm.so WMMA GEMM.
#   - flash_attn (2026-06-23): libtessera_rocm_flash_attn.so FA-2 forward — the
#     second op after matmul to execute natively on a non-Apple backend.
_ROCM_HARDWARE_VERIFIED_OPS = frozenset({"matmul", "flash_attn"})

# NVIDIA sm_120 bring-up — the consumer-Blackwell ops whose ``hardware_verified``
# claim is honestly earned, mirroring the ROCm block: each ships a real C-ABI
# ``runtime_symbol`` that NVRTC-compiles a warp-level mma.sync kernel and runs it
# on the RTX 5070 Ti (sm_120, CC 12.0, CUDA 13.3) AND a checked-in
# ``execute_compare_fixture`` that numerically validates it (and skips clean — no
# false pass — on a host without an NVIDIA GPU / NVRTC). The fixture only
# *executes* on the sm_120 box; the manifest claim rests on that on-silicon run.
#   - matmul (2026-06-24): libtessera_nvidia_gemm.so warp-level mma.sync GEMM
#     (tessera_nvidia_mma_gemm_{bf16,f16,tf32,e4m3,e5m2}) — the first NVIDIA op to
#     execute natively on silicon here.
_NVIDIA_HARDWARE_VERIFIED_OPS = frozenset({"matmul"})


def _hardware_verified_claim_is_allowed(op: str, e: BackendKernelEntry) -> bool:
    """A hardware_verified row is honest only with both evidence fields AND on a
    target whose proof has actually landed: Apple GPU encode-session ops, the
    Strix Halo ROCm WMMA ops, or the sm_120 NVIDIA mma.sync ops. Anything else
    (other targets, unexpected ops) is under-evidenced and must fail the guard."""
    if not e.runtime_symbol or not e.execute_compare_fixture:
        return False
    if e.target == "apple_gpu":
        return op in _APPLE_GPU_HARDWARE_VERIFIED_OPS
    if e.target == "nvidia_sm120":
        return op in _NVIDIA_HARDWARE_VERIFIED_OPS
    if e.target == "rocm":
        return op in _ROCM_HARDWARE_VERIFIED_OPS
    return False


def test_only_apple_gpu_claims_hardware_verified_today() -> None:
    """Every ``hardware_verified`` manifest entry must be a target whose proof
    has actually landed AND carry both evidence fields:

    * Apple GPU encode-session ops — real Metal/MPSGraph kernel on this Mac's
      GPU + checked-in execute_compare_fixture (Project 3 / Sprint A, 2026-06-01).
    * ROCm WMMA ops — real RDNA WMMA kernel on an AMD GPU (Strix Halo bring-up,
      2026-06-22) + skip-clean execute_compare_fixture.
    * NVIDIA sm_120 mma.sync ops — real warp-level mma.sync kernel on an RTX
      5070 Ti (consumer Blackwell, CC 12.0, CUDA 13.3; 2026-06-24) + skip-clean
      execute_compare_fixture.

    Any other target/op is under-evidenced and must fail this guard — the
    Phase G/H frontier audit tracks the remaining gaps."""
    by_op = all_manifests()
    hw_verified: list[tuple[str, BackendKernelEntry]] = [
        (op, entry)
        for op, entries in by_op.items()
        for entry in entries
        if entry.status == "hardware_verified"
    ]
    offenders = [
        (op, e) for op, e in hw_verified
        if not _hardware_verified_claim_is_allowed(op, e)
    ]
    assert not offenders, (
        "Unexpected hardware_verified claim — only Apple GPU encode-session/"
        "packed-quant ops (real Metal kernel), Strix Halo ROCm WMMA ops "
        "(real AMD-GPU kernel), and "
        "sm_120 NVIDIA mma.sync ops (real RTX 5070 Ti kernel), each with a "
        "checked-in execute_compare_fixture, may claim it. Offenders: "
        + ", ".join(
            f"{op}/{e.target} via {e.runtime_symbol}"
            for op, e in offenders
        )
    )
    # And the Apple GPU promotion must be complete — every expected op
    # that ``all_manifests()`` actually surfaces must be
    # hardware_verified (catches an accidental partial revert). Ops not
    # in ``OP_SPECS`` (e.g. ``bmm``, an Apple-GPU-only encode op) don't
    # appear in ``all_manifests()``; they're covered by the dedicated
    # ``test_apple_gpu_hardware_verified_promotion`` lock instead.
    apple_hw_ops = {
        op for op, e in hw_verified if e.target == "apple_gpu"}
    coverable = _APPLE_GPU_HARDWARE_VERIFIED_OPS & set(by_op)
    missing = coverable - apple_hw_ops
    assert not missing, (
        f"Expected Apple GPU hardware_verified ops missing from the "
        f"manifest: {sorted(missing)}")
