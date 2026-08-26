"""NUMPOL-CARRIER-1 (queue row 3b) — `math_mode` gets a consumer on the NVIDIA
lane, and the silent TF32 substitution stops being silent.

Measured on this tree before the change: `_NVIDIA_GEMM_SYMBOLS` mapped
``"float32" -> tessera_nvidia_mma_gemm_tf32`` and the dispatch took it
UNCONDITIONALLY, with a comment citing Decision #15a as the reason. #15a says
the opposite — "TF32 is not a storage dtype. Model as ``math_mode='tf32'`` on
fp32 via numeric_policy" — precisely so the reduced arithmetic is a choice the
program makes. The storage dtype was making it instead.

The cost, measured against an fp64 reference on 64xKx64 GEMMs (median relative
error):

    K=128    fp32 1.64e-07   ->  tf32 2.93e-04   (1783x)
    K=1024   fp32 3.02e-07   ->  tf32 3.01e-04   ( 998x)
    K=4096   fp32 3.63e-07   ->  tf32 2.91e-04   ( 800x)

TF32 keeps 11 significand bits against fp32's 24 and rounds the OPERANDS, so
no accumulator width recovers it. A program that asked for fp32 got tf32
numbers and no diagnostic.

**Why these tests can exist at all.** This box has no CUDA, and per the
claim-integrity rule a device claim needs the device. So the selection was
extracted into a pure function, `_nvidia_gemm_selection`, whose contract is
host-testable. What is proven here is the SELECTION CONTRACT. What is not
proven here — and is recorded as owed to the NR2 Pro box — is that the
selected kernel executes and produces those numbers on an RTX 5070 Ti.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import runtime as rt


def _f32():
    return np.dtype("float32")


def test_explicit_tf32_selects_the_tf32_kernel():
    sym, store = rt._nvidia_gemm_selection(_f32(), _f32(), "tf32")
    assert sym.endswith("_tf32")
    assert store is np.float32


def test_absent_math_mode_keeps_todays_behaviour():
    """Deliberate, and stated rather than assumed.

    Decision #21a says a semantic key must not silently default, and by that
    reading an absent math_mode on fp32 should fail closed rather than pick
    TF32. It does not fail closed here, because changing the default would
    alter every existing fp32 NVIDIA program from a host that cannot execute
    one — the exact shape of claim the fleet rule forbids. The behaviour is
    pinned here so the decision is visible and the NR2 Pro follow-up has a
    fixed baseline to change against.
    """
    sym, _ = rt._nvidia_gemm_selection(_f32(), _f32(), None)
    assert sym.endswith("_tf32")


def test_declaring_ieee_is_refused_rather_than_silently_given_tf32():
    """The case that previously lied. There is no IEEE-fp32 tensor-core
    instruction on any NVIDIA part, so the honest answer is a diagnostic."""
    with pytest.raises(ValueError, match="NVIDIA_MATH_MODE_UNAVAILABLE"):
        rt._nvidia_gemm_selection(_f32(), _f32(), "ieee")


def test_an_unprovided_mode_is_refused_too():
    with pytest.raises(ValueError, match="NVIDIA_MATH_MODE_UNAVAILABLE"):
        rt._nvidia_gemm_selection(_f32(), _f32(), "bf16x3")


def test_math_mode_does_not_disturb_the_narrow_storage_paths():
    """f16/bf16/fp8 storage select their own kernels; math_mode is an fp32
    concept and must not leak into them."""
    for mode in (None, "tf32", "ieee", "bf16x3"):
        sym, store = rt._nvidia_gemm_selection(
            np.dtype("float16"), np.dtype("float16"), mode)
        assert sym.endswith("_f16") and store is np.float16


def test_unsupported_storage_still_reports_the_storage_error():
    with pytest.raises(ValueError, match="nvidia_mma executor handles"):
        rt._nvidia_gemm_selection(np.dtype("int32"), np.dtype("int32"), None)


def test_the_declared_policy_is_read_from_the_op():
    """The dispatch reads math_mode off the op, from either spelling."""
    assert rt._nvidia_math_mode(
        {"kwargs": {"numeric_policy": {"math_mode": "tf32"}}}) == "tf32"
    assert rt._nvidia_math_mode({"kwargs": {"math_mode": "ieee"}}) == "ieee"
    assert rt._nvidia_math_mode({"kwargs": {}}) is None
    assert rt._nvidia_math_mode(None) is None


def test_the_substitution_this_refusal_prevents_is_worth_preventing():
    """A control on the MOTIVATION, not the code: if TF32 were close enough to
    fp32 that nobody could tell, refusing would be pedantry. It is not."""
    def to_tf32(x):
        u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
        low = u & np.uint64(0x1FFF)
        half = np.uint64(0x1000)
        up = (low > half) | ((low == half) &
                             (((u >> np.uint64(13)) & np.uint64(1)) == 1))
        u = u + np.where(up, np.uint64(0x2000), np.uint64(0))
        return (u & np.uint64(0xFFFFE000)).astype(np.uint32).view(np.float32)

    K = 1024
    rs = np.random.RandomState(11)
    A = (rs.randn(64, K) / np.sqrt(K)).astype(np.float32)
    B = (rs.randn(K, 64) / np.sqrt(K)).astype(np.float32)
    ref = np.float64(A) @ np.float64(B)
    err_fp32 = np.median(np.abs((A @ B).astype(np.float64) - ref)
                         / (np.abs(ref) + 1e-30))
    At, Bt = to_tf32(A), to_tf32(B)
    err_tf32 = np.median(np.abs(At.astype(np.float64) @ Bt.astype(np.float64)
                                - ref) / (np.abs(ref) + 1e-30))
    assert err_tf32 > 100 * err_fp32, (err_fp32, err_tf32)
