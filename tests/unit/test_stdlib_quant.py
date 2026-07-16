"""M1 keystone — packed quant layouts + fused dequant-into-GEMM.

Oracle strategy (mirrors the evaluator program): the vertical oracle is
full-precision ``x @ dequant(W)``; ``dequant_matmul``'s group-wise fp32
accumulation must reproduce it (a DESIL cross-path: group-accumulate ≡
dequant-then-matmul).
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import ops
from tessera.stdlib import quant as q


def _w(rng, K, N):
    return (rng.standard_normal((K, N)) / np.sqrt(K)).astype(np.float32)


# ── packing ──────────────────────────────────────────────────────────────────
def test_pack_unpack_int4_roundtrip():
    rng = np.random.default_rng(0)
    codes = rng.integers(-8, 8, size=(6, 16)).astype(np.int8)
    packed = q.pack_int4(codes)
    assert packed.dtype == np.uint8
    assert packed.shape == (6, 8)          # last axis halved
    assert np.array_equal(q.unpack_int4(packed), codes)


def test_pack_int4_rejects_odd_axis_and_range():
    with pytest.raises(ValueError):
        q.pack_int4(np.zeros((4, 5), dtype=np.int8))
    with pytest.raises(ValueError):
        q.pack_int4(np.full((4, 4), 9, dtype=np.int8))


# ── quantize / dequantize parity ──────────────────────────────────────────────
@pytest.mark.parametrize("dtype,gsz,rtol", [
    ("int8", None, 2e-2),
    ("int8", 32, 2e-2),
    ("int4", None, 0.2),
    ("int4", 32, 0.12),
    ("fp8_e4m3", 32, 0.12),
])
def test_quantize_dequantize_parity(dtype, gsz, rtol):
    rng = np.random.default_rng(1)
    w = _w(rng, 64, 48)
    fp8 = ops.quantize_fp8 if dtype.startswith("fp8") else None
    packed = q.quantize_weight(w, dtype, group_size=gsz, fp8_quantizer=fp8)
    deq = packed.dequantize()
    assert deq.shape == w.shape
    # relative Frobenius error is bounded by the format resolution.
    err = np.linalg.norm(deq - w) / np.linalg.norm(w)
    assert err < rtol, f"{dtype} g={gsz}: rel err {err:.4f}"


def test_group_size_beats_per_channel_for_int4():
    """Finer groups → strictly lower quant error (the reason group-wise exists)."""
    rng = np.random.default_rng(2)
    w = _w(rng, 128, 32)
    per_chan = q.quantize_weight(w, "int4", group_size=None).dequantize()
    grouped = q.quantize_weight(w, "int4", group_size=16).dequantize()
    assert (np.linalg.norm(grouped - w) < np.linalg.norm(per_chan - w))


def test_int4_packs_smaller_than_int8():
    rng = np.random.default_rng(3)
    w = _w(rng, 64, 32)
    b4 = q.quantize_weight(w, "int4", group_size=32).storage_bytes()
    b8 = q.quantize_weight(w, "int8", group_size=32).storage_bytes()
    assert b4 < b8


# ── scale-layout contract ─────────────────────────────────────────────────────
def test_scale_layout_contract():
    per_chan = q.QuantScheme("int4", group_size=None).scale_layout
    assert per_chan.granularity == "per_channel"
    grouped = q.QuantScheme("int4", group_size=64).scale_layout
    assert grouped.granularity == "block" and grouped.vector_size == 64


# ── fused dequant-into-GEMM == dequant-then-matmul (the oracle) ───────────────
@pytest.mark.parametrize("dtype,gsz", [
    ("int8", 32), ("int4", 32), ("int4", None), ("fp8_e4m3", 32)])
def test_dequant_matmul_equals_oracle(dtype, gsz):
    rng = np.random.default_rng(4)
    K, N, M = 64, 40, 12
    w = _w(rng, K, N)
    x = rng.standard_normal((M, K)).astype(np.float32)
    fp8 = ops.quantize_fp8 if dtype.startswith("fp8") else None
    packed = q.quantize_weight(w, dtype, group_size=gsz, fp8_quantizer=fp8)
    got = q.dequant_matmul(x, packed)
    oracle = q.reference_dequant_then_matmul(x, packed)
    # group-wise fp32 accumulation must match dequant-then-matmul to fp32 eps.
    np.testing.assert_allclose(got, oracle, rtol=1e-4, atol=1e-4)


def test_dequant_matmul_shape_guard():
    rng = np.random.default_rng(5)
    packed = q.quantize_weight(_w(rng, 32, 16), "int8", group_size=16)
    with pytest.raises(ValueError):
        q.dequant_matmul(rng.standard_normal((4, 31)).astype(np.float32), packed)


def _apple_gpu_available() -> bool:
    try:
        from tessera import _apple_gpu_backend as agb
        agb.gpu_matmul(np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32))
        return True
    except Exception:                                  # noqa: BLE001
        return False


@pytest.mark.hardware_apple_gpu
def test_dequant_matmul_apple_gpu_matches_oracle():
    """Fused dequant-GEMM runs on the Metal kernel and matches the oracle —
    the M1 'executes on Apple GPU' claim."""
    rng = np.random.default_rng(7)
    K, N, M = 64, 48, 16
    packed = q.quantize_weight(_w(rng, K, N), "int4", group_size=16)
    x = rng.standard_normal((M, K)).astype(np.float32)
    got = q.dequant_matmul(x, packed, backend="apple_gpu")
    oracle = q.reference_dequant_then_matmul(x, packed)
    np.testing.assert_allclose(got, oracle, rtol=1e-3, atol=1e-3)


@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize("dtype,gsz", [("int4", 16), ("int8", 32), ("fp8_e4m3", 32)])
def test_fused_dequant_kernel_in_register(dtype, gsz):
    """The M1.1 fused dequant-into-GEMM MSL kernel (in-register dequant from unit
    codes + per-group scales, fp32 accumulate) matches dequant-then-matmul. The
    full fp32 weight is never materialized — only codes + scales reach the GPU."""
    from tessera import _apple_gpu_backend as agb
    rng = np.random.default_rng(8)
    K, N, M = 64, 40, 12
    fp8 = ops.quantize_fp8 if dtype.startswith("fp8") else None
    packed = q.quantize_weight(_w(rng, K, N), dtype, group_size=gsz, fp8_quantizer=fp8)
    codes, scales, g = q.unit_codes_and_scales(packed)
    assert codes.shape == (K, N) and scales.shape == (K // gsz, N)
    x = rng.standard_normal((M, K)).astype(np.float32)
    got = agb.gpu_dequant_matmul(x, codes, scales, g)
    oracle = q.reference_dequant_then_matmul(x, packed)
    np.testing.assert_allclose(got, oracle, rtol=1e-3, atol=1e-3)


def test_dequant_grouped_gemm_matches_per_expert():
    rng = np.random.default_rng(6)
    K, N, E = 48, 24, 3
    experts = [q.quantize_weight(_w(rng, K, N), "int4", group_size=16) for _ in range(E)]
    group_sizes = np.array([4, 0, 5], dtype=np.int64)
    T = int(group_sizes.sum())
    x = rng.standard_normal((T, K)).astype(np.float32)
    got = q.dequant_grouped_gemm(x, experts, group_sizes)
    # per-expert oracle
    oracle = np.zeros((T, N), dtype=np.float64)
    off = 0
    for e in range(E):
        n = int(group_sizes[e])
        if n:
            oracle[off:off + n] = q.reference_dequant_then_matmul(x[off:off + n], experts[e])
        off += n
    np.testing.assert_allclose(got, oracle, rtol=1e-4, atol=1e-4)
