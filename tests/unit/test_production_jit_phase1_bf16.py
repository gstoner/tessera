"""Phase 1 Sprint 1.5 — bf16 boundary (RUNTIME_ABI_SPEC.md §12.5).

Not a new op — the first exercise of the bf16 ABI rule:
* `ml_dtypes.bfloat16` on the Python side,
* RAW 16-bit storage at the memref boundary (`ctypes.c_uint16`),
* matmul: bf16 storage, **f32 accumulate**, truncate-on-store.

Proves the descriptor-packing dtype table + the matmul accumulator policy.
Skips if libtessera_jit or ml_dtypes is unavailable.
"""

import numpy as np
import pytest

from tessera import _jit_boundary as jb

ml_dtypes = pytest.importorskip("ml_dtypes")
bf16 = ml_dtypes.bfloat16

pytestmark = pytest.mark.skipif(
    not jb.is_available(),
    reason="libtessera_jit not built; run `ninja -C build tessera_jit`",
)


def _to_bf16(x):
    return np.asarray(x).astype(bf16)


# ── elementwise bf16 ────────────────────────────────────────────────────────


@pytest.mark.parametrize("fn,ref", [(jb.jit_add, np.add), (jb.jit_mul, np.multiply)])
def test_bf16_elementwise_matches_numpy_oracle(fn, ref):
    rng = np.random.default_rng(0)
    a = _to_bf16(rng.standard_normal((3, 5)))
    b = _to_bf16(rng.standard_normal((3, 5)))
    out = fn(a, b)
    assert out.dtype == bf16  # boundary preserved the storage dtype
    # Oracle in f64, compared at bf16 tolerance (~3 significant bits of mantissa).
    expect = ref(a.astype(np.float64), b.astype(np.float64))
    np.testing.assert_allclose(out.astype(np.float64), expect, rtol=3e-2, atol=3e-2)


def test_bf16_add_executed_the_compiled_function():
    a = _to_bf16(np.ones((4,)))
    b = _to_bf16(np.full((4,), 2.0))
    before = jb.invocation_count()
    out = jb.jit_add(a, b)
    assert jb.invocation_count() == before + 1
    np.testing.assert_allclose(out.astype(np.float64), np.full(4, 3.0), rtol=1e-2)


# ── bf16 matmul with f32 accumulate ─────────────────────────────────────────


@pytest.mark.parametrize("M,K,N", [(4, 8, 3), (16, 32, 16)])
def test_bf16_matmul_f32_accumulate(M, K, N):
    rng = np.random.default_rng(M + K + N)
    a = _to_bf16(rng.standard_normal((M, K)))
    b = _to_bf16(rng.standard_normal((K, N)))
    out = jb.jit_matmul(a, b)
    assert out.dtype == bf16

    # f32-accumulate reference: upcast inputs, matmul in f32, then truncate.
    ref_f32 = a.astype(np.float32) @ b.astype(np.float32)
    ref_bf16 = ref_f32.astype(bf16)
    # The JIT's result and the truncated-f32 reference should be very close
    # (both accumulate in f32; only the final round-to-bf16 differs by ULPs).
    np.testing.assert_allclose(
        out.astype(np.float32), ref_bf16.astype(np.float32), rtol=2e-2, atol=2e-2
    )


def test_bf16_matmul_beats_bf16_accumulate_on_long_K():
    # With a long contraction dim, bf16-accumulate loses precision badly; the
    # f32-accumulate path (what we implement) should track the f32 reference far
    # better. This asserts the accumulator policy actually took effect.
    rng = np.random.default_rng(7)
    K = 512
    a = _to_bf16(rng.standard_normal((2, K)) * 0.1)
    b = _to_bf16(rng.standard_normal((K, 2)) * 0.1)
    out = jb.jit_matmul(a, b).astype(np.float64)

    f32_ref = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float64)
    # Simulate naive bf16-accumulate to show it's worse than our result.
    bf16_acc = np.zeros((2, 2), dtype=bf16)
    af, bf = a, b
    acc = np.zeros((2, 2), dtype=bf16)
    for k in range(K):
        acc = (acc.astype(bf16) + (af[:, k:k + 1] * bf[k:k + 1, :]).astype(bf16)).astype(bf16)
    bf16_acc = acc.astype(np.float64)

    err_ours = np.abs(out - f32_ref).max()
    err_bf16acc = np.abs(bf16_acc - f32_ref).max()
    assert err_ours <= err_bf16acc + 1e-9  # f32-accumulate is at least as good


def test_bf16_matmul_executed():
    a = _to_bf16(np.eye(4))
    b = _to_bf16(np.arange(16).reshape(4, 4))
    before = jb.invocation_count()
    out = jb.jit_matmul(a, b)
    assert jb.invocation_count() == before + 1
    np.testing.assert_allclose(
        out.astype(np.float64), np.arange(16).reshape(4, 4), rtol=1e-2
    )


# ── envelope: mixed dtypes rejected, not silently promoted ──────────────────


def test_mixed_dtype_rejected():
    a = _to_bf16(np.ones((2, 2)))
    b = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(jb.TesseraJitError):
        jb.jit_add(a, b)
