"""Workstream H (backend consumer of the transpose contract) — MatmulRegion
orientation.

The plain-GEMM region resolves its operand orientation from the matmul's
**transpose flags** (`transposeA`/`transposeB`, produced by the
`TransposeIntoMatmul` Graph-IR fold), not from value shapes — closing the
OPTIMIZING_COMPILER_PLAN §6 orientation note ("ambiguous when M==K==N") for plain
GEMM the same way `q_transposed`/`k_transposed` did for attention (M2). A backend
GEMM candidate must feed the kernel the *oriented* operands (via
`MatmulRegion._natural`), so a transposed raw operand is never silently mis-fed.
"""
from __future__ import annotations

import numpy as np

from tessera.compiler import fusion as F


def _round(x, dtype="bfloat16"):
    from tessera.compiler.fusion_core import _round_to_storage
    return _round_to_storage(np.asarray(x, np.float32), dtype)


# ── the region reference honors the transpose flags ──────────────────────────

def test_default_region_is_plain_matmul():
    # Default flags False → reference is exactly A @ B (back-compat: unchanged).
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 8)).astype(np.float32)
    B = rng.standard_normal((8, 5)).astype(np.float32)
    got = F.MatmulRegion().reference(A, B)
    want = (_round(A) @ _round(B)).astype(np.float32)
    np.testing.assert_allclose(got, want, rtol=0, atol=0)


def test_transpose_a_reference():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((8, 4)).astype(np.float32)   # raw Aᵀ; natural is (4,8)
    B = rng.standard_normal((8, 5)).astype(np.float32)
    got = F.MatmulRegion(transpose_a=True).reference(A, B)
    want = (_round(A.T) @ _round(B)).astype(np.float32)
    np.testing.assert_allclose(got, want, rtol=0, atol=0)


def test_transpose_b_reference():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((4, 8)).astype(np.float32)
    B = rng.standard_normal((5, 8)).astype(np.float32)   # raw Bᵀ; natural is (8,5)
    got = F.MatmulRegion(transpose_b=True).reference(A, B)
    want = (_round(A) @ _round(B.T)).astype(np.float32)
    np.testing.assert_allclose(got, want, rtol=0, atol=0)


def test_transpose_both_reference():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((8, 4)).astype(np.float32)
    B = rng.standard_normal((5, 8)).astype(np.float32)
    got = F.MatmulRegion(transpose_a=True, transpose_b=True).reference(A, B)
    want = (_round(A.T) @ _round(B.T)).astype(np.float32)
    np.testing.assert_allclose(got, want, rtol=0, atol=0)


def test_ambiguous_square_resolved_by_flag_not_shape():
    # The centerpiece: with M==K==N a transposed operand has the SAME shape as a
    # natural one, so value shapes can't tell them apart — only the flag can. The
    # transposed reference must differ from the plain one (else the transpose was
    # silently dropped, the exact §6 bug).
    rng = np.random.default_rng(4)
    A = rng.standard_normal((16, 16)).astype(np.float32)
    B = rng.standard_normal((16, 16)).astype(np.float32)
    plain = F.MatmulRegion().reference(A, B)
    ta = F.MatmulRegion(transpose_a=True).reference(A, B)
    tb = F.MatmulRegion(transpose_b=True).reference(A, B)
    assert not np.allclose(plain, ta)         # transpose_a actually flipped A
    assert not np.allclose(plain, tb)
    np.testing.assert_allclose(ta, (_round(A.T) @ _round(B)).astype(np.float32),
                               rtol=0, atol=0)
    np.testing.assert_allclose(tb, (_round(A) @ _round(B.T)).astype(np.float32),
                               rtol=0, atol=0)


# ── _natural: dtype-preserving path for the GPU candidate ────────────────────

def test_natural_no_cast_preserves_storage_dtype():
    A = np.ones((8, 4), np.float16)
    B = np.ones((8, 5), np.float16)
    An, Bn = F.MatmulRegion(transpose_a=True)._natural(A, B, cast=False)
    assert An.dtype == np.float16 and Bn.dtype == np.float16
    assert An.shape == (4, 8)                 # Aᵀ flipped, dtype kept


def test_natural_default_casts_to_f32():
    A = np.ones((4, 8), np.float16)
    B = np.ones((8, 5), np.float16)
    An, Bn = F.MatmulRegion()._natural(A, B)
    assert An.dtype == np.float32 and Bn.dtype == np.float32


# ── a candidate that ignores the flag is wrong (contract must be honored) ─────

def test_candidate_must_orient_to_match_reference():
    # Simulate a GEMM kernel that consumes natural operands. The correct candidate
    # orients via _natural first; a candidate that feeds the RAW operands produces
    # a different (wrong) result — proving the emit lane must honor the contract.
    rng = np.random.default_rng(5)
    region = F.MatmulRegion(dtype="float16", transpose_b=True)
    A = rng.standard_normal((16, 16)).astype(np.float32)
    B = rng.standard_normal((16, 16)).astype(np.float32)   # raw Bᵀ

    def kernel(a, b):                          # a natural-operand GEMM
        return (np.asarray(a, np.float32) @ np.asarray(b, np.float32))

    ref = region.reference(A, B)
    An, Bn = region._natural(A, B, cast=False)
    oriented = kernel(An, Bn)                   # honors the contract
    raw = kernel(A, B)                          # ignores the flag
    np.testing.assert_allclose(oriented, ref, rtol=1e-2, atol=1e-2)
    assert not np.allclose(raw, ref, rtol=1e-2, atol=1e-2)


# ── the F4 verifier probe is generated in RAW orientation (PR #310 review) ────

def test_verifier_probe_is_raw_orientation_for_transposed_region():
    # verify_synthesized_matmul must feed the candidate the region's RAW operands:
    # a transposed region's candidate flips them back via _natural, so a natural
    # (K,N) probe would be double-flipped into a shape mismatch and crash before
    # selection. Prove the probe respects the flags (raw shapes) and does not crash.
    from tessera.compiler import fusion_core as FC

    seen: dict[str, tuple] = {}

    class _Runner:
        target = "fake_tp"
        accuracy_atol = None

        def run_matmul(self, region, A, B):
            seen["A"], seen["B"] = A.shape, B.shape
            # A real kernel consuming the oriented operands — matches the reference.
            return region.reference(A, B), "fake_real"

    # transpose_b: raw B is (N,K)=(16,32), not the natural (K,N)=(32,16).
    assert FC.verify_synthesized_matmul(
        FC.MatmulRegion(transpose_b=True), runner=_Runner(), force=True) is True
    assert seen["B"] == (16, 32)

    # transpose_a: raw A is (K,M)=(32,32).
    seen.clear()
    assert FC.verify_synthesized_matmul(
        FC.MatmulRegion(transpose_a=True), runner=_Runner(), force=True) is True
    assert seen["A"] == (32, 32)

    # natural: probe stays (M,K)@(K,N).
    seen.clear()
    assert FC.verify_synthesized_matmul(
        FC.MatmulRegion(), runner=_Runner(), force=True) is True
    assert seen["A"] == (32, 32) and seen["B"] == (32, 16)
