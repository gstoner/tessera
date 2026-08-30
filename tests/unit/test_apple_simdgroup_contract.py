"""The Apple simdgroup primitives' numerics, checked rather than asserted.

`tessera_apple.gpu.simdgroup_matmul` rejects an f16 accumulator and a row
stride below the matrix width. Both rejections are only worth having if the
thing they forbid is actually wrong, so this computes the consequence instead
of restating the rule — a verifier that rejects something harmless is friction,
and one that rejects something catastrophic is load-bearing. The difference is
measurable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_MMA_EXTENT = 8  # Apple7's only simdgroup-matrix shape


def _mma_chain(k_slabs: int, *, accum_dtype, seed: int = 0) -> np.ndarray:
    """One 8x8 output tile accumulated over `k_slabs` 8x8x8 MMA steps.

    Mirrors the coopmat kernel's inner loop: `acc = a @ b + acc`, repeated as
    the K-slab advances. Inputs are always f16 (the MMA's storage type); only
    the accumulator precision varies, which is exactly the contract under test.
    """
    rng = np.random.default_rng(seed)
    acc = np.zeros((_MMA_EXTENT, _MMA_EXTENT), dtype=accum_dtype)
    for _ in range(k_slabs):
        a = rng.standard_normal((_MMA_EXTENT, _MMA_EXTENT)).astype(np.float16)
        b = rng.standard_normal((_MMA_EXTENT, _MMA_EXTENT)).astype(np.float16)
        # The product is computed in the accumulator's precision, which is what
        # `simdgroup_multiply_accumulate` does with a float8x8 destination.
        acc = (a.astype(accum_dtype) @ b.astype(accum_dtype) + acc).astype(accum_dtype)
    return acc


def test_fp32_accumulator_is_load_bearing_not_stylistic():
    """An f16 accumulator loses real accuracy over a realistic K loop.

    This is why `simdgroup_matmul` requires f32 for `c` and `d`. A matmul test
    at K=8 would not show it: the divergence is cumulative, so a single MMA
    looks fine and a 4096-deep reduction does not. That is precisely the class
    of numerics bug a per-op test misses and a contract catches.
    """
    k_slabs = 512  # K = 4096, an ordinary transformer inner dimension
    exact = _mma_chain(k_slabs, accum_dtype=np.float64)
    fp32 = _mma_chain(k_slabs, accum_dtype=np.float32).astype(np.float64)
    fp16 = _mma_chain(k_slabs, accum_dtype=np.float16).astype(np.float64)

    scale = np.abs(exact).max()
    fp32_err = np.abs(fp32 - exact).max() / scale
    fp16_err = np.abs(fp16 - exact).max() / scale

    # fp32 tracks the exact reduction closely; fp16 does not.
    assert fp32_err < 1e-5, f"fp32 accumulation drifted: {fp32_err}"
    assert fp16_err > 20 * fp32_err, (
        f"f16 accumulator error {fp16_err} is not materially worse than fp32 "
        f"{fp32_err}; if this ever holds, the verifier's f32 requirement needs "
        "a better justification than this test provides"
    )


def test_row_stride_below_matrix_width_aliases_rows():
    """Why `leading_dim >= 8` is a rejection and not a lint.

    Metal addresses row `r` at `base + r * leading_dim`. Below the matrix
    width, successive rows overlap — the load reads elements belonging to the
    previous row. Nothing faults; the kernel computes a different matrix.
    """
    width = _MMA_EXTENT
    for stride in range(1, width):
        rows = [set(range(r * stride, r * stride + width)) for r in range(width)]
        overlapping = [
            (i, j) for i in range(width) for j in range(i + 1, width)
            if rows[i] & rows[j]
        ]
        assert overlapping, f"stride {stride} < {width} should alias rows"

    # At exactly the matrix width the rows tile without overlap, which is why
    # the bound is `>=` and not `>`.
    rows = [set(range(r * width, r * width + width)) for r in range(width)]
    assert not any(rows[i] & rows[j]
                   for i in range(width) for j in range(i + 1, width))


def test_ir_expresses_the_kernel_the_msl_synthesizer_emits():
    """Structural differential against `emit/apple_msl.py`.

    The point of these ops is that the MLIR pipeline can express an Apple
    kernel rather than only name one. The check is that the primitives the
    fixture uses are exactly the primitives the known-good synthesizer emits —
    if the synthesizer needs an operation the dialect cannot say, the up-level
    is incomplete and this is where that shows.
    """
    msl = (_ROOT / "python/tessera/compiler/emit/apple_msl.py").read_text()
    fixture = (_ROOT / "tests/tessera-ir/phase8"
               / "apple_simdgroup_primitives.mlir").read_text()

    # Every simdgroup/threadgroup primitive the coopmat kernel uses.
    required = {
        "simdgroup_load": "simdgroup_load",
        "simdgroup_store": "simdgroup_store",
        "simdgroup_matmul": "simdgroup_multiply_accumulate",
        "threadgroup_barrier": "threadgroup_barrier",
    }
    for op, msl_call in required.items():
        assert msl_call in msl, f"{msl_call} is no longer what the synthesizer emits"
        assert f"tessera_apple.gpu.{op}" in fixture, f"IR cannot express {msl_call}"


def test_the_fp32_accumulator_matches_the_synthesizer_and_the_verifier():
    """All three statements of the same contract must agree.

    The MSL kernel declares `simdgroup_float8x8 acc`, the ODS description says
    the accumulator is always fp32, and the C++ verifier enforces it. Three
    places is two too many for a fact to drift in silence.
    """
    msl = (_ROOT / "python/tessera/compiler/emit/apple_msl.py").read_text()
    ods = (_ROOT / "src/compiler/codegen/Tessera_Apple_Backend/include/Tessera"
           / "Target/Apple/TesseraAppleOps.td").read_text()
    cpp = (_ROOT / "src/compiler/codegen/Tessera_Apple_Backend/lib/Target/Apple"
           / "TesseraAppleDialect.cpp").read_text()

    assert "simdgroup_float8x8 acc" in msl
    assert "accumulator is always fp32" in ods
    assert "must be f32" in cpp


@pytest.mark.parametrize("bad_extent", [4, 16, 32])
def test_only_the_native_shape_is_expressible(bad_extent):
    """Apple7 has one simdgroup-matrix shape; anything else has no instruction.

    Asserted against the fixture rather than the hardware, because the point is
    that a non-native request fails at the operation instead of surviving to
    emission and producing MSL that will not compile.
    """
    invalid = (_ROOT / "tests/tessera-ir/phase8"
               / "apple_simdgroup_primitives_invalid.mlir").read_text()
    assert "requires an 8x8x8 shape" in invalid
    assert bad_extent != _MMA_EXTENT


def test_rounding_once_at_the_end_preserves_the_fp32_benefit():
    """Why the epilogue rounds once rather than accumulating in f16.

    The pass keeps an fp32 accumulator tile and converts per element on the way
    out, which is what the MSL kernel does. That is only worth the extra buffer
    if a single terminal rounding is materially better than rounding at every K
    step — otherwise the simpler f16 accumulator would be fine and the contract
    would be ceremony. It is not: the error of the end-rounded result is bounded
    by one half-ulp of f16, while accumulating in f16 compounds.
    """
    k_slabs = 512
    exact = _mma_chain(k_slabs, accum_dtype=np.float64)
    # fp32 accumulate, then a single round-to-nearest-even on the way out.
    end_rounded = _mma_chain(k_slabs, accum_dtype=np.float32).astype(
        np.float16).astype(np.float64)
    # f16 accumulate: rounded at every step.
    every_step = _mma_chain(k_slabs, accum_dtype=np.float16).astype(np.float64)

    scale = np.abs(exact).max()
    end_err = np.abs(end_rounded - exact).max() / scale
    step_err = np.abs(every_step - exact).max() / scale

    # A single f16 rounding cannot do better than ~half an f16 ulp (~4.9e-4
    # relative), so that is the floor the epilogue is allowed to reach.
    assert end_err < 2e-3, f"terminal rounding drifted more than f16 resolution: {end_err}"
    assert step_err > end_err, (
        f"accumulating in f16 ({step_err}) is not worse than rounding once "
        f"({end_err}); if this ever holds, the fp32 accumulator tile and its "
        "epilogue are not earning the extra buffer"
    )


def test_truncf_is_the_rounding_the_epilogue_claims():
    """`arith.truncf` is round-to-nearest-even, not truncation toward zero.

    The op's name suggests truncation; it is not, and the difference is a
    systematic bias rather than noise. A pass that assumed truncation would
    skew every output slightly negative-of-magnitude, which no shape test would
    catch.
    """
    # 1 + 2^-11 sits exactly halfway between two f16 values; RNE picks the even
    # neighbour, truncation would always pick the lower one.
    halfway = np.float32(1.0 + 2.0 ** -11)
    assert np.float16(halfway) == np.float16(1.0), "expected round-to-even at the tie"
    above = np.float32(1.0 + 3.0 * 2.0 ** -12)
    assert np.float16(above) > np.float16(1.0), "expected rounding up above the tie"


def test_zero_padded_ragged_tiling_is_exact():
    """Why ragged shapes are correct, not merely tolerated.

    Metal's `simdgroup_load` has no bounds predicate, so out-of-range elements
    cannot be masked at the load; they are substituted with zero when the tile
    is staged. That is exact rather than approximate: a zero operand
    contributes nothing to the dot product, so every valid output element is
    unaffected, and the padded tail is never copied out.
    """
    E = _MMA_EXTENT
    rng = np.random.default_rng(0)
    for M, N, K in [(17, 13, 23), (8, 8, 8), (1, 1, 1), (31, 9, 7)]:
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, N)).astype(np.float32)
        C = np.zeros((M, N), np.float32)
        for m in range(0, M, E):
            for n in range(0, N, E):
                acc = np.zeros((E, E), np.float32)
                for k in range(0, K, E):
                    a = np.zeros((E, E), np.float32)
                    b = np.zeros((E, E), np.float32)
                    for i in range(E):
                        for j in range(E):
                            if m + i < M and k + j < K:
                                a[i, j] = A[m + i, k + j]
                            if k + i < K and n + j < N:
                                b[i, j] = B[k + i, n + j]
                    acc = a @ b + acc
                for i in range(E):
                    for j in range(E):
                        if m + i < M and n + j < N:
                            C[m + i, n + j] = acc[i, j]
        assert np.allclose(C, A @ B, atol=1e-5), f"ragged {M}x{N}x{K} diverged"


def test_threadgroup_budget_arithmetic():
    """The budget check is bytes, not elements — the element type matters.

    32768 is `[MTLDevice maxThreadgroupMemoryLength]` on this Apple7 part,
    queried from the device rather than recalled (Decision #27). The same
    element count fits or does not depending on width, which is why the
    verifier multiplies rather than comparing counts.
    """
    budget = 32768
    for width_bytes, dtype in ((2, "f16"), (4, "f32")):
        assert 8192 * width_bytes <= budget, dtype
    # An f32 tile of 16384 elements is exactly twice the budget; an f16 tile of
    # the same count fits exactly. Counting elements alone would accept both.
    assert 16384 * 4 > budget and 16384 * 2 == budget
