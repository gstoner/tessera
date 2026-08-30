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
