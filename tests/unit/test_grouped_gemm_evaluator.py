"""Evaluator integration for the grouped-GEMM low-precision contract.

Wires the existing fp8 / nvfp4 / fp8xfp4 grouped-GEMM oracles
(test_grouped_gemm_contract.py) into the evaluator's *metamorphic* harness
(compiler/evaluator.py): a quantize→dequantize of the inputs is the metamorphosis,
and the invariant the compiler must preserve is

    grouped_gemm(qdq(x), qdq(w)) ≈ grouped_gemm(x, w)   within the precision budget.

Both sides run natively on the real Metal backend (Darwin-only), reference-free —
"derive validates declare": the evaluator independently re-confirms the
low-precision contract the registry asserts, rather than trusting a numpy assert.
The teeth test shows a genuinely non-equivalent relation is flagged divergent.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import tessera as ts
from tessera.compiler import grouped_layout as gl
from tessera.compiler.evaluator import metamorphic_equivalence


def _gg(x, w, g):
    return ts.ops.grouped_gemm(x, w, g)


# group_sizes must be a real operand (not closed over) for the apple_gpu native
# path; it is identical on both metamorphic sides, so the only metamorphosis is
# the quantize→dequantize of x / w.
_GG = ts.jit(target="apple_gpu")(_gg)


def _qdq(x, w, quant):
    return gl.apply_quant_for_grouped(
        x, w, quant,
        quantize_fp8=ts.ops.quantize_fp8,
        quantize_nvfp4=ts.ops.quantize_nvfp4,
        dequantize_nvfp4=ts.ops.dequantize_nvfp4)


def _operands(seed=0):
    rng = np.random.default_rng(seed)
    T, K, N, E = 64, 128, 256, 4
    x = rng.standard_normal((T, K)).astype(np.float32)
    w = rng.standard_normal((E, K, N)).astype(np.float32)
    gs = np.array([T // E] * E, dtype=np.int64)
    return x, w, gs


# Per-precision elementwise-max tolerance budgets for grouped GEMM at this shape
# (looser than the norm-relative bounds in the contract test — max-abs over all
# elements is a stricter relation, and fp4's 1-bit mantissa needs headroom).
_BUDGETS = {"fp8_e4m3": 0.30, "nvfp4": 0.75, "fp8xfp4": 0.75}


@pytest.mark.skipif(sys.platform != "darwin", reason="Metal execution is Darwin-only.")
@pytest.mark.parametrize("quant", ["fp8_e4m3", "nvfp4", "fp8xfp4"])
def test_grouped_gemm_quant_equivalent_to_f32_via_evaluator(quant):
    """The low-precision grouped GEMM ≡ the f32 grouped GEMM within the
    precision budget — confirmed natively through the metamorphic oracle."""
    x, w, gs = _operands()
    xq, wq = _qdq(x, w, quant)
    v = metamorphic_equivalence("apple_gpu", _GG, (xq, wq, gs), (x, w, gs),
                                rtol=_BUDGETS[quant], atol=1e-3)
    assert v.relation == "equivalent", (quant, v.detail, v.max_abs_err)
    assert not v.is_divergent


@pytest.mark.skipif(sys.platform != "darwin", reason="Metal execution is Darwin-only.")
def test_evaluator_oracle_has_teeth():
    """A non-equivalent relation (weights scaled 4×) must be flagged divergent —
    proving the oracle isn't vacuously passing on any input pair."""
    x, w, gs = _operands()
    v = metamorphic_equivalence("apple_gpu", _GG, (x, 4.0 * w, gs), (x, w, gs),
                                rtol=1e-3, atol=1e-4)
    assert v.is_divergent, v.detail
