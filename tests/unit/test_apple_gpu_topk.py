"""On-device hard top-k (k>1) — the segmented_topk_gpu primitive.

argmax/argmin only give k==1; this lands the k>1 case via MPSGraph's native
TopK op (``topKWithSourceTensor:axis:k:``), exposed as the C ABI symbol
``tessera_apple_gpu_mpsgraph_topk_f32`` (values + indices) and a directly-
callable runtime dispatch (``runtime._apple_gpu_dispatch_topk``).

Scope of this proof: the Metal kernel + runtime dispatch are hardware-verified
here via a direct dispatch call. ``tessera.top_k`` is intentionally NOT in the
runtime envelope: the @jit AST frontend cannot emit the multi-output op (tuple
return) into Graph IR, so it never flows the Tile→Apple pipeline. Adding it to
the envelope would assert a metal_runtime invariant the C++ pass can't honor for
an op it never sees; the full envelope wiring (Python envelope + C++
isAppleGpuRuntimeOp + runtime-ops .inc) lands atomically with the frontend.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import tessera as ts


# ── portable: top_k is a routed envelope op ──────────────────────────────────

def test_top_k_is_in_the_runtime_envelope():
    from tessera.compiler.apple_gpu_envelope import (
        APPLE_GPU_LANE_BY_OP,
        _APPLE_GPU_RUNTIME_OPS,
    )

    assert "tessera.top_k" in _APPLE_GPU_RUNTIME_OPS
    assert APPLE_GPU_LANE_BY_OP["tessera.top_k"] == "topk"


def test_topk_lane_has_a_registered_handler():
    from tessera.runtime import _apple_gpu_lane_handlers

    assert "topk" in _apple_gpu_lane_handlers()


# ── Darwin: full @jit pipeline — top_k flows to metal_runtime ────────────────

@pytest.mark.hardware_apple_gpu
def test_jit_top_k_routes_to_metal_runtime_at_rung8():
    from tessera.compiler.evaluator import Rung, evaluate
    from tests._support.apple import assert_native_apple_jit

    rng = np.random.default_rng(7)
    x = rng.standard_normal((4, 64)).astype(np.float32)
    kk = 5

    def _tk(x):
        return ts.ops.top_k(x, kk)                      # positional scalar k

    j = ts.jit(target="apple_gpu")(_tk)
    oidx = np.argsort(x, axis=-1)[:, ::-1][:, :kk]
    ovals = np.take_along_axis(x, oidx, axis=-1)
    oracle = np.stack([ovals, oidx.astype(np.float32)])  # (values, indices) tuple
    verdict = evaluate("apple_gpu", j, (x,), oracle, rtol=5e-3, atol=1e-3)
    assert verdict.rung is Rung.HARDWARE_VERIFIED, verdict.detail
    assert_native_apple_jit(j)


# ── Darwin: hardware-verified Metal TopK (direct dispatch) ───────────────────

@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize("rows,cols,k", [(5, 64, 4), (8, 256, 1), (3, 1024, 16)])
def test_metal_topk_matches_numpy(rows, cols, k):
    from tessera.runtime import _apple_gpu_dispatch_topk

    rng = np.random.default_rng(rows * 100 + cols + k)
    x = rng.standard_normal((rows, cols)).astype(np.float32)
    vals, idx = _apple_gpu_dispatch_topk("tessera.top_k", [x], {"k": k}, np)

    oidx = np.argsort(x, axis=-1)[:, ::-1][:, :k]
    ovals = np.take_along_axis(x, oidx, axis=-1)
    assert np.allclose(vals, ovals, atol=1e-5)        # values: Metal == numpy
    assert np.array_equal(idx, oidx)                  # indices: Metal == numpy
    assert vals.shape == (rows, k)
    assert idx.dtype == np.int64


@pytest.mark.hardware_apple_gpu
def test_metal_topk_symbol_present():
    from tessera.runtime import _apple_gpu_mpsgraph_topk_f32

    assert _apple_gpu_mpsgraph_topk_f32() is not None


@pytest.mark.hardware_apple_gpu
def test_metal_topk_axis_folding():
    # a non-last axis is folded to the last axis and restored
    from tessera.runtime import _apple_gpu_dispatch_topk

    rng = np.random.default_rng(7)
    x = rng.standard_normal((32, 6)).astype(np.float32)   # top-k over axis 0
    vals, idx = _apple_gpu_dispatch_topk("tessera.top_k", [x], {"k": 3, "axis": 0}, np)
    assert vals.shape == (3, 6)
    oidx = np.argsort(x, axis=0)[::-1][:3]
    ovals = np.take_along_axis(x, oidx, axis=0)
    assert np.allclose(vals, ovals, atol=1e-5)
