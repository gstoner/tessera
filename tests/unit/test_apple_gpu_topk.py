"""On-device hard top-k (k>1) — the segmented_topk_gpu primitive.

argmax/argmin only give k==1; this lands the k>1 case via MPSGraph's native
TopK op (``topKWithSourceTensor:axis:k:``), exposed as the C ABI symbol
``tessera_apple_gpu_mpsgraph_topk_f32`` (values + indices) and routed through the
apple_gpu ``topk`` dispatch lane.

Scope of this proof: the Metal kernel + runtime dispatch + envelope wiring are
hardware-verified here. The full ``@jit(target="apple_gpu")`` single-call path
for ``top_k`` still falls back to eager because the frontend AST lowerer does not
yet emit the multi-output ``tessera.top_k`` op (tuple return) into Graph IR —
that frontend integration is tracked separately.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from tessera.compiler.apple_gpu_envelope import (
    APPLE_GPU_LANE_BY_OP,
    _APPLE_GPU_RUNTIME_OPS,
)


# ── portable: envelope + lane + handler wiring ───────────────────────────────

def test_top_k_is_in_the_apple_gpu_envelope():
    assert "tessera.top_k" in _APPLE_GPU_RUNTIME_OPS
    assert APPLE_GPU_LANE_BY_OP["tessera.top_k"] == "topk"


def test_topk_lane_has_a_registered_handler():
    from tessera.runtime import _apple_gpu_lane_handlers

    assert "topk" in _apple_gpu_lane_handlers()


# ── Darwin: hardware-verified Metal TopK ─────────────────────────────────────

@pytest.mark.skipif(sys.platform != "darwin", reason="TopK executes on Metal.")
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


@pytest.mark.skipif(sys.platform != "darwin", reason="TopK executes on Metal.")
def test_metal_topk_symbol_present():
    from tessera.runtime import _apple_gpu_mpsgraph_topk_f32

    assert _apple_gpu_mpsgraph_topk_f32() is not None


@pytest.mark.skipif(sys.platform != "darwin", reason="TopK executes on Metal.")
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
