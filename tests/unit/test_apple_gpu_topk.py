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

# ── portable: top_k is intentionally NOT an envelope op (frontend-gated) ──────

def test_top_k_is_not_yet_in_the_runtime_envelope():
    # Honest state: the Metal kernel + dispatch are landed and hardware-verified
    # below, but top_k is NOT pipeline-routed — the @jit frontend can't emit the
    # multi-output op, so it never reaches the Tile→Apple pass.  Claiming it in
    # the envelope would assert a metal_runtime invariant the C++ pass can't
    # honor.  When the frontend lands, this flips together with the C++
    # isAppleGpuRuntimeOp + runtime-ops .inc wiring.
    from tessera.compiler.apple_gpu_envelope import _APPLE_GPU_RUNTIME_OPS

    assert "tessera.top_k" not in _APPLE_GPU_RUNTIME_OPS


# ── Darwin: hardware-verified Metal TopK (direct dispatch) ───────────────────

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
