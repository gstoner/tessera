"""Phase-G attention sprint, Sub-sprint D — hybrid_attention policy wrapper.

hybrid_attention is a named per-layer policy that dispatches to a now-GPU-proven
delegate: Lightning / Kimi-Delta / Gated-DeltaNet linear attention, or softmax
flash attention for the MLA slot.  The dispatcher mirrors that routing and calls
the matching proven delegate dispatcher.  metal_runtime.
"""

import numpy as np
import pytest

import tessera as ts
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera import runtime as _runtime
from tessera.compiler import driver as _driver

R = _runtime
_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.skipif(not _GPU, reason="apple_gpu runtime / libtessera_jit unavailable")

_B, _H, _S, _D = 2, 3, 12, 12


def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    return dict(
        Q=rng.standard_normal((_B, _H, _S, _D)).astype(np.float32),
        K=rng.standard_normal((_B, _H, _S, _D)).astype(np.float32),
        V=rng.standard_normal((_B, _H, _S, _D)).astype(np.float32),
        beta=rng.uniform(0.3, 1.0, (_B, _H, _S)).astype(np.float32),
        decay=rng.uniform(0.85, 0.99, (_B, _H, _S)).astype(np.float32),
    )


def test_in_envelope():
    op = "tessera.hybrid_attention"
    assert op in _driver._APPLE_GPU_RUNTIME_OPS
    assert op in _runtime._APPLE_GPU_RUNTIME_OPS
    assert _driver._APPLE_GPU_HYBRID_ATTN_OPS == _runtime._APPLE_GPU_HYBRID_ATTN_OPS


@gpu
@pytest.mark.parametrize("kw", [
    {"pattern": "auto"},
    {"pattern": "lightning"},
    {"pattern": "ling_2_5", "layer_index": 0},
    {"pattern": "ling_2_5", "layer_index": 7},   # MLA/flash slot
    {"pattern": "kimi", "layer_index": 0},
    {"pattern": "kimi", "layer_index": 1},        # MLA/flash slot
    {"pattern": "delta"},
])
def test_hybrid_attention(kw):
    d = _inputs(hash(tuple(sorted(kw.items()))) % 999)
    full = dict(kw, beta=d["beta"], decay=d["decay"], causal=True)
    out = R._apple_gpu_dispatch_hybrid_attn("tessera.hybrid_attention", [d["Q"], d["K"], d["V"]], full, np)
    ref = ts.ops.hybrid_attention(d["Q"], d["K"], d["V"], **full)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-3, atol=1e-3)


@gpu
def test_return_state_falls_back():
    d = _inputs(1)
    assert R._apple_gpu_dispatch_hybrid_attn(
        "tessera.hybrid_attention", [d["Q"], d["K"], d["V"]],
        {"pattern": "auto", "return_state": True, "causal": True}, np) is None


@gpu
def test_hybrid_jit_metal_runtime():
    d = _inputs(2)
    Q, K, V = d["Q"], d["K"], d["V"]

    @ts.jit(target="apple_gpu")
    def f(q, k, v):
        return ts.ops.hybrid_attention(q, k, v, pattern="auto", causal=True)

    np.testing.assert_allclose(
        np.asarray(f(Q, K, V)),
        np.asarray(ts.ops.hybrid_attention(Q, K, V, pattern="auto", causal=True)),
        rtol=1e-3, atol=1e-3)
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"
