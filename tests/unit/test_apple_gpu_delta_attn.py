"""Phase-G attention sprint, Sub-sprint D — delta-rule attention on Apple GPU.

gated_deltanet / kimi_delta_attention / modified_delta_attention are each a
sequential delta recurrence that is algebraically the quadratic form
O = (QKᵀ ⊙ mask) @ V with a per-token *column* weight c_r = β_r·decay_ratio
[/(1+‖K_r‖‖V_r‖) for the modified rule] plus an optional output gate.  Two GPU
batched matmuls + a host-constructed mask + a GPU mask multiply.  metal_runtime.
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
gpu = pytest.mark.hardware_apple_gpu

_B, _H, _S, _D = 2, 3, 14, 12


def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    return dict(
        Q=rng.standard_normal((_B, _H, _S, _D)).astype(np.float32),
        K=rng.standard_normal((_B, _H, _S, _D)).astype(np.float32),
        V=rng.standard_normal((_B, _H, _S, _D)).astype(np.float32),
        beta=rng.uniform(0.3, 1.0, (_B, _H, _S)).astype(np.float32),
        decay=rng.uniform(0.85, 0.99, (_B, _H, _S)).astype(np.float32),
        gate=rng.standard_normal((_B, _H, _S, _D)).astype(np.float32),
    )


def test_in_envelope():
    for op in ("tessera.gated_deltanet", "tessera.kimi_delta_attention",
               "tessera.modified_delta_attention"):
        assert op in _driver._APPLE_GPU_RUNTIME_OPS, op
        assert op in _runtime._APPLE_GPU_RUNTIME_OPS, op
        assert op in _driver._APPLE_GPU_DELTA_ATTN_OPS, op


def test_driver_runtime_delta_attn_envelopes_agree():
    assert _driver._APPLE_GPU_DELTA_ATTN_OPS == _runtime._APPLE_GPU_DELTA_ATTN_OPS


@gpu
@pytest.mark.parametrize("op,ref_fn", [
    ("tessera.gated_deltanet", "gated_deltanet"),
    ("tessera.kimi_delta_attention", "kimi_delta_attention"),
    ("tessera.modified_delta_attention", "modified_delta_attention"),
])
@pytest.mark.parametrize("variant", ["plain", "beta", "beta_decay", "beta_decay_gate"])
def test_delta_attention(op, ref_fn, variant):
    d = _inputs(hash((op, variant)) % 999)
    kw = {"causal": True}
    if "beta" in variant:
        kw["beta"] = d["beta"]
    if "decay" in variant:
        kw["decay"] = d["decay"]
    if "gate" in variant:
        kw["gate"] = d["gate"]
    out = R._apple_gpu_dispatch_delta_attn(op, [d["Q"], d["K"], d["V"]], kw, np)
    ref = getattr(ts.ops, ref_fn)(d["Q"], d["K"], d["V"], **kw)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-3, atol=1e-3)


def test_return_state_falls_back_to_numpy():
    d = _inputs(1)
    # return_state path is not GPU-routed → dispatcher returns None.
    assert R._apple_gpu_dispatch_delta_attn(
        "tessera.gated_deltanet", [d["Q"], d["K"], d["V"]],
        {"causal": True, "return_state": True}, np) is None


@gpu
def test_gated_deltanet_jit_metal_runtime():
    d = _inputs(2)
    Q, K, V = d["Q"], d["K"], d["V"]

    @ts.jit(target="apple_gpu")
    def f(q, k, v):
        return ts.ops.gated_deltanet(q, k, v, causal=True)

    np.testing.assert_allclose(
        np.asarray(f(Q, K, V)), np.asarray(ts.ops.gated_deltanet(Q, K, V, causal=True)),
        rtol=1e-3, atol=1e-3)
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"


@gpu
def test_modified_delta_jit_metal_runtime():
    d = _inputs(3)
    Q, K, V = d["Q"], d["K"], d["V"]

    @ts.jit(target="apple_gpu")
    def f(q, k, v):
        return ts.ops.modified_delta_attention(q, k, v, causal=True)

    np.testing.assert_allclose(
        np.asarray(f(Q, K, V)),
        np.asarray(ts.ops.modified_delta_attention(Q, K, V, causal=True)),
        rtol=1e-3, atol=1e-3)
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"
