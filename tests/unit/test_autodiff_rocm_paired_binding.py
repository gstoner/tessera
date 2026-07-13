"""ROCm Tier-3 backward kernels satisfy the public paired ABI seam."""

from __future__ import annotations

import numpy as np

import tessera as ts


@ts.jit(target="rocm", autodiff="reverse", wrt=("q", "k", "v"))
def _flash(q, k, v):
    return ts.ops.flash_attn(q, k, v, causal=True)


def test_rocm_flash_backward_binds_paired_inputs_to_verified_lane(monkeypatch):
    import tessera.runtime as runtime

    seen = {}

    def fake_launch(artifact, args):
        seen["metadata"] = artifact.metadata
        seen["args"] = args
        return {
            "ok": True,
            "execution_mode": "hip_runtime",
            "execution_kind": "native_gpu",
            "output": tuple(np.zeros_like(x) for x in args[1:]),
        }

    monkeypatch.setattr(runtime, "launch", fake_launch)
    q = np.zeros((1, 2, 4, 16), np.float16)
    k = np.zeros_like(q)
    v = np.zeros_like(q)
    dout = np.ones_like(q)
    grads = _flash.native_backward(q, k, v, out_cotangents=dout)

    assert len(grads) == 3
    assert seen["metadata"]["compiler_path"] == "rocm_flash_attn_bwd_compiled"
    assert seen["metadata"]["ops"][0]["kwargs"]["causal"] is True
    assert seen["args"][0] is not q  # cotangent-first paired adapter
    assert _flash.last_backward_execution["evidence_target"] == "rocm_gfx1151"


@ts.jit(target="rocm", autodiff="reverse", wrt=("a", "b"))
def _matmul(a, b):
    return ts.ops.matmul(a, b)


def test_rocm_matmul_backward_is_two_forward_gemm_launches(monkeypatch):
    import tessera.runtime as runtime

    calls = []

    def fake_launch(artifact, args):
        calls.append((artifact.metadata, args))
        return {"ok": True, "execution_mode": "hip_runtime",
                "output": np.asarray(args[0], np.float32) @ np.asarray(args[1], np.float32)}

    monkeypatch.setattr(runtime, "launch", fake_launch)
    a = np.arange(12, dtype=np.float16).reshape(3, 4)
    b = np.arange(20, dtype=np.float16).reshape(4, 5)
    dout = np.ones((3, 5), np.float16)
    da, db = _matmul.native_backward(a, b, out_cotangents=dout)

    assert len(calls) == 2
    assert all(call[0]["compiler_path"] == "rocm_compiled" for call in calls)
    np.testing.assert_array_equal(da, dout.astype(np.float32) @ b.astype(np.float32).T)
    np.testing.assert_array_equal(db, a.astype(np.float32).T @ dout.astype(np.float32))
    assert _matmul.last_backward_execution["implementation"] == "composition"
