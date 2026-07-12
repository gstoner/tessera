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

