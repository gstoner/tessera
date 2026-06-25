"""Compiler-generated GQA/MQA through ``runtime.launch()`` on gfx1151.

The same `rocm_flash_attn_compiled` executor that runs MHA flash_attn detects
grouped-query attention from the operand shapes — Q is [B,H,Sq,D], K/V are
[B,G,Sk,D] with G<H — builds the `gqa = true` kernel variant in-process, and
launches it with the two trailing runtime args (heads H, kv_ratio = H/G). No new
executor / matrix row: GQA rides the flash_attn lane, distinguished by shape.
Validated vs a numpy GQA reference (MQA / GQA / MHA-equivalence).

Skip-clean: tessera-opt not built, or no usable AMD GPU.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")


def _fa_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built (ninja -C build tessera-opt)")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt, causal, scale):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_flash_attn_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["q", "k", "v"], "output_name": "o",
        "ops": [{"op_name": "tessera.flash_attn", "result": "o",
                 "operands": ["q", "k", "v"],
                 "kwargs": {"causal": bool(causal), "scale": scale}}],
    })


def _gqa_ref(q, k, v, scale, causal):
    B, H, Sq, D = q.shape
    G = k.shape[1]; ratio = H // G
    o = np.zeros((B, H, Sq, D), np.float32)
    for b in range(B):
        for h in range(H):
            g = h // ratio
            s = scale * (q[b, h].astype(np.float32) @ k[b, g].astype(np.float32).T)
            if causal:
                i = np.arange(Sq)[:, None]; j = np.arange(k.shape[2])[None, :]
                s = np.where(j > i, -1e30, s)
            s = s - s.max(-1, keepdims=True)
            p = np.exp(s); p = p / p.sum(-1, keepdims=True)
            o[b, h] = p @ v[b, g].astype(np.float32)
    return o


@pytest.mark.parametrize("D,B,H,G,Sq,Sk,causal", [
    (16, 1, 8, 1, 32, 32, 0),    # MQA
    (16, 2, 8, 2, 32, 48, 0),    # GQA (ratio 4)
    (64, 1, 8, 4, 48, 48, 1),    # GQA (ratio 2), causal
    (16, 1, 4, 4, 32, 32, 0),    # MHA-equivalence (rides the MHA path)
])
def test_launch_rocm_gqa_matches_numpy(D, B, H, G, Sq, Sk, causal):
    rt = _fa_or_skip()
    rng = np.random.default_rng(9 + D + H + G + causal)
    q = (rng.standard_normal((B, H, Sq, D)) * 0.3).astype(np.float16)
    k = (rng.standard_normal((B, G, Sk, D)) * 0.3).astype(np.float16)
    v = (rng.standard_normal((B, G, Sk, D)) * 0.3).astype(np.float16)
    scale = 1.0 / float(np.sqrt(D))

    res = rt.launch(_artifact(rt, causal, scale), (q, k, v))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_flash_attn_compiled"
    out = res["output"]
    assert out.shape == q.shape

    ref = _gqa_ref(q, k, v, scale, causal)
    maxerr = float(np.max(np.abs(out - ref)))
    assert maxerr < 2e-2, f"GQA launch maxerr={maxerr} H={H} G={G} causal={causal}"
