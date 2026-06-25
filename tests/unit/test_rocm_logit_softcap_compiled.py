"""Compiler-generated flash attention with Gemma-2 logit soft-capping on gfx1151.

Logit soft-capping (Gemma-2): each scaled attention score S is passed through
`cap * tanh(S / cap)` before masking + softmax, bounding the logits to
`(-cap, cap)`. The `tessera_rocm.flash_attn` directive gains a `logit_softcap`
attr; the generated kernel takes `cap` as a trailing f32 runtime arg and applies
the transform in the same step it scales the score (reusing the `tanh` →
`__ocml_tanh_f32` lowering). Reachable through `runtime.launch()` via a
`logit_softcap` kwarg on the flash_attn op — no new executor / matrix row.

Validated vs a numpy soft-capped attention reference; composes with causal.

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


def _artifact(rt, scale, cap, causal):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_flash_attn_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["q", "k", "v"], "output_name": "o",
        "ops": [{"op_name": "tessera.flash_attn", "result": "o",
                 "operands": ["q", "k", "v"],
                 "kwargs": {"scale": scale, "logit_softcap": cap,
                            "causal": bool(causal)}}],
    })


def _softcap_ref(q, k, v, scale, cap, causal):
    """Gemma-2 soft-capped attention reference. Q/K/V are [B, H, S, D]."""
    B, H, Sq, D = q.shape
    Sk = k.shape[2]
    o = np.zeros((B, H, Sq, D), np.float32)
    for b in range(B):
        for h in range(H):
            s = scale * (q[b, h].astype(np.float32) @ k[b, h].astype(np.float32).T)
            if cap > 0:
                s = cap * np.tanh(s / cap)
            if causal:
                i = np.arange(Sq)[:, None]
                j = np.arange(Sk)[None, :]
                s = np.where(j > i, -1e30, s)
            s = s - s.max(-1, keepdims=True)
            p = np.exp(s)
            p = p / p.sum(-1, keepdims=True)
            o[b, h] = p @ v[b, h].astype(np.float32)
    return o


@pytest.mark.parametrize("D,B,H,S,cap,causal", [
    (16, 1, 2, 48, 30.0, 0),
    (16, 1, 2, 64, 50.0, 1),
    (64, 2, 2, 48, 20.0, 0),
    (16, 1, 1, 32, 5.0, 1),    # tight cap — strong saturation
])
def test_launch_logit_softcap_matches_numpy(D, B, H, S, cap, causal):
    rt = _fa_or_skip()
    rng = np.random.default_rng(5 + D + H + S + int(cap) + causal)
    # Scale up so raw logits exceed the cap and the tanh actually bites.
    q = (rng.standard_normal((B, H, S, D)) * 1.2).astype(np.float16)
    k = (rng.standard_normal((B, H, S, D)) * 1.2).astype(np.float16)
    v = (rng.standard_normal((B, H, S, D)) * 0.3).astype(np.float16)
    scale = 1.0 / float(np.sqrt(D))

    res = rt.launch(_artifact(rt, scale, cap, causal), (q, k, v))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_flash_attn_compiled"
    out = res["output"].reshape(B, H, S, D)

    ref = _softcap_ref(q, k, v, scale, cap, causal)
    maxerr = float(np.max(np.abs(out - ref)))
    assert maxerr < 2e-2, f"softcap maxerr={maxerr} S={S} cap={cap} D={D}"


def test_softcap_differs_from_uncapped():
    """A real soft-cap must change the result vs uncapped attention when logits
    exceed the cap — guards against the cap being silently ignored."""
    rt = _fa_or_skip()
    D, B, H, S, cap = 16, 1, 1, 48, 5.0
    rng = np.random.default_rng(123)
    q = (rng.standard_normal((B, H, S, D)) * 2.0).astype(np.float16)
    k = (rng.standard_normal((B, H, S, D)) * 2.0).astype(np.float16)
    v = (rng.standard_normal((B, H, S, D)) * 0.5).astype(np.float16)
    scale = 1.0 / float(np.sqrt(D))

    capped = rt.launch(_artifact(rt, scale, cap, 0), (q, k, v))["output"]
    uncapped = _softcap_ref(q, k, v, scale, 0.0, 0)
    diff = float(np.max(np.abs(capped.reshape(B, H, S, D) - uncapped)))
    assert diff > 1e-2, "soft-capped output is indistinguishable from uncapped"
