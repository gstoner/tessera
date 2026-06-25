"""Compiler-generated sliding-window flash attention on gfx1151.

Sliding-window attention (Mistral-style): query position p attends only to keys
in ``(p - W, p]`` — a causal band of width W. The `tessera_rocm.flash_attn`
directive gains a `sliding_window` attr; the generated kernel takes W as a
trailing runtime arg, skips KV tiles entirely below the window's lower edge
(reusing the causal tile-skip mechanism), and trims the boundary with the
per-element mask. Reachable through ``runtime.launch()`` via a `window` kwarg on
the flash_attn op — no new executor / matrix row.

Validated vs a numpy windowed-attention reference (varied W, sequence length,
head dim), including the W ≥ S degenerate case (= plain causal).

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


def _artifact(rt, scale, window):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_flash_attn_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["q", "k", "v"], "output_name": "o",
        "ops": [{"op_name": "tessera.flash_attn", "result": "o",
                 "operands": ["q", "k", "v"],
                 "kwargs": {"scale": scale, "window": window}}],
    })


def _window_ref(q, k, v, scale, window):
    """Causal sliding-window attention reference: query p attends to keys in
    (p - window, p]. Q/K/V are [B, H, S, D]."""
    B, H, Sq, D = q.shape
    Sk = k.shape[2]
    o = np.zeros((B, H, Sq, D), np.float32)
    for b in range(B):
        for h in range(H):
            s = scale * (q[b, h].astype(np.float32) @ k[b, h].astype(np.float32).T)
            i = np.arange(Sq)[:, None]
            j = np.arange(Sk)[None, :]
            mask = (j > i) | (i - j >= window)   # future OR older than window
            s = np.where(mask, -1e30, s)
            s = s - s.max(-1, keepdims=True)
            p = np.exp(s)
            p = p / p.sum(-1, keepdims=True)
            o[b, h] = p @ v[b, h].astype(np.float32)
    return o


@pytest.mark.parametrize("D,B,H,S,window", [
    (16, 1, 2, 48, 16),    # window spans one tile
    (16, 1, 2, 64, 24),    # window straddles tiles
    (64, 2, 2, 80, 32),    # larger head dim, multi-tile window
    (16, 1, 1, 48, 8),     # tight window (< tile)
    (16, 1, 1, 32, 64),    # W >= S -> degenerate to plain causal
])
def test_launch_sliding_window_matches_numpy(D, B, H, S, window):
    rt = _fa_or_skip()
    rng = np.random.default_rng(3 + D + H + S + window)
    q = (rng.standard_normal((B, H, S, D)) * 0.3).astype(np.float16)
    k = (rng.standard_normal((B, H, S, D)) * 0.3).astype(np.float16)
    v = (rng.standard_normal((B, H, S, D)) * 0.3).astype(np.float16)
    scale = 1.0 / float(np.sqrt(D))

    res = rt.launch(_artifact(rt, scale, window), (q, k, v))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_flash_attn_compiled"
    out = res["output"].reshape(B, H, S, D)

    ref = _window_ref(q, k, v, scale, window)
    maxerr = float(np.max(np.abs(out - ref)))
    assert maxerr < 2e-2, f"window maxerr={maxerr} S={S} W={window} D={D}"


def test_window_differs_from_full_attention():
    """A real window must change the result vs full causal attention — guards
    against the window being silently ignored."""
    rt = _fa_or_skip()
    D, B, H, S, window = 16, 1, 1, 64, 8
    rng = np.random.default_rng(99)
    q = (rng.standard_normal((B, H, S, D)) * 0.5).astype(np.float16)
    k = (rng.standard_normal((B, H, S, D)) * 0.5).astype(np.float16)
    v = (rng.standard_normal((B, H, S, D)) * 0.5).astype(np.float16)
    scale = 1.0 / float(np.sqrt(D))

    windowed = rt.launch(_artifact(rt, scale, window), (q, k, v))["output"]
    full = _window_ref(q, k, v, scale, S)          # W=S -> plain causal
    # Late query rows (row index >> window) must diverge from full causal.
    diff = float(np.max(np.abs(windowed.reshape(B, H, S, D)[0, 0, -1] - full[0, 0, -1])))
    assert diff > 1e-2, "windowed output is indistinguishable from full causal"
