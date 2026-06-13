"""P1 — DFlash attention core on the Apple GPU metal_runtime lane.

The DFlash draft layer folds heads into batch and runs its attention as a
rank-3 ``ops.flash_attn(+attn_bias)`` workload (the P0 substrate). This proves
the *exact* shapes a DFlash draft produces — GQA-folded, concatenated context +
proposal KV, optional sliding-window bias — execute on Metal and match the
numpy reference. Skipped off-Darwin / without a Metal device.
"""
import sys

import numpy as np
import pytest

import tessera as ts
from tessera import runtime as R
from tessera.nn import functional as F

try:
    from tessera._apple_gpu_dispatch import apple_gpu_available, apple_gpu_skip_reason
except Exception:  # pragma: no cover
    apple_gpu_available = lambda: False  # noqa: E731
    apple_gpu_skip_reason = lambda: "apple_gpu dispatch unavailable"  # noqa: E731

DARWIN = sys.platform == "darwin"

pytestmark = pytest.mark.skipif(
    not (DARWIN and apple_gpu_available()),
    reason=f"Metal device required ({apple_gpu_skip_reason()})",
)


# Module-level so @jit can inspect the source (heredoc / closures can't be).
@ts.jit(target="apple_gpu")
def _jit_flash_attn_bias(q, k, v, b):
    return ts.ops.flash_attn(q, k, v, attn_bias=b)


def _gpu_attn(q, k, v, *, scale, causal, attn_bias=None):
    """Route the rank-3 attention core through the Apple GPU metal_runtime
    dispatcher (the P0 flash_attn / flash_attn_bias symbols)."""
    operands = [q, k, v] if attn_bias is None else [q, k, v, attn_bias]
    return R._apple_gpu_dispatch_flash_attn(
        "tessera.flash_attn", operands, {"scale": scale, "causal": causal}, np)


def _weights(rng, D, Hq, Hkv, Dh):
    return dict(
        q_proj=rng.standard_normal((D, Hq * Dh)).astype(np.float32) * 0.1,
        k_proj=rng.standard_normal((D, Hkv * Dh)).astype(np.float32) * 0.1,
        v_proj=rng.standard_normal((D, Hkv * Dh)).astype(np.float32) * 0.1,
        o_proj=rng.standard_normal((Hq * Dh, D)).astype(np.float32) * 0.1,
        q_norm=rng.standard_normal(Dh).astype(np.float32) * 0.1 + 1.0,
        k_norm=rng.standard_normal(Dh).astype(np.float32) * 0.1 + 1.0,
    )


def test_jit_apple_gpu_flash_attn_bias_metal_runtime():
    """@jit(target='apple_gpu') compiles flash_attn(attn_bias=) through the
    Graph IR -> runtime pipeline and executes the bias op on the metal_runtime
    lane (not just the direct dispatcher path)."""
    rng = np.random.default_rng(42)
    B, S, Dh = 4, 8, 16          # folded (batch*heads), seq, head_dim
    q = rng.standard_normal((B, S, Dh)).astype(np.float32)
    k = rng.standard_normal((B, S, Dh)).astype(np.float32)
    v = rng.standard_normal((B, S, Dh)).astype(np.float32)
    bias = rng.standard_normal((B, S, S)).astype(np.float32)
    out = _jit_flash_attn_bias(q, k, v, bias)
    meta = _jit_flash_attn_bias.runtime_artifact().metadata
    assert meta["execution_mode"] == "metal_runtime"
    s = np.einsum("bqd,bkd->bqk", q, k) * (Dh ** -0.5) + bias
    s = s - s.max(-1, keepdims=True)
    a = np.exp(s); a /= a.sum(-1, keepdims=True)
    ref = np.einsum("bqk,bkd->bqd", a, v)
    assert np.allclose(np.asarray(out), ref, rtol=1e-3, atol=1e-3)


def test_public_apple_gpu_attention_fn_seam():
    """The public dflash.apple_gpu_attention_fn drives block_diffusion_attention's
    attention core onto Metal and matches the numpy reference."""
    from tessera import dflash as Df

    rng = np.random.default_rng(99)
    B, Hq, Hkv, Dh, D = 1, 4, 2, 16, 32
    L, S = 12, 6
    x = rng.standard_normal((B, L, D)).astype(np.float32)
    x_ctx = rng.standard_normal((B, S, D)).astype(np.float32)
    w = _weights(rng, D, Hq, Hkv, Dh)
    common = dict(num_heads=Hq, num_kv_heads=Hkv, head_dim=Dh, sliding_window=3, **w)
    ref = F.block_diffusion_attention(x, x_ctx, **common)
    gpu = F.block_diffusion_attention(x, x_ctx, attention_fn=Df.apple_gpu_attention_fn, **common)
    assert np.allclose(np.asarray(gpu), np.asarray(ref), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("sliding", [None, 4])
def test_block_diffusion_attention_core_on_metal(sliding):
    rng = np.random.default_rng(100 + (sliding or 0))
    B, Hq, Hkv, Dh, D = 1, 4, 2, 16, 32
    L, S, Sc = 15, 8, 32  # block, this-step context, prior cached context
    x = rng.standard_normal((B, L, D)).astype(np.float32)
    x_ctx = rng.standard_normal((B, S, D)).astype(np.float32)
    ck = rng.standard_normal((B, Sc, Hkv, Dh)).astype(np.float32)
    cv = rng.standard_normal((B, Sc, Hkv, Dh)).astype(np.float32)
    w = _weights(rng, D, Hq, Hkv, Dh)
    common = dict(num_heads=Hq, num_kv_heads=Hkv, head_dim=Dh,
                  cache_keys=ck, cache_values=cv, cache_offset=Sc,
                  sliding_window=sliding, **w)

    ref = F.block_diffusion_attention(x, x_ctx, **common)                  # numpy
    gpu = F.block_diffusion_attention(x, x_ctx, attention_fn=_gpu_attn, **common)
    assert gpu.shape == (B, L, D)
    assert np.allclose(np.asarray(gpu), np.asarray(ref), rtol=1e-3, atol=1e-3)


def test_full_dflash_layer_on_metal():
    """A whole DFlash decoder layer (attention core on Metal) matches numpy."""
    from tessera import dflash as Df

    rng = np.random.default_rng(7)
    cfg = Df.DFlashConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=16, intermediate_size=64,
                          vocab_size=41, block_size=16, target_layer_ids=(0, 1, 2))
    lw = Df.DFlashLayerWeights(
        q_proj=rng.standard_normal((32, 64)).astype(np.float32) * 0.1,
        k_proj=rng.standard_normal((32, 32)).astype(np.float32) * 0.1,
        v_proj=rng.standard_normal((32, 32)).astype(np.float32) * 0.1,
        o_proj=rng.standard_normal((64, 32)).astype(np.float32) * 0.1,
        q_norm=rng.standard_normal(16).astype(np.float32) * 0.1 + 1.0,
        k_norm=rng.standard_normal(16).astype(np.float32) * 0.1 + 1.0,
        input_layernorm=rng.standard_normal(32).astype(np.float32) * 0.1 + 1.0,
        post_attention_layernorm=rng.standard_normal(32).astype(np.float32) * 0.1 + 1.0,
        mlp_gate=rng.standard_normal((32, 64)).astype(np.float32) * 0.1,
        mlp_up=rng.standard_normal((32, 64)).astype(np.float32) * 0.1,
        mlp_down=rng.standard_normal((64, 32)).astype(np.float32) * 0.1)
    x = rng.standard_normal((1, 15, 32)).astype(np.float32)
    x_ctx = rng.standard_normal((1, 8, 32)).astype(np.float32)

    # Patch the layer's attention to use Metal by wrapping block_diffusion_attention.
    import tessera.nn.functional as FF
    orig = FF.block_diffusion_attention

    def gpu_bda(*a, **k):
        k.setdefault("attention_fn", _gpu_attn)
        return orig(*a, **k)

    FF.block_diffusion_attention = gpu_bda
    try:
        gpu = Df.dflash_decoder_layer(x, x_ctx, lw, cfg, 0)
    finally:
        FF.block_diffusion_attention = orig
    ref = Df.dflash_decoder_layer(x, x_ctx, lw, cfg, 0)
    assert np.allclose(np.asarray(gpu), np.asarray(ref), rtol=1e-3, atol=1e-3)
