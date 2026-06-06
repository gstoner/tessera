"""Phase 3 Sprint 3.2 — Apple GPU kernel coverage toward the full transformer
block (docs/spec/PRODUCTION_COMPILER_PLAN.md).

Sprint 3.1 wired matmul / softmax / fused matmul→softmax / gelu. This sprint adds
the kernels that complete a transformer block on the Apple GPU back-half:
norms (rmsnorm/layer_norm), silu, the **fused single-head attention block**
``softmax(A@B)@C``, and the fused MLP chains (matmul→gelu, matmul→rmsnorm).

Contract (D4): every GPU result must match the **compiled CPU production lane**
(`tessera._jit_boundary`), which matches numpy. f32 only this sprint.

Skips on non-Darwin / when the Apple GPU runtime or libtessera_jit can't load.
"""

import numpy as np
import pytest

from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb

pytestmark = pytest.mark.skipif(
    not (agb.is_available() and jb.is_available()),
    reason="Apple GPU runtime or libtessera_jit unavailable",
)


# ── norms: GPU (unweighted) == CPU lane == numpy ─────────────────────────────


@pytest.mark.parametrize("shape", [(4, 8), (1, 16), (16, 64), (8, 128)])
def test_gpu_rmsnorm_matches_cpu_lane(shape):
    rng = np.random.default_rng(abs(hash(("rms", shape))) & 0xFFFF)
    x = (rng.standard_normal(shape) * 2.0).astype(np.float32)
    gpu = agb.gpu_rmsnorm(x)
    cpu = jb.jit_rmsnorm(x)
    np.testing.assert_allclose(gpu, cpu, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(4, 8), (1, 16), (16, 64), (8, 128)])
def test_gpu_layer_norm_matches_cpu_lane(shape):
    rng = np.random.default_rng(abs(hash(("ln", shape))) & 0xFFFF)
    x = (rng.standard_normal(shape) * 2.0 + 0.5).astype(np.float32)
    gpu = agb.gpu_layer_norm(x)
    cpu = jb.jit_layer_norm(x)
    np.testing.assert_allclose(gpu, cpu, rtol=1e-4, atol=1e-4)


# ── silu: GPU (MPSGraph) == CPU lane ─────────────────────────────────────────


@pytest.mark.parametrize("shape", [(4, 16), (1, 256), (8, 8)])
def test_gpu_silu_matches_cpu_lane(shape):
    rng = np.random.default_rng(abs(hash(("silu", shape))) & 0xFFFF)
    x = (rng.standard_normal(shape) * 3.0).astype(np.float32)
    gpu = agb.gpu_silu(x)
    cpu = jb.jit_silu(x)
    np.testing.assert_allclose(gpu, cpu, rtol=1e-4, atol=1e-4)


# ── fused single-head attention: ONE Metal kernel == CPU composition ─────────


@pytest.mark.parametrize("M,K,N,P", [(4, 8, 16, 8), (8, 16, 32, 16), (2, 4, 64, 4)])
def test_gpu_attention_matches_cpu_composition(M, K, N, P):
    """O = softmax(A@B)@C as one fused kernel must equal the CPU lane's un-fused
    matmul→softmax→matmul (the D2 fused-chain target override)."""
    rng = np.random.default_rng(M * 13 + N)
    a = (rng.standard_normal((M, K)) / np.sqrt(K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)
    c = rng.standard_normal((N, P)).astype(np.float32)

    gpu = agb.gpu_attention(a, b, c)
    cpu = jb.jit_matmul(jb.jit_softmax(jb.jit_matmul(a, b), axis=-1), c)

    np.testing.assert_allclose(gpu, cpu, rtol=1e-4, atol=1e-4)
    # numpy oracle
    s = a @ b
    e = np.exp(s - s.max(-1, keepdims=True))
    ref = (e / e.sum(-1, keepdims=True)) @ c
    np.testing.assert_allclose(gpu, ref, rtol=1e-4, atol=1e-4)


# ── fused MLP chains: matmul→gelu, matmul→rmsnorm ────────────────────────────


@pytest.mark.parametrize("M,K,N", [(4, 8, 16), (8, 16, 64), (2, 32, 128)])
def test_gpu_matmul_gelu_matches_cpu_composition(M, K, N):
    rng = np.random.default_rng(M * 5 + N)
    a = (rng.standard_normal((M, K)) / np.sqrt(K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)
    gpu = agb.gpu_matmul_gelu(a, b)
    cpu = jb.jit_gelu(jb.jit_matmul(a, b))
    # both tanh-approx gelu; tolerate kernel-vs-lowering spread
    np.testing.assert_allclose(gpu, cpu, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("M,K,N", [(4, 8, 16), (8, 16, 64), (2, 32, 128)])
def test_gpu_matmul_rmsnorm_matches_cpu_composition(M, K, N):
    rng = np.random.default_rng(M * 3 + N)
    a = (rng.standard_normal((M, K)) / np.sqrt(K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)
    gpu = agb.gpu_matmul_rmsnorm(a, b)
    cpu = jb.jit_rmsnorm(jb.jit_matmul(a, b))
    np.testing.assert_allclose(gpu, cpu, rtol=1e-4, atol=1e-4)


# ── composition: a pre-norm attention sub-block assembles from GPU kernels ───


def test_gpu_prenorm_attention_block_matches_cpu():
    """rmsnorm → self-attention(softmax(QKᵀ/√d)V) → residual, every op on the
    GPU back-half, matched against the same composition on the CPU lane.

    This is the Sprint-3.2 milestone: the attention half of a transformer block
    composes from production GPU kernels and stays oracle-clean end to end."""
    rng = np.random.default_rng(2026)
    T, D = 16, 32  # T=N=P kept <=256 for the fused-attention GPU path
    x = (rng.standard_normal((T, D)) * 0.5).astype(np.float32)
    wq = (rng.standard_normal((D, D)) / np.sqrt(D)).astype(np.float32)
    wk = (rng.standard_normal((D, D)) / np.sqrt(D)).astype(np.float32)
    wv = (rng.standard_normal((D, D)) / np.sqrt(D)).astype(np.float32)
    scale = np.float32(1.0 / np.sqrt(D))

    # GPU path
    xn_g = agb.gpu_rmsnorm(x)
    q_g = agb.gpu_matmul(xn_g, wq) * scale  # pre-scale into the no-scale kernel
    k_g = agb.gpu_matmul(xn_g, wk)
    v_g = agb.gpu_matmul(xn_g, wv)
    attn_g = agb.gpu_attention(q_g, k_g.T.copy(), v_g)  # softmax(Q Kᵀ) V
    out_g = x + attn_g  # residual

    # CPU-lane oracle (same composition)
    xn_c = jb.jit_rmsnorm(x)
    q_c = jb.jit_matmul(xn_c, wq) * scale
    k_c = jb.jit_matmul(xn_c, wk)
    v_c = jb.jit_matmul(xn_c, wv)
    scores_c = jb.jit_matmul(q_c, jb.jit_transpose(k_c))
    attn_c = jb.jit_matmul(jb.jit_softmax(scores_c, axis=-1), v_c)
    out_c = jb.jit_add(x, attn_c)

    np.testing.assert_allclose(out_g, out_c, rtol=1e-4, atol=1e-4)


# ── envelope: non-f32 rejected (no silent cast) ──────────────────────────────


def test_gpu_rmsnorm_rejects_non_f32():
    with pytest.raises(agb.AppleGpuError):
        agb.gpu_rmsnorm(np.ones((4, 4), np.float64))


def test_gpu_attention_rejects_shape_mismatch():
    with pytest.raises(agb.AppleGpuError):
        agb.gpu_attention(
            np.ones((4, 8), np.float32),
            np.ones((8, 16), np.float32),
            np.ones((8, 4), np.float32),  # C rows must equal N=16, not 8
        )
