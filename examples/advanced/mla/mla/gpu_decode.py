"""MLA on the new Apple GPU compiler path.

Demonstrates the MLA decode work that landed in the runtime — weight absorption,
decoupled RoPE, paged + block-paged serving, and the GPU-resident decode loop —
driven from this example's config and cross-checked against a numpy reference.

Everything degrades to a numpy fallback off Apple Silicon, so the demo runs
anywhere; on a Mac with Metal it executes on the GPU.

    from mla import run_gpu_decode_demo, tiny_config
    summary = run_gpu_decode_demo(tiny_config())
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from tessera import runtime as R
from tessera import rng as TR
from tessera import cache as TC

from .config import MLAConfig


@dataclass(frozen=True)
class GPUDecodeSummary:
    absorbed_matches_explicit: bool
    paged_matches_reference: bool
    resident_loop_tokens: int
    block_paged_matches_reference: bool
    cache_bytes_per_token_latent: int
    cache_bytes_per_token_explicit: int
    backend: str


# --------------------------------------------------------------------------
# numpy references (decoupled-RoPE MLA)
# --------------------------------------------------------------------------
def _rope(x, cos, sin, style="interleaved"):
    dr = x.shape[-1]
    half = dr // 2
    out = np.empty_like(x)
    if style == "interleaved":
        a, b = x[..., 0::2], x[..., 1::2]
        out[..., 0::2] = a * cos - b * sin
        out[..., 1::2] = a * sin + b * cos
    else:
        a, b = x[..., :half], x[..., half:]
        out[..., :half] = a * cos - b * sin
        out[..., half:] = b * cos + a * sin
    return out


def _softmax(z, axis=-1):
    z = z - z.max(axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis, keepdims=True)


def _rope_tables(positions, dr, base=10000.0):
    half = dr // 2
    inv = base ** (-(np.arange(half) * 2.0 / dr))
    ang = positions[:, None] * inv[None, :]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def _ref_absorb(q_nope, q_rope, c_kv, k_rope, Wuk_t, Wuv, cosQ, sinQ, cosK, sinK):
    B, H, Sq, dn = q_nope.shape
    dr = q_rope.shape[-1]
    Skv, Dl = c_kv.shape[-2], c_kv.shape[-1]
    dv = Wuv.shape[-1]
    scale = 1.0 / math.sqrt(dn + dr)
    qrR = _rope(q_rope.astype(np.float64), cosQ[None, None], sinQ[None, None])
    krR = _rope(k_rope.astype(np.float64), cosK[None], sinK[None])
    O = np.empty((B, H, Sq, dv))
    for b in range(B):
        for h in range(H):
            qabs = q_nope[b, h].astype(np.float64) @ Wuk_t[h].astype(np.float64)
            s = (qabs @ c_kv[b].T.astype(np.float64)
                 + qrR[b, h] @ krR[b].T) * scale
            O[b, h] = _softmax(s) @ (c_kv[b].astype(np.float64) @ Wuv[h].astype(np.float64))
    return O


# --------------------------------------------------------------------------
def _build_mla(cfg: MLAConfig, *, seed: int = 0):
    """Build a small decoupled-RoPE MLA decode problem from the config."""
    rng = np.random.RandomState(seed)
    H = cfg.num_q_heads
    dn = cfg.head_dim
    dr = cfg.rope_dim if cfg.rope_dim % 2 == 0 else cfg.rope_dim + 1
    dv = cfg.head_dim
    Dl = cfg.latent_dim
    Skv = cfg.seq_len
    f = lambda *s: (rng.randn(*s) * 0.1).astype(np.float32)
    return dict(H=H, dn=dn, dr=dr, dv=dv, Dl=Dl, Skv=Skv,
                q_nope=f(1, H, 1, dn), q_rope=f(1, H, 1, dr),
                c_kv=f(1, Skv, Dl), k_rope=f(1, Skv, dr),
                Wuk=f(H, Dl, dn), Wuv=f(H, Dl, dv), rng=rng)


def run_gpu_decode_demo(cfg: MLAConfig) -> GPUDecodeSummary:
    """Exercise the new MLA decode surfaces and validate them against numpy."""
    m = _build_mla(cfg)
    H, dn, dr, dv, Dl, Skv = m["H"], m["dn"], m["dr"], m["dv"], m["Dl"], m["Skv"]
    Wuk_t = np.ascontiguousarray(np.swapaxes(m["Wuk"], 1, 2))      # [H, dn, Dl]
    cosK, sinK = _rope_tables(np.arange(Skv), dr)
    cosQ, sinQ = _rope_tables(np.asarray([Skv - 1]), dr)

    backend = "metal" if R.DeviceTensor.is_metal() else "numpy"

    # 1. Weight absorption == explicit decoupled-RoPE (the core MLA identity).
    absorbed = R._apple_gpu_mla_absorb_decode(
        m["q_nope"], m["q_rope"], m["c_kv"], m["k_rope"], Wuk_t, m["Wuv"],
        cosQ, sinQ, cosK, sinK, np)
    ref = _ref_absorb(m["q_nope"], m["q_rope"], m["c_kv"], m["k_rope"], Wuk_t,
                      m["Wuv"], cosQ, sinQ, cosK, sinK)
    absorbed_ok = absorbed is not None and np.allclose(
        absorbed, ref, rtol=1e-3, atol=1e-3)

    # 2. Paged single-sequence decode (MLAPagedDecoder) vs the same reference.
    dec = TC.MLAPagedDecoder(num_heads=H, nope_dim=dn, rope_dim=dr, v_dim=dv,
                             latent_dim=Dl, Wuk_t=Wuk_t, Wuv=m["Wuv"],
                             max_seq=max(64, Skv))
    dec.append(m["c_kv"][0], m["k_rope"][0])
    paged = dec.decode(m["q_nope"][0, :, 0, :], m["q_rope"][0, :, 0, :])
    paged_ok = np.allclose(paged, ref[0, :, 0, :], rtol=1e-3, atol=1e-3)

    # 3. GPU-resident multi-step decode loop (ResidentMLADecoder).
    vocab = 64
    rng = m["rng"]
    rdec = TC.ResidentMLADecoder(
        num_heads=H, head_dim=dn, vocab=vocab,
        rmsnorm_gamma=rng.randn(dn).astype(np.float32),
        w_logit=(rng.randn(H * dn, vocab) * 0.1).astype(np.float32))
    n_tokens = 0
    for step in range(4):
        kt = rng.randn(H, dn, 2 + step).astype(np.float32)
        v = rng.randn(H, 2 + step, dn).astype(np.float32)
        tok = rdec.step(rng.randn(H, dn).astype(np.float32), kt, v,
                        key=TR.RNGKey.from_seed(step))
        assert 0 <= tok < vocab
        n_tokens += 1
    rdec.free()

    # 4. Concurrent block-paged serving (MLABlockPagedCache) vs per-seq decode.
    bp = TC.MLABlockPagedCache(num_heads=H, nope_dim=dn, rope_dim=dr, v_dim=dv,
                               latent_dim=Dl, Wuk_t=Wuk_t, Wuv=m["Wuv"],
                               num_blocks=32, block_size=4)
    bp.add_sequence("s")
    bp.append("s", m["c_kv"][0], m["k_rope"][0])
    bp_out = bp.decode("s", m["q_nope"][0, :, 0, :], m["q_rope"][0, :, 0, :])
    bp_ok = np.allclose(bp_out, ref[0, :, 0, :], rtol=1e-3, atol=1e-3)

    return GPUDecodeSummary(
        absorbed_matches_explicit=bool(absorbed_ok),
        paged_matches_reference=bool(paged_ok),
        resident_loop_tokens=n_tokens,
        block_paged_matches_reference=bool(bp_ok),
        cache_bytes_per_token_latent=(Dl + dr) * 4,
        cache_bytes_per_token_explicit=H * (dn + dr + dv) * 4,
        backend=backend,
    )
