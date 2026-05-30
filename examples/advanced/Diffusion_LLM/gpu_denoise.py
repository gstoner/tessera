"""Masked-diffusion LM (MDLM) denoising on the new Apple GPU compiler path.

A standalone, **torch-free** demo (independent of the `tessera_diffusion_llm`
research sketch, which still imports torch) that runs an MDLM-style iterative
unmasking loop driven by the Apple GPU surfaces that landed in the runtime:

  * the bidirectional backbone — RMSNorm + attention (`bmm` → softmax → `bmm`) +
    MLP — runs through the device bmm / rowop kernels, and
  * each unmasking step samples the newly-revealed tokens with the **GPU
    Gumbel-max sampler** (`runtime._apple_gpu_gumbel_sample`).

Diffusion LMs are exactly the workload where on-device sampling pays off: every
denoising step samples, so the sampler is on the critical path. Everything is
cross-checked against a numpy reference and degrades to numpy off Apple Silicon.

    from gpu_denoise import run_mdlm_demo, tiny_diffusion_config
    summary = run_mdlm_demo(tiny_diffusion_config())
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from tessera import runtime as R
from tessera import rng as TR


@dataclass(frozen=True)
class DiffusionDemoConfig:
    vocab_size: int = 64
    mask_token_id: int = 63
    seq_len: int = 12
    hidden: int = 32
    num_heads: int = 4
    steps: int = 6
    rms_eps: float = 1e-6


def tiny_diffusion_config() -> DiffusionDemoConfig:
    return DiffusionDemoConfig()


@dataclass(frozen=True)
class MDLMDemoSummary:
    backend: str
    steps: int
    all_unmasked: bool
    tokens_in_range: bool
    gpu_backbone_matches_numpy: bool
    gpu_sampler_matches_numpy: bool
    deterministic: bool


def _softmax(z, axis=-1):
    z = z - z.max(axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis, keepdims=True)


def _rmsnorm(x, gamma, eps):
    d = x.astype(np.float64)
    return (d / np.sqrt((d * d).mean(-1, keepdims=True) + eps)
            * gamma.astype(np.float64))


class _TinyDenoiser:
    """One-layer bidirectional transformer producing per-position logits.

    Weights are small + random; the demo is about the compiler path + the
    sampling loop, not model quality."""

    def __init__(self, cfg: DiffusionDemoConfig, *, seed: int = 0) -> None:
        self.cfg = cfg
        H, D, V = cfg.num_heads, cfg.hidden, cfg.vocab_size
        rng = np.random.RandomState(seed)
        f = lambda *s: (rng.randn(*s) * 0.08).astype(np.float32)
        self.embed = f(V, D)
        self.g1 = (rng.randn(D) * 0.1 + 1).astype(np.float32)
        self.Wq = f(D, D); self.Wk = f(D, D); self.Wv = f(D, D); self.Wo = f(D, D)
        self.g2 = (rng.randn(D) * 0.1 + 1).astype(np.float32)
        self.W1 = f(D, 4 * D); self.W2 = f(4 * D, D)
        self.gf = (rng.randn(D) * 0.1 + 1).astype(np.float32)
        self.Wlogit = f(D, V)
        self.head_dim = D // H

    # -- numpy reference forward --------------------------------------------
    def forward_numpy(self, tokens: np.ndarray) -> np.ndarray:
        cfg, H, dh = self.cfg, self.cfg.num_heads, self.head_dim
        S, D = cfg.seq_len, cfg.hidden
        x = self.embed[tokens].astype(np.float64)                 # [S, D]
        h = _rmsnorm(x, self.g1, cfg.rms_eps)
        q = (h @ self.Wq).reshape(S, H, dh).transpose(1, 0, 2)    # [H, S, dh]
        k = (h @ self.Wk).reshape(S, H, dh).transpose(1, 0, 2)
        v = (h @ self.Wv).reshape(S, H, dh).transpose(1, 0, 2)
        scores = np.einsum("hsd,htd->hst", q, k) / math.sqrt(dh)  # bidirectional
        attn = _softmax(scores, axis=-1)
        ctx = np.einsum("hst,htd->hsd", attn, v).transpose(1, 0, 2).reshape(S, D)
        x = x + ctx @ self.Wo
        h2 = _rmsnorm(x, self.g2, cfg.rms_eps)
        hidden = np.maximum(h2 @ self.W1, 0.0)                    # relu MLP
        x = x + hidden @ self.W2
        return _rmsnorm(x, self.gf, cfg.rms_eps) @ self.Wlogit    # [S, V]

    # -- Apple GPU forward (device bmm / rowop) -----------------------------
    def forward_gpu(self, tokens: np.ndarray) -> "np.ndarray | None":
        if not R.DeviceTensor.is_metal():
            return None
        cfg, H, dh = self.cfg, self.cfg.num_heads, self.head_dim
        S, D = cfg.seq_len, cfg.hidden
        dt = R.DeviceTensor

        def rmsnorm(xnp, gamma):
            dx = dt.from_numpy(np.ascontiguousarray(xnp, np.float32))
            dg = dt.from_numpy(gamma)
            out = R._apple_gpu_rowop_device(dx, 1, dg, cfg.rms_eps) if hasattr(R, "_apple_gpu_rowop_device") else None
            dx.free(); dg.free()
            return out.numpy().copy() if out is not None else _rmsnorm(xnp, gamma, cfg.rms_eps)

        def softmax_rows(xnp):
            dx = dt.from_numpy(np.ascontiguousarray(xnp, np.float32))
            out = R._apple_gpu_rowop_device(dx, 2, None, 0.0) if hasattr(R, "_apple_gpu_rowop_device") else None
            r = out.numpy().copy() if out is not None else _softmax(xnp.astype(np.float64)).astype(np.float32)
            dx.free()
            return r

        def bmm(a, b):  # a [batch,M,K], b [batch|1,K,N]
            da, db = dt.from_numpy(np.ascontiguousarray(a, np.float32)), dt.from_numpy(np.ascontiguousarray(b, np.float32))
            out = R._apple_gpu_bmm_device(da, db)
            r = out.numpy().copy() if out is not None else np.matmul(a, b)
            da.free(); db.free()
            if out is not None:
                out.free()
            return r

        x = self.embed[tokens].astype(np.float32)                 # [S, D]
        h = rmsnorm(x, self.g1)
        # projections as a single batch-1 bmm each
        q = bmm(h[None], self.Wq[None])[0].reshape(S, H, dh).transpose(1, 0, 2)
        k = bmm(h[None], self.Wk[None])[0].reshape(S, H, dh).transpose(1, 0, 2)
        v = bmm(h[None], self.Wv[None])[0].reshape(S, H, dh).transpose(1, 0, 2)
        kt = np.ascontiguousarray(k.transpose(0, 2, 1))           # [H, dh, S]
        scores = bmm(np.ascontiguousarray(q), kt) / math.sqrt(dh)  # [H, S, S]
        attn = np.stack([softmax_rows(scores[hh]) for hh in range(H)])
        ctx = bmm(np.ascontiguousarray(attn), np.ascontiguousarray(v))  # [H, S, dh]
        ctx = ctx.transpose(1, 0, 2).reshape(S, D)
        x = x + bmm(ctx[None], self.Wo[None])[0]
        h2 = rmsnorm(x, self.g2)
        hidden = np.maximum(bmm(h2[None], self.W1[None])[0], 0.0)
        x = x + bmm(hidden[None], self.W2[None])[0]
        xn = rmsnorm(x, self.gf)
        return bmm(xn[None], self.Wlogit[None])[0]                # [S, V]


def _mdlm_loop(denoiser, cfg, *, seed, use_gpu_forward, use_gpu_sampler):
    """Iterative unmasking: start fully masked, reveal the most-confident
    positions each step, sampling their tokens (Gumbel) until none remain."""
    S, V = cfg.seq_len, cfg.vocab_size
    tokens = np.full(S, cfg.mask_token_id, np.int64)
    masked = np.ones(S, bool)
    per_step = max(1, S // cfg.steps)
    step = 0
    while masked.any():
        logits = None
        if use_gpu_forward:
            logits = denoiser.forward_gpu(tokens)
        if logits is None:
            logits = denoiser.forward_numpy(tokens)
        logits = np.asarray(logits, np.float64)
        probs = _softmax(logits, axis=-1)
        conf = probs.max(-1)
        conf[~masked] = -1.0                       # only consider masked positions
        reveal = np.argsort(-conf)[:per_step]
        reveal = [p for p in reveal if masked[p]]
        if not reveal:
            break
        rows = logits[reveal].astype(np.float32)   # [k, V]
        key = TR.RNGKey.from_seed(seed).fold_in(step)
        if use_gpu_sampler:
            ids = R._apple_gpu_gumbel_sample(rows, np, key=key, temperature=0.7)
        else:
            noise = R._gumbel_noise_from_key(rows.shape, key, np)
            ids = np.argmax(rows / 0.7 + noise, axis=-1)
        for j, p in enumerate(reveal):
            tokens[p] = int(np.atleast_1d(ids)[j])
            masked[p] = False
        step += 1
    return tokens, step


def run_mdlm_demo(cfg: DiffusionDemoConfig) -> MDLMDemoSummary:
    den = _TinyDenoiser(cfg, seed=0)
    backend = "metal" if R.DeviceTensor.is_metal() else "numpy"

    # backbone: GPU forward == numpy forward (on a fixed token sequence)
    rng = np.random.RandomState(7)
    toks = rng.randint(0, cfg.vocab_size, size=cfg.seq_len).astype(np.int64)
    ref_logits = den.forward_numpy(toks)
    gpu_logits = den.forward_gpu(toks)
    backbone_ok = (gpu_logits is None) or np.allclose(gpu_logits, ref_logits, rtol=2e-3, atol=2e-3)

    # sampler: GPU gumbel == numpy gumbel on the same noise (a row block)
    block = ref_logits[:4].astype(np.float32)
    key = TR.RNGKey.from_seed(3)
    gpu_ids = R._apple_gpu_gumbel_sample(block, np, key=key, temperature=0.7)
    noise = R._gumbel_noise_from_key(block.shape, key, np)
    np_ids = np.argmax(block / 0.7 + noise, axis=-1)
    sampler_ok = np.array_equal(np.asarray(gpu_ids), np_ids.astype(np.asarray(gpu_ids).dtype))

    # full MDLM denoising loop (GPU forward + GPU sampler)
    tokens, steps = _mdlm_loop(den, cfg, seed=11, use_gpu_forward=True,
                               use_gpu_sampler=True)
    all_unmasked = not np.any(tokens == cfg.mask_token_id)
    in_range = bool((tokens >= 0).all() and (tokens < cfg.vocab_size).all())

    # determinism: same seed -> same generation
    t2, _ = _mdlm_loop(den, cfg, seed=11, use_gpu_forward=True, use_gpu_sampler=True)
    deterministic = bool(np.array_equal(tokens, t2))

    return MDLMDemoSummary(
        backend=backend, steps=steps, all_unmasked=bool(all_unmasked),
        tokens_in_range=in_range, gpu_backbone_matches_numpy=bool(backbone_ok),
        gpu_sampler_matches_numpy=bool(sampler_ok), deterministic=deterministic,
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
    s = run_mdlm_demo(tiny_diffusion_config())
    print("OK diffusion mdlm:", s.backend, "steps", s.steps,
          "all_unmasked", s.all_unmasked, "backbone==np", s.gpu_backbone_matches_numpy,
          "sampler==np", s.gpu_sampler_matches_numpy, "deterministic", s.deterministic)
