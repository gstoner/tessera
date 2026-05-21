"""CorrDiff-core model — Sub-5 (2026-05-20).

A regional-weather diffusion model port that mirrors the NVIDIA CorrDiff
architecture at small scale:

  inputs       — coarse forecast fields ``(B, H, W, C_in)``
  conv stem    — two-layer NHWC conv2d (replace with U-Net later)
  spatial bias — 2D local-window attention over (H, W) spatial axes
  diffusion    — one denoising step with deterministic Philox noise
  outputs      — refined high-resolution field ``(B, H, W, C_out)``

This implementation is correctness-first.  Hot paths route through
``tessera.ops.conv2d`` (NHWC) and ``tessera.ops.attn_local_window_2d``
(B, H, Hq, Wq, D), both of which already have Graph IR + halo-aware
metadata + (for window-attention) the ODS op landed in Sub-1.

The model is small enough to run on a laptop CPU but exercises every
moving piece the Phase 7 asks built end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

import tessera as ts
from tessera.autodiff import checkpoint
from tessera.rng import RNGKey, normal


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CorrDiffConfig:
    """Minimal CorrDiff configuration.

    ``H``, ``W`` are the spatial extents of one *tile*; for distributed
    training each rank holds a (H, W) tile and halo exchange stitches
    neighbour boundaries (Ask 3 + Ask 4-B from the prior turn).
    """
    B:       int = 2
    H:       int = 32
    W:       int = 32
    C_in:    int = 4              # coarse forecast input channels
    C_hid:   int = 16             # conv stem hidden channels (= attention D)
    C_out:   int = 4              # refined output channels
    heads:   int = 2              # attention heads
    window:  tuple[int, int] = (1, 1)  # local-window halo (rh, rw)
    sigma:   float = 0.1          # diffusion noise stddev (one-step variant)
    seed:    int = 0              # RNG seed — deterministic across runs


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def tile_field(arr: np.ndarray, tile: tuple[int, int]) -> list[np.ndarray]:
    """Decompose an ``(H, W)``-shaped field into row-major tiles.

    Each tile is a contiguous spatial slab; with halo width = (1, 1) the
    integration walker (HaloMeshIntegrationPass) wraps a tile-local
    op's input with ``halo.exchange`` of the same width.  This helper
    keeps the benchmark's tiling story honest: real input pipelines do
    the same partitioning.

    Args:
      arr:  (H, W) or (..., H, W) array.
      tile: (tile_h, tile_w) — must divide H and W respectively.
    Returns:
      list of tile views, length (H // tile_h) * (W // tile_w),
      ordered row-major (axis-0 outer, axis-1 inner).
    """
    if arr.ndim < 2:
        raise ValueError(f"tile_field expects rank>=2; got {arr.ndim}")
    H, W = arr.shape[-2], arr.shape[-1]
    th, tw = tile
    if H % th or W % tw:
        raise ValueError(
            f"tile shape {tile} does not divide field shape ({H}, {W})"
        )
    out = []
    for i in range(0, H, th):
        for j in range(0, W, tw):
            out.append(arr[..., i:i + th, j:j + tw])
    return out


def diffusion_noise_step(x: np.ndarray, rng: RNGKey, sigma: float) -> np.ndarray:
    """One forward-diffusion noising step with deterministic Philox RNG.

    ``y = x + sigma * eps``, ``eps ~ N(0, I)``, sampled via the typed
    ``RNGKey`` so the same ``seed`` reproduces the same trajectory.
    """
    eps = normal(rng, x.shape, dtype=x.dtype)
    return x + sigma * eps


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────


class CorrDiffModel:
    """One-step CorrDiff regression+diffusion block.

    Forward path:

      x          : (B, H, W, C_in)            — coarse forecast tile
      h1         : conv2d(x, W1) → (B, H, W, C_hid)
      h2         : checkpoint(conv2d)(h1, W2) → (B, H, W, C_hid)
                   (activation checkpoint — recomputed during backward)
      tokens     : reshape(h2) → (B, heads, H, W, C_hid // heads)
      attended   : attn_local_window_2d(tokens, window=cfg.window)
      flat       : reshape(attended) → (B, H, W, C_hid)
      noised     : diffusion_noise_step(flat) — one-step Philox noise
      y          : conv2d(noised, W_out) → (B, H, W, C_out)

    Parameters are deterministic from ``cfg.seed`` so two model
    instances built with the same config produce bit-identical
    weights, activations, and outputs.
    """

    def __init__(self, cfg: CorrDiffConfig):
        if cfg.C_hid % cfg.heads:
            raise ValueError(
                f"C_hid={cfg.C_hid} must be divisible by heads={cfg.heads}"
            )
        self.cfg = cfg
        # Deterministic weight init from the config seed.  RNGKey carries
        # Philox state; fold_in derives a sub-key per parameter.
        master = RNGKey.from_seed(cfg.seed)
        self.W1 = self._init(
            master.fold_in("W1"),
            (3, 3, cfg.C_in, cfg.C_hid),
        )
        self.W2 = self._init(
            master.fold_in("W2"),
            (3, 3, cfg.C_hid, cfg.C_hid),
        )
        self.W_out = self._init(
            master.fold_in("Wout"),
            (1, 1, cfg.C_hid, cfg.C_out),
        )
        # Noise RNG is split from the master so each forward call gets a
        # fresh sub-key derived from the call's step counter.
        self.noise_master = master.fold_in("noise")

    # ----- helpers ------------------------------------------------------

    @staticmethod
    def _init(key: RNGKey, shape: tuple[int, ...]) -> np.ndarray:
        fan_in = np.prod(shape[:-1]) if len(shape) > 1 else 1.0
        scale = float(np.sqrt(2.0 / max(float(fan_in), 1.0)))
        return normal(key, shape, dtype="fp32") * scale

    # ----- forward ------------------------------------------------------

    def forward(self, x: np.ndarray, *, step: int = 0) -> np.ndarray:
        cfg = self.cfg
        B, H, W, C_in = x.shape
        if (H, W, C_in) != (cfg.H, cfg.W, cfg.C_in):
            raise ValueError(
                f"input shape {x.shape} does not match cfg "
                f"(H={cfg.H}, W={cfg.W}, C_in={cfg.C_in})"
            )

        # ── Conv stem ─────────────────────────────────────────────
        # First conv unchecked; second wrapped in checkpoint so its
        # activations are recomputed during backward (training-step
        # smoke).
        h1 = ts.ops.conv2d(x, self.W1, padding=1, layout="nhwc")

        def _h2(h: np.ndarray) -> np.ndarray:
            return ts.ops.conv2d(h, self.W2, padding=1, layout="nhwc")

        h2 = checkpoint(_h2)(h1)

        # ── 2D local-window attention ─────────────────────────────
        # Reshape NHWC → (B, heads, H, W, head_dim) so the rank-5
        # attn op can attend over spatial neighbourhoods.
        D = cfg.C_hid // cfg.heads
        tokens = h2.reshape(B, H, W, cfg.heads, D).transpose(0, 3, 1, 2, 4)
        attended = ts.ops.attn_local_window_2d(
            tokens, tokens, tokens, window=cfg.window,
        )
        flat = attended.transpose(0, 2, 3, 1, 4).reshape(B, H, W, cfg.C_hid)

        # ── Diffusion noise (one-step variant) ────────────────────
        rng = self.noise_master.fold_in(int(step))
        noised = diffusion_noise_step(flat, rng, cfg.sigma)

        # ── Output projection ─────────────────────────────────────
        y = ts.ops.conv2d(noised, self.W_out, padding=0, layout="nhwc")
        return y

    __call__ = forward
