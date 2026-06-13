"""DiffusionGemma Phase E — KV-cache promotion + block-diffusion decode loop.

Iterative block-diffusion decoding orchestrated over the real ``KVCacheHandle``:

  * single block  — embed the (mostly-masked) canvas, run one denoise step, commit
                    the confident positions;
  * multi-step    — repeat (≤ max_steps, ≤ 48) until the whole canvas commits,
                    freezing committed positions and re-noising the rest, with the
                    temperature annealing across steps (Phase C);
  * multi-block   — append the committed canvas's K/V to the context cache (KV
                    promotion) and decode the next canvas reading the grown cache.

This is **reference orchestration** (NumPy + the Phase B/C/D references over a
real KVCacheHandle). It proves single/multi-step/multi-block *correctness*; it
does **not** prove the "no host-only round-trips in the native path" property —
that needs the native fused kernels and is gated on Phase G. So no runtime/e2e
coverage is claimed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..cache import KVCacheHandle
from .diffusion_gemma import DiffusionGemmaConfig
from .sampler import SamplerConfig
from .block_diffusion import run_block_diffusion_step


@dataclass(frozen=True)
class BlockDecodeResult:
    tokens: np.ndarray          # (canvas,) committed tokens
    steps: int                  # denoise iterations used (≤ max_steps)
    stop_reason: str            # all_committed | max_steps
    committed: int              # positions committed (== canvas when all_committed)
    progress: tuple[int, ...]   # committed count after each step (monotone)


class BlockDiffusionDecoder:
    """Iterative block-diffusion decoder with a promoted context KV cache.

    ``embed_table`` (vocab, H) maps committed token ids to embeddings;
    ``mask_embedding`` (H,) is used for not-yet-committed canvas positions.
    Each decoded block's committed K/V is appended to ``KVCacheHandle`` so later
    blocks attend over the growing committed context.
    """

    def __init__(
        self,
        config: DiffusionGemmaConfig,
        weights: dict,
        embed_table: np.ndarray,
        *,
        num_denoise_layers: int,
        max_steps: int,
        sampler_config: SamplerConfig,
        top_k: int,
        mask_embedding: np.ndarray | None = None,
        max_context_blocks: int = 4,
    ) -> None:
        if max_steps > 48:
            raise ValueError("max_steps must be <= 48 (block-diffusion budget)")
        self.cfg = config
        self.w = weights
        self.embed = np.asarray(embed_table, dtype=np.float64)
        H = config.hidden_size
        self.mask_emb = (np.zeros(H) if mask_embedding is None
                         else np.asarray(mask_embedding, dtype=np.float64))
        self.num_layers = num_denoise_layers
        self.max_steps = max_steps
        self.sampler = sampler_config
        self.top_k = top_k
        self.kv = KVCacheHandle(
            num_heads=config.num_attention_heads, head_dim=config.head_dim,
            max_seq=config.canvas_size * max_context_blocks, dtype="fp32")
        self.blocks: list[np.ndarray] = []

    @property
    def context_len(self) -> int:
        return self.kv.current_seq

    def _embed(self, committed: np.ndarray, frozen: np.ndarray) -> np.ndarray:
        e = self.embed[committed].copy()
        e[~frozen] = self.mask_emb
        return e

    def context_kv(self, *, sliding: bool):
        """Read the committed context K/V (full, or last ``sliding_window``)."""
        n = self.kv.current_seq
        H = self.cfg.hidden_size
        if n == 0:
            return np.zeros((0, H)), np.zeros((0, H))
        start = max(0, n - self.cfg.sliding_window) if sliding else 0
        k, v = self.kv.read(start, n)              # (s, heads, head_dim)
        s = k.shape[0]
        return (np.asarray(k, np.float64).reshape(s, H),
                np.asarray(v, np.float64).reshape(s, H))

    def decode_block(self, *, rng_key: int, sliding: bool = False) -> BlockDecodeResult:
        cfg = self.cfg
        canvas = cfg.canvas_size
        committed = np.full(canvas, self.sampler.mask_id, dtype=np.int64)
        frozen = np.zeros(canvas, dtype=bool)
        progress: list[int] = []
        stop = "max_steps"
        steps = 0

        for step in range(self.max_steps):
            steps = step + 1
            embeds = self._embed(committed, frozen)
            enc = self.context_kv(sliding=sliding)
            res = run_block_diffusion_step(
                embeds, enc, self.w, step=step, sampler_config=self.sampler,
                num_denoise_layers=self.num_layers, rng_key=rng_key + step,
                top_k=self.top_k)
            newly = res.accepted_mask & ~frozen
            if newly.any():
                committed[newly] = res.tokens[newly]
            else:
                # Guaranteed progress: commit the single most-confident (lowest
                # entropy) still-unfrozen position (its raw sampled token) so
                # decoding always converges within max_steps.
                unfrozen = np.flatnonzero(~frozen)
                pick = int(unfrozen[int(np.argmin(res.entropy[unfrozen]))])
                committed[pick] = res.sampled[pick]
                newly = np.zeros(canvas, dtype=bool)
                newly[pick] = True
            frozen |= newly
            progress.append(int(frozen.sum()))
            if frozen.all():
                stop = "all_committed"
                break

        # KV promotion — append this committed block's K/V to the context cache.
        final = self._embed(committed, np.ones(canvas, dtype=bool))
        kv = final.reshape(canvas, cfg.num_attention_heads, cfg.head_dim).astype(np.float32)
        self.kv.append(kv, kv)
        self.blocks.append(committed.copy())

        return BlockDecodeResult(
            tokens=committed, steps=steps, stop_reason=stop,
            committed=int(frozen.sum()), progress=tuple(progress))


__all__ = ["BlockDiffusionDecoder", "BlockDecodeResult"]
