"""
tessera_gemma/model_tessera.py — Tessera-Gemma decoder-only model.

Supports Gemma 4 (4B / 12B / 27B) with:
  • Per-layer alternating full / sliding-window attention
  • Grouped-Query Attention (GQA)
  • GeGLU MLP (or SwiGLU via config)
  • Paged KV cache for efficient autoregressive decoding
  • NTK-scaled RoPE
  • Tessera compiler annotation markers on every kernel module

Changes vs v0.1:
  • `DecoderBlock` now accepts `layer_idx` and routes to full or SWA attention.
  • Separate q/k/v projections replace the packed qkv Linear.
  • `GemmaMLP` replaces `SwiGLU` (separate gate/up/down projections).
  • `forward` accepts optional `kv_caches` list and `use_cache` flag.
  • Added `generate()` convenience method (greedy by default).
  • RoPE cache built from `head_dim` (not hidden_size // num_heads), which is
    critical for Gemma 4 where head_dim=256 ≠ hidden_size // num_heads.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .configs import GemmaConfig
from .kernels.rmsnorm_tessera import RMSNorm
from .kernels.mlp_swiglu_tessera import GemmaMLP
from .kernels.attention_tessera import TesseraAttention
from .kernels.kv_cache_tessera import PagedKVCache
from .ops.rope import precompute_rope_cache
from .utils.shapes import ShapeSpec, check_shape


# ---------------------------------------------------------------------------
# Decoder block
# ---------------------------------------------------------------------------

class GemmaDecoderBlock(nn.Module):
    """Single Gemma transformer decoder layer.

    Pre-norm architecture:
        x = x + attn(norm1(x))
        x = x + mlp(norm2(x))
    """

    def __init__(self, cfg: GemmaConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        # Attention — full or SWA based on layer index
        attn_kind = cfg.layer_attention_kind(layer_idx)
        swa_size = cfg.sliding_window_size if attn_kind == "sliding_window" else 0

        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = TesseraAttention(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            sliding_window=swa_size or 0,
            dropout_p=cfg.dropout_p,
        )
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = GemmaMLP(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            mlp_type=cfg.mlp_type,
        )

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: Optional[torch.Tensor],
        rope_sin: Optional[torch.Tensor],
        kv_cache: Optional[PagedKVCache] = None,
        use_cache: bool = False,
        update_cache: bool = True,
    ) -> torch.Tensor:
        # Self-attention sub-layer
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(
            x,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            kv_cache=kv_cache,
            use_cache=use_cache,
            update_cache=update_cache,
        )
        x = residual + x

        # MLP sub-layer
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


# ---------------------------------------------------------------------------
# Full causal LM
# ---------------------------------------------------------------------------

class TesseraGemmaForCausalLM(nn.Module):
    """Tessera port of the Gemma decoder-only language model.

    Usage::

        cfg   = GemmaConfig.gemma4_4b()
        model = TesseraGemmaForCausalLM(cfg).eval()

        # Prefill
        logits = model(input_ids)   # (B, T, vocab)

        # Generate with paged KV cache
        out_ids = model.generate(input_ids, max_new_tokens=64)
    """

    def __init__(self, cfg: GemmaConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [GemmaDecoderBlock(cfg, i) for i in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # RoPE cache (built lazily on first forward)
        self._rope_built = False
        self.rope_cos: Optional[torch.Tensor] = None
        self.rope_sin: Optional[torch.Tensor] = None

    # -----------------------------------------------------------------------
    # RoPE cache management
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def _build_rope(self, device: torch.device, dtype: torch.dtype) -> None:
        if self._rope_built:
            return
        cos, sin = precompute_rope_cache(
            seqlen=self.cfg.max_position_embeddings,
            head_dim=self.cfg.head_dim,
            theta=self.cfg.rope_theta,
            device=device,
            dtype=dtype,
            rope_scaling=self.cfg.rope_scaling,
        )
        self.rope_cos = cos  # (1, maxlen, 1, head_dim//2)
        self.rope_sin = sin
        self._rope_built = True

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor,
        kv_caches: Optional[List[Optional[PagedKVCache]]] = None,
        use_cache: bool = False,
        update_cache: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:   (B, T) integer token ids.
            kv_caches:   List of per-layer PagedKVCache (or None per layer),
                         length == num_hidden_layers.  Pass None to disable.
            use_cache:   Whether to read/write from kv_caches.
            update_cache: Whether to write new KV into the caches (False for
                          re-processing cached context without appending).

        Returns:
            logits: (B, T, vocab_size)
        """
        syms: dict = {}
        check_shape("input_ids", tuple(input_ids.shape), ShapeSpec(["B", "T"]), syms)

        B, T = input_ids.shape
        x = self.embed_tokens(input_ids)       # (B, T, hidden)

        self._build_rope(x.device, x.dtype)

        # Slice RoPE to current sequence length
        cos = self.rope_cos[:, :T]  # (1, T, 1, head_dim//2)
        sin = self.rope_sin[:, :T]

        for i, layer in enumerate(self.layers):
            layer_cache = (
                kv_caches[i]
                if (use_cache and kv_caches is not None and i < len(kv_caches))
                else None
            )
            x = layer(
                x,
                rope_cos=cos,
                rope_sin=sin,
                kv_cache=layer_cache,
                use_cache=use_cache,
                update_cache=update_cache,
            )

        x = self.norm(x)
        return self.lm_head(x)

    # -----------------------------------------------------------------------
    # Convenience: generate
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
        page_size: int = 128,
    ) -> torch.LongTensor:
        """Generate tokens autoregressively.

        Uses paged KV cache when use_cache=True (recommended).
        Falls back to full re-processing when use_cache=False.

        Returns:
            (B, T + new_tokens) integer tensor.
        """
        if temperature <= 0.0 or (top_k == 1):
            # Pure greedy — avoid sampling altogether
            temperature = 1.0
            top_k = 1

        if use_cache:
            from .utils.decoding import greedy_decode_cached, sample_decode
            if top_k == 1:
                return greedy_decode_cached(
                    self, input_ids,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=eos_token_id,
                    page_size=page_size,
                )
            # Sampling with cache: prefill first, then step
            B = input_ids.size(0)
            from .utils.kv_cache_factory import make_kv_caches
            from .kernels.kv_cache_tessera import PagedKVCache
            import torch.nn.functional as F

            dtype = next(self.parameters()).dtype
            caches = make_kv_caches(
                self.cfg.num_hidden_layers, B,
                self.cfg.num_kv_heads, self.cfg.head_dim,
                page_size=page_size, device=input_ids.device, dtype=dtype,
            )
            logits = self.forward(input_ids, kv_caches=caches,
                                  use_cache=True, update_cache=True)
            tokens = input_ids.clone()
            eos_hit = torch.zeros(B, dtype=torch.bool, device=input_ids.device)

            for _ in range(max_new_tokens):
                lgt = logits[:, -1, :] / max(temperature, 1e-8)
                from .utils.decoding import _top_k_filter, _top_p_filter
                lgt = _top_k_filter(lgt, top_k)
                lgt = _top_p_filter(lgt, top_p)
                probs = F.softmax(lgt, dim=-1)
                next_tok = torch.multinomial(probs, 1)
                tokens = torch.cat([tokens, next_tok], dim=1)
                if eos_token_id is not None:
                    eos_hit |= next_tok.squeeze(-1) == eos_token_id
                    if eos_hit.all():
                        break
                logits = self.forward(next_tok, kv_caches=caches,
                                      use_cache=True, update_cache=True)
            return tokens

        else:
            from .utils.decoding import greedy_decode, sample_decode
            if top_k == 1:
                return greedy_decode(self, input_ids,
                                     max_new_tokens=max_new_tokens,
                                     eos_token_id=eos_token_id)
            return sample_decode(self, input_ids,
                                 max_new_tokens=max_new_tokens,
                                 temperature=temperature,
                                 top_k=top_k, top_p=top_p,
                                 eos_token_id=eos_token_id)

    # -----------------------------------------------------------------------
    # Convenience: count parameters
    # -----------------------------------------------------------------------
    def num_parameters(self, trainable_only: bool = False) -> int:
        params = (p for p in self.parameters()
                  if p.requires_grad or not trainable_only)
        return sum(p.numel() for p in params)


# ---------------------------------------------------------------------------
# Back-compat alias
# ---------------------------------------------------------------------------
# The old class was `TesseraGemmaForCausalLM` — unchanged name, so imports
# from old code continue to work.  `DecoderBlock` is renamed to
# `GemmaDecoderBlock`; expose both.
DecoderBlock = GemmaDecoderBlock
