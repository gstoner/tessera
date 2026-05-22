"""
tessera_gemma/utils/decoding.py — Autoregressive generation utilities.

Provides:
  • greedy_decode          — argmax at every step, no KV cache
  • greedy_decode_cached   — argmax with paged KV cache
  • sample_decode          — temperature + top-k + top-p sampling
  • batch_decode           — batched decode with EOS stopping

All functions take a `model` that exposes::

    model.forward(input_ids, kv_caches=None, use_cache=False, update_cache=True)
      -> logits: (B, T, vocab_size)

    model.blocks            — list of decoder blocks (length = num_layers)
    model.cfg               — GemmaConfig instance
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .kv_cache_factory import make_kv_caches


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero-out all logits below the top-k values."""
    if k <= 0:
        return logits
    kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
    return logits.masked_fill(logits < kth, float("-inf"))


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus (top-p) filtering."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    cumprob = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens where cumprob - current > p (shift right to keep at least one)
    remove = cumprob - F.softmax(sorted_logits, dim=-1) > p
    sorted_logits[remove] = float("-inf")
    # Scatter back
    return logits.scatter(-1, sorted_idx, sorted_logits)


# ---------------------------------------------------------------------------
# Greedy decode (no cache)
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_decode(
    model: torch.nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int = 32,
    eos_token_id: Optional[int] = None,
) -> torch.LongTensor:
    """Greedy decode without KV cache (re-processes full context each step).

    Returns:
        (B, T + max_new_tokens) token tensor, or shorter if EOS hit.
    """
    tokens = input_ids.clone()
    eos_hit = torch.zeros(tokens.size(0), dtype=torch.bool, device=tokens.device)

    for _ in range(max_new_tokens):
        logits = model(tokens)               # (B, T_cur, V)
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
        tokens = torch.cat([tokens, next_tok], dim=1)
        if eos_token_id is not None:
            eos_hit |= next_tok.squeeze(-1) == eos_token_id
            if eos_hit.all():
                break

    return tokens


# ---------------------------------------------------------------------------
# Greedy decode with paged KV cache
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_decode_cached(
    model: torch.nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int = 32,
    eos_token_id: Optional[int] = None,
    page_size: int = 128,
) -> torch.LongTensor:
    """Greedy decode using a per-layer paged KV cache.

    Prefill processes all input_ids in one forward pass; subsequent steps
    only process the single new token.

    Returns:
        (B, T + new_tokens) tensor.
    """
    B = input_ids.size(0)
    cfg = model.cfg
    device = input_ids.device
    dtype = next(model.parameters()).dtype

    # Build one PagedKVCache per layer
    kv_caches = make_kv_caches(
        num_layers=cfg.num_hidden_layers,
        batch=B,
        kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        page_size=page_size,
        device=device,
        dtype=dtype,
    )

    # Prefill
    logits = model(input_ids, kv_caches=kv_caches, use_cache=True, update_cache=True)
    next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tokens = torch.cat([input_ids, next_tok], dim=1)

    eos_hit = torch.zeros(B, dtype=torch.bool, device=device)
    if eos_token_id is not None:
        eos_hit |= next_tok.squeeze(-1) == eos_token_id
    if eos_hit.all():
        return tokens

    # Decode — one token at a time
    for _ in range(max_new_tokens - 1):
        logits = model(next_tok, kv_caches=kv_caches, use_cache=True, update_cache=True)
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_tok], dim=1)
        if eos_token_id is not None:
            eos_hit |= next_tok.squeeze(-1) == eos_token_id
            if eos_hit.all():
                break

    return tokens


# ---------------------------------------------------------------------------
# Sampling decode (temperature + top-k + top-p)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_decode(
    model: torch.nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
    seed: Optional[int] = None,
) -> torch.LongTensor:
    """Autoregressive sampling with temperature / top-k / top-p.

    Returns:
        (B, T + new_tokens) tensor.
    """
    if seed is not None:
        torch.manual_seed(seed)

    tokens = input_ids.clone()
    eos_hit = torch.zeros(tokens.size(0), dtype=torch.bool, device=tokens.device)

    for _ in range(max_new_tokens):
        logits = model(tokens)[:, -1, :]       # (B, V)
        logits = logits / max(temperature, 1e-8)
        logits = _top_k_filter(logits, top_k)
        logits = _top_p_filter(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)  # (B, 1)
        tokens = torch.cat([tokens, next_tok], dim=1)
        if eos_token_id is not None:
            eos_hit |= next_tok.squeeze(-1) == eos_token_id
            if eos_hit.all():
                break

    return tokens
