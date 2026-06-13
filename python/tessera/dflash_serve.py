"""DFlash serving — tokenizer-wired text generation (#9c) + a scheduler (#9d).

Thin orchestration over :func:`tessera.dflash.dflash_generate_cached`: a
string-in/string-out path using a Tessera tokenizer, and a ``DFlashScheduler``
that holds a draft + (stateful) target and serves generation requests. The
heavy lifting (cached drafting, stateful verify, rollback, sampling) lives in
``tessera.dflash`` / ``tessera.dflash_reference``; this is the user-facing seam.
"""
from __future__ import annotations

from typing import List, Optional

from . import dflash as _df


def dflash_generate_text(prompt: str, tokenizer, draft_w, cfg, target, *,
                         max_new_tokens: int, block_size: Optional[int] = None,
                         rope_fn=None, temperature: float = 0.0, top_k: int = 0,
                         top_p: float = 0.0, rng=None, eos_id: Optional[int] = None,
                         return_new_only: bool = True) -> str:
    """String-in/string-out DFlash generation.

    Encodes ``prompt`` with ``tokenizer`` (any object exposing ``encode(str) ->
    list[int]`` / ``decode(ids) -> str``, e.g. ``tessera.data.tokenizer_byte()``),
    runs the cached speculative loop, and decodes. ``return_new_only`` decodes
    just the generated continuation; set ``False`` to include the prompt.
    """
    ids = list(tokenizer.encode(prompt))
    prompt_len = len(ids)
    out = _df.dflash_generate_cached(ids, draft_w, cfg, target,
                                     max_new_tokens=max_new_tokens, block_size=block_size,
                                     rope_fn=rope_fn, temperature=temperature,
                                     top_k=top_k, top_p=top_p, rng=rng, eos_id=eos_id)
    emitted = out[prompt_len:] if return_new_only else out
    return tokenizer.decode(emitted)


class DFlashScheduler:
    """Minimal DFlash serving scheduler.

    Holds a draft (DFlashWeights) + a stateful target (e.g.
    :class:`tessera.dflash_reference.ReferenceDecoderLM`) and serves generation
    requests. One request at a time (the target carries a single KV cache that is
    reset per request); batched continuous-batching is future work. Greedy
    output is identical to greedy autoregressive decode from the target.
    """

    def __init__(self, draft_w, cfg, target, *, rope_fn=None,
                 default_max_new_tokens: int = 128):
        self.draft_w = draft_w
        self.cfg = cfg
        self.target = target
        self.rope_fn = rope_fn if rope_fn is not None else _df.make_rope(cfg.head_dim, cfg.rope_theta)
        self.default_max_new_tokens = int(default_max_new_tokens)

    def generate(self, prompt_ids, *, max_new_tokens: Optional[int] = None,
                 block_size: Optional[int] = None, temperature: float = 0.0,
                 top_k: int = 0, top_p: float = 0.0, rng=None,
                 eos_id: Optional[int] = None) -> List[int]:
        """Generate token ids for one request (prompt + continuation)."""
        return _df.dflash_generate_cached(
            prompt_ids, self.draft_w, self.cfg, self.target,
            max_new_tokens=max_new_tokens or self.default_max_new_tokens,
            block_size=block_size, rope_fn=self.rope_fn, temperature=temperature,
            top_k=top_k, top_p=top_p, rng=rng, eos_id=eos_id)

    def generate_text(self, prompt: str, tokenizer, *, max_new_tokens: Optional[int] = None,
                      return_new_only: bool = True, **kw) -> str:
        """String-in/string-out wrapper around :meth:`generate`."""
        return dflash_generate_text(
            prompt, tokenizer, self.draft_w, self.cfg, self.target,
            max_new_tokens=max_new_tokens or self.default_max_new_tokens,
            rope_fn=self.rope_fn, return_new_only=return_new_only, **kw)


__all__ = ["dflash_generate_text", "DFlashScheduler"]
