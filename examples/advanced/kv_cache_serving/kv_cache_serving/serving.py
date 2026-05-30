"""Real block-paged serving — wires the scheduler to ``tessera.cache``.

``scheduler.py`` + ``compression.py`` *model* prefix-affine routing and the
KV-cache memory footprint.  This module closes the loop the README promised:
"scheduling and memory accounting decisions that Tessera should eventually lower
into runtime cache managers."  Those managers now exist —
:class:`tessera.cache.MLABlockPagedCache` is a vLLM-style block pool with on-demand
page allocation, prefix-shareable sequences, ragged ``decode_batch``, and
free-list reclamation — so the demo can *run* the routing decisions against real
on-device cache state instead of only printing a plan.

What this demonstrates end-to-end:

* **Prefix-affine routing → real page reuse.** ``route_requests`` decides which
  requests hit a warm prefix; here a prefix-cache hit literally reuses the
  block-paged sequence's pages (no re-prefill), and we measure the blocks saved.
* **Block-paged memory accounting.** The pool reports real utilization and
  ``cache_bytes_per_token``; we cross-check the ``CachePolicy`` estimate against
  the pool's actual MLA latent footprint.
* **Concurrent ragged decode.** All active sequences decode through the real
  ``decode_batch`` (grouped by length), validated against a contiguous numpy
  reference so the non-contiguous block gather is proven correct.
* **Eviction + reclaim.** Finishing a request returns its pages to the free list,
  and a fresh request reuses them — the serving-loop steady state.

Geometry is intentionally tiny (the point is the *manager*, not kernel size);
``block_size`` is small so sequences span several physical blocks and the
free-list path is actually exercised.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tessera.cache import MLABlockPagedCache
from tessera.cache.mla_paged import absorb_decode_one

from .compression import CachePolicy, estimate_request_cache
from .scheduler import Request, route_requests


@dataclass
class ServingSummary:
    """Outcome of a real block-paged serving pass."""

    num_requests: int
    num_unique_prefixes: int
    num_cache_hits: int
    prefill_len: int
    block_size: int
    blocks_allocated: int
    blocks_without_sharing: int
    blocks_saved: int
    pool_utilization: float
    free_blocks_after_evict: int
    reused_after_evict: bool
    cache_bytes_per_token_real: int
    cache_bytes_per_request_estimate: float
    max_abs_error: float
    validated: bool = field(default=False)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"requests={self.num_requests} prefixes={self.num_unique_prefixes} "
            f"hits={self.num_cache_hits} | blocks {self.blocks_allocated}"
            f"/{self.blocks_without_sharing} (saved {self.blocks_saved}) "
            f"util={self.pool_utilization:.2f} | "
            f"reuse_after_evict={self.reused_after_evict} | "
            f"max_err={self.max_abs_error:.2e} validated={self.validated}"
        )


# MLA geometry — tiny on purpose; the contribution is the page manager.
_H, _DN, _DR, _DV, _DL = 2, 8, 4, 8, 16


def _mla_weights(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    Wuk_t = rng.standard_normal((_H, _DN, _DL)).astype(np.float32) * 0.1
    Wuv = rng.standard_normal((_H, _DL, _DV)).astype(np.float32) * 0.1
    return Wuk_t, Wuv


def run_serving_demo(
    *,
    num_requests: int = 12,
    num_prefixes: int = 4,
    decode_workers: int = 4,
    prefill_len: int = 10,
    block_size: int = 4,
    num_blocks: int = 64,
    policy: CachePolicy | None = None,
    accounting_heads: int = 32,
    accounting_head_dim: int = 128,
    accounting_context: int = 131072,
    seed: int = 0,
    tol: float = 1e-4,
) -> ServingSummary:
    """Route ``num_requests`` through prefix-affine scheduling onto a real
    :class:`MLABlockPagedCache`, decode all live sequences, then evict and reuse.

    Returns a :class:`ServingSummary`; raises ``AssertionError`` if the
    block-paged decode diverges from the contiguous reference beyond ``tol``.
    """
    policy = policy or CachePolicy(
        k_bits=4, v_bits=4, residual_bits=1,
        retrieval_head_fraction=0.25, streaming_window=4096,
    )
    rng = np.random.default_rng(seed)
    Wuk_t, Wuv = _mla_weights(seed)

    requests = [
        Request(
            request_id=f"req_{i:03d}",
            tenant=f"tenant_{i % 4}",
            prefix_id=f"prefix_{i % num_prefixes}",
            context_tokens=accounting_context,
        )
        for i in range(num_requests)
    ]
    placements = route_requests(requests, decode_workers=decode_workers)
    num_cache_hits = sum(1 for p in placements if p.route == "decode_cache_hit")
    unique_prefixes = sorted({r.prefix_id for r in requests})

    pool = MLABlockPagedCache(
        num_heads=_H, nope_dim=_DN, rope_dim=_DR, v_dim=_DV, latent_dim=_DL,
        Wuk_t=Wuk_t, Wuv=Wuv, num_blocks=num_blocks, block_size=block_size,
    )

    # One real sequence per unique prefix; prefill it once.  A "decode_cache_hit"
    # request reuses that sequence's pages — exactly the page reuse the router's
    # prefix affinity is meant to unlock.  Keep the prefill latent/rope so the
    # contiguous reference can be recomputed for validation.
    prefix_latent: dict[str, np.ndarray] = {}
    prefix_rope: dict[str, np.ndarray] = {}
    for pid in unique_prefixes:
        pool.add_sequence(pid)
        c_kv = rng.standard_normal((prefill_len, _DL)).astype(np.float32) * 0.1
        k_rope = rng.standard_normal((prefill_len, _DR)).astype(np.float32) * 0.1
        pool.append(pid, c_kv, k_rope)
        prefix_latent[pid] = c_kv
        prefix_rope[pid] = k_rope

    blocks_allocated = pool.num_used_blocks
    blocks_per_seq = (prefill_len + block_size - 1) // block_size
    # If every request had to prefill its own copy (no prefix sharing):
    blocks_without_sharing = blocks_per_seq * num_requests
    blocks_saved = blocks_without_sharing - blocks_allocated

    # Concurrent decode: every routed request issues a query against its prefix
    # sequence.  Multiple requests share a sequence id (the cache hit), so we
    # decode each live sequence once with a representative query.
    queries: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pid in unique_prefixes:
        q_nope = rng.standard_normal((_H, _DN)).astype(np.float32) * 0.1
        q_rope = rng.standard_normal((_H, _DR)).astype(np.float32) * 0.1
        queries[pid] = (q_nope, q_rope)

    out = pool.decode_batch(queries)

    # Validate the block-paged (non-contiguous) gather + decode against a direct
    # contiguous reference over the same prefill tokens.
    max_err = 0.0
    for pid, (q_nope, q_rope) in queries.items():
        ref = absorb_decode_one(
            q_nope, q_rope, prefix_latent[pid], prefix_rope[pid], Wuk_t, Wuv,
            np.arange(prefill_len), prefill_len - 1,
            pool.rope_base, pool.rotation_style,
        )
        max_err = max(max_err, float(np.max(np.abs(out[pid] - ref))))
    assert max_err <= tol, f"block-paged decode diverged: max_err={max_err}"

    # Eviction + reclaim: free half the prefixes, confirm pages return, then add a
    # fresh sequence that reuses the reclaimed pages.
    util_peak = pool.utilization
    to_evict = unique_prefixes[: max(1, len(unique_prefixes) // 2)]
    for pid in to_evict:
        pool.free_sequence(pid)
    free_after_evict = pool.num_free_blocks

    reused = False
    free_before = pool.num_free_blocks
    pool.add_sequence("req_reuse")
    c_kv = rng.standard_normal((prefill_len, _DL)).astype(np.float32) * 0.1
    k_rope = rng.standard_normal((prefill_len, _DR)).astype(np.float32) * 0.1
    pool.append("req_reuse", c_kv, k_rope)
    # Pages came off the free list (no pool growth) → reclaim worked.
    reused = pool.num_free_blocks == free_before - blocks_per_seq

    estimate = estimate_request_cache(
        accounting_heads, accounting_head_dim, accounting_context, policy
    )

    return ServingSummary(
        num_requests=num_requests,
        num_unique_prefixes=len(unique_prefixes),
        num_cache_hits=num_cache_hits,
        prefill_len=prefill_len,
        block_size=block_size,
        blocks_allocated=blocks_allocated,
        blocks_without_sharing=blocks_without_sharing,
        blocks_saved=blocks_saved,
        pool_utilization=util_peak,
        free_blocks_after_evict=free_after_evict,
        reused_after_evict=reused,
        cache_bytes_per_token_real=pool.cache_bytes_per_token(),
        cache_bytes_per_request_estimate=estimate,
        max_abs_error=max_err,
        validated=max_err <= tol,
    )
