# KV-Cache Compression and Long-Context Serving

This example provides a compact reference for the highest-value long-context
inference work items:

- TurboQuant-style polar/int cache compression
- DuoAttention-style retrieval vs streaming cache budgets
- Mooncake-style prefill/decode disaggregation and cache-aware routing

The scheduling/accounting layer is intentionally dependency-light
(`compression.py` + `scheduler.py`). The reference cache managers it targets now
exist, so `serving.py` **executes** the routing decisions against a real
`tessera.cache.MLABlockPagedCache` — a vLLM-style block pool with on-demand page
allocation, prefix-shareable sequences, ragged `decode_batch`, and free-list
reclamation — instead of only modeling them.

`run_serving_demo` closes the loop the rest of this README describes:

- **Prefix-affine routing → real page reuse.** A `decode_cache_hit` reuses an
  already-prefilled block-paged sequence; the demo reports the blocks saved vs.
  prefilling every request independently.
- **Block-paged accounting.** The pool's real `cache_bytes_per_token` (MLA
  compressed latent + shared RoPE key) is reported alongside the `CachePolicy`
  full-scale estimate.
- **Concurrent ragged decode.** All live sequences decode through the real kernel
  path and are validated against a contiguous numpy reference, proving the
  non-contiguous block gather is correct.
- **Eviction + reclaim.** Freeing a finished request returns its pages to the
  pool, and a fresh request reuses them.

## Quick Start

```bash
# Plan + execute against the real block-paged cache manager:
python3 examples/advanced/kv_cache_serving/demo.py --requests 24 --context 131072

# Just the routing/accounting plan (original behavior):
python3 examples/advanced/kv_cache_serving/demo.py --plan-only
```

## Tessera Mapping

- Graph IR: mark cache tensors with compression policy and head role.
- Schedule IR: split prefill/decode placement and select cache tier.
- Runtime: route requests to cache-affine decode workers before falling back to
  cold prefill.
- Target kernels: specialize dequantization and attention score paths for the
  selected cache format.
