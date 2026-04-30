# KV-Cache Compression and Long-Context Serving

This example provides a compact reference for the highest-value long-context
inference work items:

- TurboQuant-style polar/int cache compression
- DuoAttention-style retrieval vs streaming cache budgets
- Mooncake-style prefill/decode disaggregation and cache-aware routing

The implementation is intentionally dependency-light. It models the scheduling and
memory accounting decisions that Tessera should eventually lower into runtime cache
managers and target kernels.

## Quick Start

```bash
python3 examples/advanced/kv_cache_serving/demo.py --requests 24 --context 131072
```

## Tessera Mapping

- Graph IR: mark cache tensors with compression policy and head role.
- Schedule IR: split prefill/decode placement and select cache tier.
- Runtime: route requests to cache-affine decode workers before falling back to
  cold prefill.
- Target kernels: specialize dequantization and attention score paths for the
  selected cache format.
