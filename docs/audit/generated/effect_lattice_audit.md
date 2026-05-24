# Effect Lattice + Determinism Audit

Generated from `python/tessera/compiler/effect_audit.py`.  Don't edit by hand — regenerate via `python -c "from tessera.compiler.effect_audit import render_dashboard; open('docs/audit/generated/effect_lattice_audit.md', 'w').write(render_dashboard())"`.  Drift gated by `tests/unit/test_effect_audit.py`.

## Headline

- **245** ops in `OP_SPECS` carry an effect.
- **0** mismatch the TSOL spec anchors (of 25 anchored ops).
- **0** ops sit at the conservative `top` fallback level.
- **22** ops declare deterministic-aware numeric policies.

## Effect distribution

| Effect level | Count | Description |
|--------------|------:|-------------|
| `pure` | 208 | No side effects; output depends only on inputs. |
| `random` | 3 | RNG-bearing; result varies across calls. |
| `movement` | 2 | Explicit prefetch / async copy / wait. |
| `state` | 25 | Reads or writes compiler-visible state (KV cache). |
| `collective` | 7 | Async device / rank communication. |
| `memory` | 0 | Writes mutable tensors or aliases host memory. |
| `io` | 0 | Host I/O or unknown external calls. |
| `top` | 0 | Conservative fallback (unknown / unconstrained). |

## TSOL spec anchor cross-check

_No mismatches — every TSOL spec anchor (matmul, dropout, all_reduce, etc.) carries the expected effect._

## Ops at `top` (conservative fallback)

_No ops at `top` — every op narrows to a specific lattice level._

## Determinism-aware numeric policies

The TSOL spec promises `deterministic=True` flips ops to deterministic implementations.  Today's numeric-policy system (Sprint C2, 2026-05-11) attaches a `NumericPolicy` to 67 ops; the `deterministic` field controls per-op behavior.

**22** ops declare deterministic-aware default policies:

- `attn_compressed_blocks`
- `attn_local_window_2d`
- `attn_sliding_window`
- `attn_top_k_blocks`
- `deepseek_sparse_attention`
- `flash_attn`
- `gated_attention`
- `gated_deltanet`
- `gqa_attention`
- `hybrid_attention`
- `kimi_delta_attention`
- `lightning_attention`
- `linear_attn`
- `log_softmax`
- `logsumexp`
- `mla_decode`
- `mla_decode_fused`
- `modified_delta_attention`
- `mqa_attention`
- `multi_head_attention`
- `softmax`
- `softmax_safe`

## Per-anchor verification (TSOL effect map)

| Op | Spec effect | Declared effect | OK |
|----|-------------|-----------------|----|
| `all_gather` | `collective` | `collective` | ✅ |
| `all_reduce` | `collective` | `collective` | ✅ |
| `all_to_all` | `collective` | `collective` | ✅ |
| `cast` | `pure` | `pure` | ✅ |
| `conv2d` | `pure` | `pure` | ✅ |
| `conv3d` | `pure` | `pure` | ✅ |
| `dropout` | `random` | `random` | ✅ |
| `fft` | `pure` | `pure` | ✅ |
| `gelu` | `pure` | `pure` | ✅ |
| `gemm` | `pure` | `pure` | ✅ |
| `ifft` | `pure` | `pure` | ✅ |
| `irfft` | `pure` | `pure` | ✅ |
| `layer_norm` | `pure` | `pure` | ✅ |
| `matmul` | `pure` | `pure` | ✅ |
| `moe_combine` | `collective` | `collective` | ✅ |
| `moe_dispatch` | `collective` | `collective` | ✅ |
| `reduce_scatter` | `collective` | `collective` | ✅ |
| `relu` | `pure` | `pure` | ✅ |
| `rfft` | `pure` | `pure` | ✅ |
| `rmsnorm` | `pure` | `pure` | ✅ |
| `rng_normal` | `random` | `random` | ✅ |
| `rng_uniform` | `random` | `random` | ✅ |
| `silu` | `pure` | `pure` | ✅ |
| `softmax` | `pure` | `pure` | ✅ |
| `transpose` | `pure` | `pure` | ✅ |
