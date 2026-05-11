---
status: Informative
classification: Audit Snapshot
authority: Derived from `python/tessera/compiler/primitive_coverage.py`
last_updated: 2026-05-10
---

# Primitive Coverage State

This is the cleaner current state of Tessera standalone-compiler primitive
coverage.

The important distinction:

- **Existing reference surface** means the primitive/API exists in Python with
  unit coverage and can be used by small standalone models or compiler-facing
  tests.
- **Contract complete** means the primitive has complete mathematical,
  shape, dtype/layout, autodiff, batching, transpose, sharding, lowering,
  backend-kernel, and test contracts. Most shipped primitives are not yet
  contract complete.

## Registry Summary

Generated from `all_primitive_coverages()` on 2026-05-10:

| Metric | Count |
|---|---:|
| Total tracked primitives | 374 |
| Existing / shipped partial entries | 374 |
| Planned entries | 0 |
| Contract-complete entries | 0 |
| Entries at `contract_schema=explicit_semantic` | **75** (was 48) |
| Entries at `contract_schema=explicit_partial` | 299 |

The registry intentionally reports shipped work as **partial** until the full
compiler contract is complete. Within partial, the `explicit_semantic`
metadata tag distinguishes "shipped + axis-audited" from "shipped but axes
still default to partial/planned" — see *Contract-axis hardening pass*
below.

## Contract-axis hardening pass (2026-05-10)

A focused pass through `_EXISTING_CONTRACT_OVERRIDES` in
`python/tessera/compiler/primitive_coverage.py` promoted **27 entries** from
`explicit_partial` to `explicit_semantic` by declaring math/shape/dtype/
batching/masking-effect axes complete and transpose/sharding axes
explicitly per the standard transformer / KV-cache / RL-loss conventions.
Backend-kernel hardening stays gated behind Phase G / H / I.

| Group | Entries (count) | Math / shape / dtype / batching | Transpose | Sharding | Effect |
|---|---:|---|---|---|---|
| KV cache state ops | `kv_cache_append`, `kv_cache_prune`, **NEW** `kv_cache_read` (3) | `complete` | `not_applicable` | `partial` | `complete` |
| Position encodings | `rope`, `rope_split`, `rope_merge`, `alibi`, `ntk_rope` (5) | `complete` | `complete` | `complete` | `not_applicable` |
| Attention wrappers | `flash_attn`, `multi_head_attention`, `gqa_attention`, `mqa_attention` (4) | `complete` | `partial` | `partial` | `complete` |
| MLA family | `latent_kv_compress`, `latent_kv_expand_k`, `latent_kv_expand_v`, `mla_decode`, `mla_decode_fused` (5) | `complete` | `partial` | `partial` | `complete` |
| Sparse attention | `attn_sliding_window`, `attn_top_k_blocks`, `attn_compressed_blocks` (3) | `complete` | `partial` | `partial` | `complete` |
| Linear / recurrent attention | `linear_attn`, `linear_attn_state`, `power_attn`, `retention` (4) | `complete` | `partial` | `partial` | `complete` |
| Reasoning-model attention family | `deepseek_sparse_attention`, `lightning_attention`, `gated_attention`, `hybrid_attention`, `gated_deltanet`, `kimi_delta_attention`, `modified_delta_attention` (7) | `complete` | `partial` | `partial` | `complete` |
| RL policy losses | `ppo_policy_loss`, `grpo_policy_loss`, `cispo_policy_loss` (3) | `complete` | `not_applicable` | `partial` | `not_applicable` |

The KV-cache audit confirms that `vjp`/`jvp` should remain `not_applicable`
for the trio: cache append/prune/read are state-effect ops whose gradient
boundary is the cache write — the K/V slices feeding the cache are still
differentiable upstream, but no gradient flows through the cache write
itself. The `effect=state` annotation in `OP_SPECS` is the explicit
declaration of this boundary.

## Done By Sprint

| Sprint | Current shipped surface |
|---|---|
| S1 | Primitive coverage registry, dashboard snapshot, duplicate-name guardrails. |
| S2 | Tensor algebra, indexing, reductions, scalar math, comparisons/logical ops, numeric helpers, stability primitives. |
| S3 | State trees: flatten/unflatten/map/reduce/transpose, state filters, partitions, module state projection. |
| S4 | RNG keys, split/fold-in/clone/state replay, samplers including normal, Bernoulli, categorical, multinomial, randint, permutation, gamma, beta, dirichlet, poisson. |
| S5 | Control flow and transforms: scan, associative_scan, while_loop, fori_loop, cond, switch, map, vmap, pmap, vjp, jvp, remat/checkpoint, autocast, axis helpers. |
| S6 | Sharding and collectives: named mesh/sharding, partition specs, shard_map, psum/pmean/pmax/pmin, collective_permute, broadcast_to_axis. |
| S7 | Model layers: LinearGeneral, Einsum, Conv1d, ConvTranspose1d, LoRA, GroupNorm, InstanceNorm, WeightNorm, SpectralNorm, pooling, SimpleRNN/GRU helpers, RoPE/ALiBi/NTK RoPE, MHA/GQA/MQA/MLA wrappers, DeepSeek/NSA sparse attention, Lightning, Gated DeltaNet, Kimi/Modified Delta, gated and hybrid attention. |
| S8 | Tiny conformance smoke tests: recurrent, diffusion-like, attention-like, plus training-step conformance using data/loss/optimizer/checkpoint. |
| S9 | Quantization/numerics: int8/int4/fp8/fp4 quant/dequant, fake quant, STE VJP/JVP coverage, calibration observer, grad scaler step. |
| S10 | Optimizers/schedules/grad transforms: SGD, Momentum, Adam, AdamW, Adafactor, Lion, Muon, LAMB, mixed-precision dtype policy, schedules, clipping, EMA, Polyak, transform chain. |
| S11 | Losses: regression, classification, distribution, contrastive, diffusion, sequence, CTC reference, plus PPO/GRPO/CISPO reasoning-RL helpers. |
| S12 | State serialization/checkpointing: save/load state, checksums, partial collection load, migration, mock sharded save/load. |
| S13 | Custom primitive API: custom primitive/call, VJP/JVP/batching metadata, per-target lowering registration. |
| S14 | AOT/cache: reference AOT export/load, StableHLO text, GGUF metadata, safetensors-like export, persistent compilation cache. |
| S15 | Data/tokenizers: eager Dataset combinators, sharded/iterable datasets, checkpoint/restore, deterministic shuffle, byte/vocab tokenizers. |

## Newly Promoted

The prior planned entries are now shipped as Python-reference or catalog-backed
primitives:

| Category | Promoted entries |
|---|---|
| Memory | `memory_read`, `memory_write`, `memory_evict` |
| Reductions | `max`, `min`, `cummax`, `cummin` |
| RoPE / MLA / sparse attention | `rope`, `rope_split`, `rope_merge`, `ntk_rope`, `latent_kv_compress`, `latent_kv_expand_k`, `latent_kv_expand_v`, `mla_decode`, `mla_decode_fused`, `attn_sliding_window`, `attn_compressed_blocks`, `attn_top_k_blocks`, `deepseek_sparse_attention` |
| Modern attention variants | `multi_head_attention`, `gqa_attention`, `mqa_attention`, `gated_attention`, `hybrid_attention`, `lightning_attention`, `gated_deltanet`, `kimi_delta_attention`, `modified_delta_attention`, `power_attn`, `retention` |
| Quantization STE | `quantize_fp8`, `dequantize_fp8`, `quantize_fp4`, `dequantize_fp4`, `fake_quantize` |
| MoE transport | `moe_dispatch`, `moe_combine` |
| Optimizers | `adam`, `adamw`, `momentum`, `adafactor`, `lion` |
| Reasoning RL | `normalize_group_advantages`, `ppo_policy_loss`, `grpo_policy_loss`, `cispo_policy_loss` |

The memory entries unblock Titans/Atlas-style active-memory conformance at the
reference level. The reduction entries close the small API alias/coverage gaps
around the broader reduction surface that already includes `amax`, `amin`,
`argmax`, `argmin`, `cumsum`, `cumprod`, `mean`, `var`, and related helpers.
The attention, quantization, MoE, optimizer, and RL promotions are the
reasoning-model support batch for MoSA, DeepSeek-R1-style MLA/NSA, MiniMax
Lightning/long-context paths, Kimi Delta-style recurrence, and post-training
loops.

## Largest Remaining Contract Gaps

Across the registry, the broad gaps are compiler-contract gaps rather than
missing Python API names:

| Contract axis | Missing / partial count | Δ from prior pass |
|---|---:|---:|
| Mathematical semantics | 299 | −26 |
| Shape rule | 299 | −26 |
| Dtype/layout rule | 299 | −26 |
| Batching rule | 340 | −33 |
| Transpose rule | 313 | −8 |
| **Sharding rule** | **156** | **−213** (category-based classifier) |
| Backend kernel | 374 | unchanged (Phase G/H/I gate) |
| JVP | 221 | unchanged |
| VJP | 137 | unchanged |
| Tests | 196 | +1 (new `kv_cache_read` entry) |
| Lowering rule | 147 | unchanged |
| Masking/effect rule | 16 | −11 |

## Long-tail sharding-rule pass (2026-05-10)

The previous `sharding_rule` audit showed 369 entries still
`partial`/`planned` — Decision #25's flagship long pole. A
**category-based classifier** (`_SHARDING_RULE_BY_CATEGORY` in
`primitive_coverage.py`) routes each primitive to one of three
verdicts based on its compiler category. Per-name overrides in
`_EXISTING_CONTRACT_OVERRIDES` still win.

Final distribution across 374 entries:

| Verdict | Count | Share | Examples |
|---|---:|---:|---|
| **`complete`** | **184** | 49% | All elementwise (`add`, `mul`, `silu`, `gelu`, ...), scalar math, comparisons / logical, reductions (`sum`, `mean`, `argmax`, `cumsum`), stability primitives (`logsumexp`, `log_softmax`), RNG samplers (per-shard via `fold_in`), losses (reduce to scalar/per-sample), collectives themselves (`psum`/`pmean`/`shard_map`), quantization (per-tensor symmetric), optimizers (ZeRO-style per-parameter), all transforms (`vjp`/`jvp`/`vmap`/`pmap`/`remat`), position encodings (`rope`/`alibi`/`ntk_rope`), `custom_*` extension API |
| **`partial`** | **156** | 42% | All attention variants (TP-along-head / SP-along-sequence well-understood, mesh-dependent), `matmul`/`gemm`/`einsum` (contract-axis all-reduce), structural ops (`reshape`/`cat`/`split`/`permute` — depends on partition spec), indexing (`gather`/`scatter` — indices replicated), MoE (`moe`/`moe_dispatch`/`moe_combine`), normalization (feature-axis all-reduce when sharded), stencil (`depthwise_conv1d` — halo exchange), spectral / FFT (ring/butterfly), linalg solvers, `kv_cache_*` (handle layout matters), `memory_*` |
| **`not_applicable`** | **34** | 9% | `state_tree` family (pytrees, not tensors), `aot_*` (artifact export), `serialization` (save/load), `conformance` tests, LR `schedule`s (scalar functions), tokenizers (stateless string→int)... data combinators keep `partial` because `ShardedDataset` IS sharding-aware |

The classifier's design rationale lives in the module doc comment for
`_SHARDING_RULE_BY_CATEGORY`. Per-category mappings:

```
complete       → elementwise, scalar_math, numeric_helper, comparison,
                 logical, reduction, stable_reduction, rng, random_source,
                 random_mask, collective, sharding, quantize, quantization,
                 numerics, functional_optimizer_step, optimizer, rl_loss,
                 loss, rotary_embedding, position_encoding, transform,
                 extension

partial        → attention, loop_nest, model_layer, contraction,
                 projection, fused_epilogue, moe, moe_transport,
                 state_update, state_space, recurrent, stencil, pooling,
                 tensor_algebra, layout_transform, indexing,
                 segment_reduce, spectral, linalg_solver,
                 linalg_decomposition, sparse, normalization,
                 grad_transform, control_flow, memory, sort

not_applicable → state_tree, schedule, aot, serialization, conformance
                 (also data + tokenizer get `partial` via the
                 separate special-case, since `ShardedDataset` is
                 sharding-aware)
```

This means the next quality jump is split between two threads:
1. **Push the 156 `partial` entries to `complete`** as Phase G mesh
   integration lands and per-axis rules become testable.
2. **Continue the math/shape/dtype/batching hardening pass** through the
   long tail (~300 entries on each axis), category-by-category.

The `sharding_rule` axis is no longer the long-pole gate — `batching_rule`
(340), `transpose_rule` (313), and `math/shape/dtype` (299 each) now lead.
Backend_kernel (374, every entry) stays universally gated behind Phase G/H/I.

## Recommended Next Work

1. **Contract schema hardening**
   - Continue promoting selected primitives from "Python reference" to explicit
     contracts. This pass hardened S15 data/tokenizer non-differentiability
     declarations, reduction aliases, and memory primitive math/shape/dtype
     contracts. A follow-up pass promoted `linear_general`, `sgd`, and core
     S11 losses to `explicit_semantic` contracts with complete math, shape, and
     dtype/layout axes. The S7 `conv1d` reference now also carries an
     `explicit_semantic` contract plus VJP/JVP coverage.

2. **Autodiff coverage**
   - A focused pass now registers VJP/JVP rules for `linear_general`, `sgd`,
     and core differentiable S11 losses including MSE/MAE/Huber/SmoothL1,
     log-cosh, BCE, DDPM noise-prediction, score matching, and VLB reducers.
     Follow-up reasoning-model passes added RoPE-family, MLA, sparse attention,
     MoE transport, quantization STE, optimizer, reasoning-RL, cumulative
     extrema, and modern attention-family VJP/JVP coverage. Continue with the
     remaining classification/contrastive/sequence losses and lower-priority
     transform rules.

3. **Backend/lowering gates**
   - The registry now exposes `graph_ir_lowering` and `backend_kernel`
     metadata. A focused pass moved `linear_general`, `sgd`, and core S11
     losses into the Graph IR op catalog; subsequent passes promoted optimizer,
     attention, quantization, MoE, and RL names. Next work is turning the
     remaining `stub_required` entries into real Graph IR lowering paths and
     replacing reference-only attention kernels with backend fused kernels.

4. **Memory architecture sprint**
   - `memory_read` now has differentiable VJP/JVP coverage and the memory trio
     has explicit math/shape/dtype contracts. Continue with sharded memory
     layout rules, batching-axis overrides, and persistent memory state ABI
     integration.

5. **Reduction hardening**
   - `cummax` and `cummin` now have reverse-prefix autodiff coverage. Continue
     with batching, transpose, and sharding rules for reductions and cumulative
     extrema.

6. **S8 model-family expansion**
   - Build dedicated tiny Mamba/SSM, Hyena/FNet, Linformer/cosFormer,
     Griffin/Megalodon, JEPA, and Titans/Atlas memory examples using the
     now-shipped S10-S15 training stack.

7. **Reasoning-model backend hardening**
   - The public/reference surface now covers MoSA, DeepSeek-style MLA/NSA,
     MiniMax Lightning, Kimi Delta, Ling-style hybrid attention, mixed-precision
     optimizers, and PPO/GRPO/CISPO losses. The next jump is fused lowering,
     sharding rules for expert/recurrent state, and long-context performance
     tests.
