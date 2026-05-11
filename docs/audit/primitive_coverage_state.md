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
| Total tracked primitives | 373 |
| Existing / shipped partial entries | 373 |
| Planned entries | 0 |
| Contract-complete entries | 0 |

The registry intentionally reports shipped work as **partial** until the full
compiler contract is complete. This avoids confusing "has a Python reference"
with "is fully lowered and transform-complete."

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

| Contract axis | Missing / partial count |
|---|---:|
| Mathematical semantics | 325 |
| Shape rule | 325 |
| Dtype/layout rule | 325 |
| Batching rule | 373 |
| Transpose rule | 321 |
| Sharding rule | 373 |
| Backend kernel | 373 |
| JVP | 221 |
| VJP | 137 |
| Tests | 195 |
| Lowering rule | 147 |
| Masking/effect rule | 27 |

This means the next quality jump is not adding more names. It is hardening the
contract axes for the primitives already present.

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
