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
| Total tracked primitives | 362 |
| Existing / shipped partial entries | 362 |
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
| S7 | Model layers: LinearGeneral, Einsum, Conv1d, ConvTranspose1d, LoRA, GroupNorm, InstanceNorm, WeightNorm, SpectralNorm, pooling, SimpleRNN/GRU helpers, ALiBi/NTK RoPE, GQA/MQA/MLA attention wrappers. |
| S8 | Tiny conformance smoke tests: recurrent, diffusion-like, attention-like, plus training-step conformance using data/loss/optimizer/checkpoint. |
| S9 | Quantization/numerics: int8/int4 quant/dequant, fake quant, calibration observer, grad scaler step. |
| S10 | Optimizers/schedules/grad transforms: SGD, Adam, AdamW, Adafactor, Lion, Muon, LAMB, schedules, clipping, EMA, Polyak, transform chain. |
| S11 | Losses: regression, classification, distribution, contrastive, diffusion, sequence, CTC reference. |
| S12 | State serialization/checkpointing: save/load state, checksums, partial collection load, migration, mock sharded save/load. |
| S13 | Custom primitive API: custom primitive/call, VJP/JVP/batching metadata, per-target lowering registration. |
| S14 | AOT/cache: reference AOT export/load, StableHLO text, GGUF metadata, safetensors-like export, persistent compilation cache. |
| S15 | Data/tokenizers: eager Dataset combinators, sharded/iterable datasets, checkpoint/restore, deterministic shuffle, byte/vocab tokenizers. |

## Newly Promoted

The prior seven planned entries are now shipped as Python-reference primitives:

| Category | Promoted entries |
|---|---|
| Memory | `memory_read`, `memory_write`, `memory_evict` |
| Reductions | `max`, `min`, `cummax`, `cummin` |

The memory entries unblock Titans/Atlas-style active-memory conformance at the
reference level. The reduction entries close the small API alias/coverage gaps
around the broader reduction surface that already includes `amax`, `amin`,
`argmax`, `argmin`, `cumsum`, `cumprod`, `mean`, `var`, and related helpers.

## Largest Remaining Contract Gaps

Across the registry, the broad gaps are compiler-contract gaps rather than
missing Python API names:

| Contract axis | Missing / partial count |
|---|---:|
| Mathematical semantics | 339 |
| Shape rule | 339 |
| Dtype/layout rule | 339 |
| Batching rule | 362 |
| Transpose rule | 312 |
| Sharding rule | 362 |
| Backend kernel | 362 |
| JVP | 298 |
| VJP | 209 |
| Tests | 193 |
| Lowering rule | 169 |
| Masking/effect rule | 27 |

This means the next quality jump is not adding more names. It is hardening the
contract axes for the primitives already present.

## Recommended Next Work

1. **Contract schema hardening**
   - Continue promoting selected primitives from "Python reference" to explicit
     contracts. This pass hardened S15 data/tokenizer non-differentiability
     declarations, reduction aliases, and memory primitive math/shape/dtype
     contracts.

2. **Autodiff coverage**
   - Prioritize VJP/JVP rules for S7 layers, S10 optimizers where differentiable,
     S11 losses, and S15 data/tokenizer non-differentiability declarations.

3. **Backend/lowering gates**
   - The registry now exposes `graph_ir_lowering` and `backend_kernel`
     metadata. Next work is turning `stub_required` entries into real Graph IR
     lowering paths.

4. **Memory architecture sprint**
   - Extend `memory_read`, `memory_write`, and `memory_evict` with sharded
     memory layout rules, differentiable read VJP/JVP, and persistent memory
     state ABI integration.

5. **Reduction hardening**
   - Add JVP/batching/transpose/sharding rules for `max`, `min`, `cummax`, and
     `cummin`; cumulative extrema still need full autodiff decisions.

6. **S8 model-family expansion**
   - Build dedicated tiny Mamba/SSM, Hyena/FNet, Linformer/cosFormer,
     Griffin/Megalodon, JEPA, and Titans/Atlas memory examples using the
     now-shipped S10-S15 training stack.
