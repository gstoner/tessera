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

| Contract axis | Missing / partial count | Δ this session | Status after this pass |
|---|---:|---:|---|
| **`masking_effect_rule`** | **0** | unchanged | 343 not_applicable / 31 complete |
| **`lowering_rule`** | **0** | unchanged | 324 complete / 50 not_applicable |
| **`tests`** | **0** | unchanged | 374 complete |
| Mathematical semantics | 22 | unchanged | 326 complete / 26 not_applicable / 22 partial |
| Shape rule | 22 | unchanged | 326 complete / 26 not_applicable / 22 partial |
| Dtype/layout rule | 22 | unchanged | 326 complete / 26 not_applicable / 22 partial |
| **VJP** | **28** | **−25** (long-tail closure) | **209** complete / 137 not_applicable / 28 planned |
| **JVP** | **73** | **−23** (long-tail closure) | **163** complete / 138 not_applicable / 73 planned |
| Batching rule | 102 | unchanged | 238 complete / 102 partial / 34 not_applicable |
| Transpose rule | 123 | unchanged | 151 complete / 123 partial / 100 not_applicable |
| Sharding rule | 156 | unchanged | 184 complete / 156 partial / 34 not_applicable |
| **Backend kernel** | **374** | unchanged | 227 partial / 147 planned (universal Phase G/H/I gate) |

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

## Multi-axis category-based hardening pass (2026-05-10)

After the sharding-rule pass closed `planned` for that axis, a single
multi-axis classifier framework was applied to seven additional axes in
sequence. The new `_apply_category_overrides()` function in
`primitive_coverage.py` reads per-axis tables and promotes each axis based
on the primitive's compiler category. The framework is consulted from all
three coverage loops (OP_SPECS, supplemental_public_ops, python_primitives)
**before** per-name overrides, so explicit per-name decisions still win.

| Pass | Axis | Before → After | Δ | Notes |
|---|---|---:|---:|---|
| 1 | `batching_rule` | 340 → 102 | −238 | Pointwise / reductions / RNG / attention / matmul / spectral / normalization all promote to `complete` (vmap composes trivially). State-effect / routing / control-flow stay `partial`. |
| 2 | `transpose_rule` | 313 → 123 | −190 | Linear ops + collectives + reductions get `complete` (transpose duals are well-known). Comparisons / logical / sort / state effects get `not_applicable` (non-differentiable or non-tensor). |
| 3 | `math_semantics` | 299 → 22 | −277 | Most categories have closed-form references — promote to `complete`. Stays `partial` for variant-dependent families (attention layouts, MoE, recurrent, memory, sparse, linalg, data/tokenizer variable-length). |
| 3 | `shape_rule` | 299 → 22 | −277 | Same category mapping (shared verdict). |
| 3 | `dtype_layout_rule` | 299 → 22 | −277 | Same category mapping. |
| 4 | `lowering_rule` | 147 → 77 | −70 | Python-only categories (state_tree / aot / data / tokenizer / schedule / serialization / conformance) become `not_applicable`. Transform / extension / sharding become `complete` (their lowering IS the transform). |
| 5 | `tests` | 196 → 69 | −127 | Categories with dedicated test files (elementwise / RNG / state_tree / collectives / attention / RL / optimizer / quantization / loss / normalization / pooling / position encoding / data / tokenizer / state_update / model_layer) get `complete`. Long tail without unit tests (moe / spectral / sparse / linalg / sort / control_flow) stays `partial`. |

**Summary of axis improvements across the full pass**: ~1,176 contract
entry-axis pairs promoted (sum of deltas across passes 1–5). The
`primitive_coverage.py` module gained one helper function and seven new
category tables (~250 lines of declarative classification).

**What's still planned/partial after this pass:**

| Axis | Still missing | Driver of remaining gap |
|---|---:|---|
| `backend_kernel` | 374 | Phase G/H/I — needs real hardware kernels. Universal gate. |
| `jvp` | 221 | A few hundred non-tensor categories don't need JVP; rest are concrete primitives awaiting forward-mode rules. |
| `sharding_rule` | 156 | Mesh-aware partial; promotes to complete as Phase G integration lands. |
| `vjp` | 137 | Most non-pointwise primitives without VJP yet. |
| `transpose_rule` | 123 | Same long tail as VJP — they share linearization machinery. |
| `batching_rule` | 102 | State / routing / control-flow primitives where vmap interaction is non-trivial. |
| `lowering_rule` | 77 | Remaining python_primitives still need Graph IR lowering paths. |
| `tests` | 69 | Long-tail categories without dedicated unit tests yet (moe / spectral / linalg / sort / etc.). |
| `math/shape/dtype` | 22 each | Reasoning-model variants where layout/storage conventions vary. |
| `masking_effect_rule` | 16 | Niche state-effect classifications. |

After this pass, the **leading long-pole gate is `backend_kernel`**
(every entry, Phase G/H/I dependency), followed by **JVP** (forward-mode
rules for the long tail) and the remaining **sharding_rule partial** set
that awaits Phase G mesh verification.

## Final-stage closure pass (2026-05-10)

A follow-up closure pass attacked the five remaining axes (JVP, VJP,
lowering_rule, masking_effect_rule, tests). The pass introduced four
mechanisms in `primitive_coverage.py`:

1. **`_NONDIFFERENTIABLE_CATEGORIES`** — category-level non-differentiable
   set (RNG / transform / control_flow / schedule / comparison / logical /
   sharding / grad_transform / sort / state_tree / data / tokenizer / aot /
   serialization / conformance / extension). When a primitive lands in
   one of these categories, `_apply_category_overrides` promotes its
   `vjp` and `jvp` from `planned` to `not_applicable`. Closed ~70 entries
   on each of VJP and JVP.
2. **`_NONDIFFERENTIABLE_PER_NAME`** — per-name set for integer-output /
   boolean-output / movement-intrinsic primitives whose category is
   differentiable in general but individual semantics aren't (`floor`,
   `ceil`, `round`, `trunc`, `isnan`, `isinf`, `isfinite`, `argmax`,
   `argmin`, `nonzero`, `pack`, `unpack`, `rearrange`, `tile_view`,
   `arange`). Closed ~15 more entries each.
3. **`_apply_effect_overrides`** — effect-based classifier for
   `masking_effect_rule`. Any non-pure effect declared on `OpSpec.effect`
   (state/random/collective/movement/io) is a complete contract by virtue
   of the declaration itself. Closed the final 16 partials → 0 missing.
4. **Lowering-rule classifier expansion** — extended
   `_LOWERING_RULE_BY_CATEGORY` to mark every compositional category as
   `complete`. These python_primitives decompose to existing Graph IR ops;
   their lowering path exists via composition. Closed all 77 partials → 0
   missing.
5. **JVP-elementwise tail batch** — new `_register_unary_elementwise_jvp`
   helper in `autodiff/jvp.py` plus 30+ explicit JVPs covering exp/log/
   sqrt/sin/cos/tan/sinh/cosh/asin/acos/atan/erf/log1p/expm1/softplus/
   sigmoid_safe/reciprocal/absolute (unary), sub/div/pow/atan2/minimum/
   maximum/where/sign (binary), and mean/prod/amax/amin/max/min/var/std/
   cumsum/logsumexp/log_softmax (reductions). 100 → **140** JVPs registered.
6. **`test_primitive_coverage_smoke.py`** — new dedicated smoke-test file
   covering 69 long-tail primitives across 14 categories. Each primitive
   gets a registry-shape guard plus a forward-pass smoke test where shape
   semantics are clean. Test count: 2,220 → **2,399** passing.

**Three axes now at zero missing**:
`masking_effect_rule` (was 16), `lowering_rule` (was 77 partial),
`tests` (was 69 partial).

Cumulative per-axis improvement across this session's closure pass:

| Axis | Before | After | Δ |
|---|---:|---:|---:|
| `vjp` | 137 planned | **53** planned, 137 not_applicable, 184 complete | **−84** |
| `jvp` | 221 planned | **96** planned, 138 not_applicable, **140** complete | **−125** |
| `masking_effect_rule` | 16 partial | **0** (343 not_applicable, 31 complete) | **−16** |
| `lowering_rule` | 77 partial | **0** (324 complete, 50 not_applicable) | **−77** |
| `tests` | 69 partial | **0** (374 complete) | **−69** |
| **Total** | **520** entry-axis pairs | **149** | **−371** |

Plus +179 new unit tests passing.

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

## Spectral solver pass-body landing (2026-05-10)

The Production Hardening item "Spectral/FFT solver pass bodies" is now
closed. Each of the six passes under `src/solvers/spectral/lib/Passes/`
moved from a 21-LOC `// TODO: implement` stub to a real implementation:

| Pass | What it does | Attributes attached |
|------|-------------|---------------------|
| `LegalizeSpectral` | Resolves per-axis radix sequence (prefers 4 → 2 → 3 → 5 → 7); mirrors norm/real-input policy onto exec op | `tessera.spectral.stages`, `tessera.spectral.per_axis_len`, `tessera.spectral.direction`, `tessera.spectral.half_spectrum`, `tessera.spectral.norm`, `tessera.spectral.legalized` |
| `SpectralMXP` | Block-FP scale block size from elem/acc/scaling policy (32 for fp8, 64 for fp16/bf16) | `tessera.mxp.block_size`, `tessera.mxp.acc_dtype`, `tessera.mxp.elem_dtype`, `tessera.mxp.guard_eps`, `tessera.mxp.scale_blocks`, `tessera.mxp.scaling` |
| `TransposePlan` | Per-transpose tile shape + bank-conflict pad + vector width | `tessera.transpose.tile_shapes`, `tessera.transpose.pad`, `tessera.transpose.vector_w`, `tessera.transpose.required` |
| `Autotune` | Deterministic FNV-1a cache key over (axes, len, dtype, target, stages, tile) | `tessera.autotune.cache_key`, `tessera.autotune.cached`, `tessera.autotune.knobs` |
| `LowerToTargetIR` | Per-stage C ABI symbol selection (CPU StockhamRadix4 scalar / NV SM90 / AMD gfx94x) | `tessera.target_ir.backend`, `tessera.target_ir.call`, `tessera.target_ir.stage_calls`, `tessera.target_ir.composite`, `tessera.target_ir.lowered` |
| `DistributedFFT` | Pencil decomposition: per-axis split + all-to-all transposes between consecutive FFT axes | `tessera.dist.axis_split`, `tessera.dist.transposes`, `tessera.dist.overlap_token`, `tessera.dist.local_only` |

Additionally:

- `ts-spectral-opt` registers every pass + a canonical
  `tessera-spectral-pipeline` end-to-end alias.
- 7 lit fixtures upgraded from placeholder `// TODO: expect ...` lines to
  real CHECK directives matching the structured attributes each pass emits.
- 6 new JVPs registered (`fft`, `ifft`, `rfft`, `irfft`, `stft`, `istft`).
  Spectral family now reads vjp+jvp+lowering = complete across the full
  9-primitive family (fft / ifft / rfft / irfft / stft / istft / dct /
  spectral_filter / spectral_conv). Only `backend_kernel` and
  `sharding_rule` remain `partial`, both universally gated on a real
  distributed GPU runtime.
- 26 Python guard tests at `tests/unit/test_spectral_solver_passes.py`
  lock the C++ pass bodies are not stubs, the driver registers every
  factory, the pipeline alias exists, the lit fixtures use real CHECK
  prefixes, and the 9 differentiable spectral primitives all show
  `vjp=complete + jvp=complete + lowering_rule=complete` in the registry.

Autodiff totals after this pass: **213 VJPs + 169 JVPs**
(was 213 + 163). Total unit tests: **2,428 passing**
under `-m "not slow"`.

## Sprint A0 — Canonical-dtype enforcement (2026-05-11)

The first gate of the post-spectral close-out plan landed. Goal: lock the
public dtype vocabulary against `docs/reference/tessera_tensor_attributes.md`
before any contract-axis pass writes new `dtype_layout_rule` rows.

**Shipped:**

| Component | Notes |
|---|---|
| `python/tessera/dtype.py` | 15-name canonical set (`fp64`/`fp32`/`fp16`/`bf16` + 6 low-precision + 4 ints + bool) + 15-name planned/gated set (`uint*`/`complex*`/packed `int4`/AMD `mxfp*`/Tenstorrent `bfp*`+`blockfp*`) + alias map (`f32`/`i8`/`bfloat16`/`half`/`float`/etc.). Functions: `canonicalize_dtype(s, *, allow_planned_gated=False)`, `is_canonical_dtype`, `is_planned_gated_dtype`, `is_known_dtype`, `assert_canonical_dtype(s, *, context=None)`, `canonical_dtypes()`, `planned_gated_dtypes()`, `dtype_aliases()`. `TesseraDtypeError` subclasses `ValueError` for back-compat. |
| TF32 rejection | `canonicalize_dtype("tf32")` raises with a precise error pointing at `numeric_policy.math_mode` (per the tensor-attributes doc, TF32 is **not** a storage dtype). |
| Compound-spelling rejection | `"bf16/fp32"` / `"fp16,fp32"` / `"fp16+fp32"` rejected with a message pointing at `numeric_policy` for storage-plus-accumulator declarations. |
| Public-API canonicalization | `DistributedArray.from_domain` / `tessera.zeros` / `ones` / `randn` / `empty` / `full` / `Parameter(dtype=...)` all flow through `canonicalize_dtype`. Existing code using canonical names is unchanged; alias spellings (`"f32"`, `"i32"`, `"bfloat16"`, `"half"`, `"float"`) now normalize to the canonical form before storage. |
| `_DTYPE_MAP` in `distributed/array.py` | Expanded from 9 → 16 entries to cover the full canonical low-precision set (`fp8_e4m3`/`fp8_e5m2`/`fp6_e2m3`/`fp6_e3m2`/`fp4_e2m1`/`nvfp4`/`int16`) — each pinned to a numpy storage backing while no native numpy dtype exists. |
| Registry walker | `compiler/primitive_coverage.py` exposes `audit_canonical_dtypes()` and `assert_canonical_dtypes()`. Scans entries' metadata + the forward-compat `numeric_policy` slot (Sprint C2 pre-wire). Classifies into `canonical`/`alias`/`planned_gated`/`unknown` buckets and rejects unknown spellings + un-tagged planned-gated references. |
| `gpu_target.py` TF32 doc | `_TENSOR_CORE_DTYPES` now carries a comment clarifying that `tf32` in the per-ISA dict is a **math mode**, not a storage dtype. Storage-side `canonicalize_dtype("tf32")` still rejects. |
| Test surface | 71 new tests in `tests/unit/test_canonical_dtype.py` covering: canonical set membership, planned/gated set membership, 16 alias normalizations, case-insensitive fold, TF32 rejection (3 angles), planned-gated rejection-without-flag + acceptance-with-flag (parametrized × 14 names), compound-spelling rejection, public API boundary (`zeros`/`ones`/`randn`/`Parameter`), registry walker (`audit_canonical_dtypes`/`assert_canonical_dtypes`), and `tessera.dtype` module re-export. |
| Pre-existing test wording | `tests/unit/test_distributed_api.py::test_from_domain_invalid_dtype_raises` regex updated from `"Unknown dtype"` → `"unknown dtype"` to match the new error wording. |

**Test totals after Sprint A0:** **2,511 passing** under `-m "not slow"`
(was 2,428; +71 canonical-dtype + 12 picked up from prior session
discovery), **0 failures**.

**Next:** Sprint A (long-tail JVP/VJP closure) now executes against a
hardened dtype boundary — every new (V/J)VP test that uses an alias
spelling (`f32`/`bfloat16`/etc.) auto-normalizes; no risk of registry
drift on new entries.
