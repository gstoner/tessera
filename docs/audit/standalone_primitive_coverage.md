---
status: Informative
classification: Audit Dashboard
authority: Companion dashboard for `docs/audit/execution_roadmap.md` S1
last_updated: 2026-05-10
---

# Standalone Primitive Coverage

This dashboard tracks Tessera-native compiler primitive completeness. PyTorch,
JAX, and Flax are reference vocabularies only; they are not runtime
dependencies and do not define Tessera semantics.

The source of truth for this dashboard is
`python/tessera/compiler/primitive_coverage.py`. `op_catalog.py` remains the
source of truth for currently accepted operators; this dashboard can include
planned primitives without falsely marking them as supported.

For a fuller audit narrative, see
`docs/audit/primitive_coverage_state.md`.

## Contract Axes

Every primitive is tracked across these contract fields:

- mathematical semantics
- shape rule
- dtype/layout rule
- VJP
- JVP
- batching/vectorization rule
- transpose rule
- sharding rule
- masking/effect behavior
- lowering rule
- backend kernel
- tests

## Current Registry State

Generated from `all_primitive_coverages()` on 2026-05-10:

| Metric | Count |
|---|---:|
| Total tracked primitives | 373 |
| Existing / shipped partial entries | 373 |
| Planned entries | 0 |
| Contract-complete entries | 0 |

The current truth is that Tessera has a broad Python-reference surface, but no
primitive should be treated as contract-complete yet. Contract completion still
requires fully specified semantics, transform rules, lowering, backend kernels,
and tests for each primitive.

Current lowering metadata: `222` registered, `115` still `stub_required`, `32`
not applicable, and `4` missing.

Current backend metadata: `222` partial backend entries and `151`
reference-only entries.

Current contract schemas: `48` primitives are promoted to
`explicit_semantic`; the rest remain `explicit_partial` until more contract
axes are hardened.

## Milestone Groups

The dashboard groups primitives by the S-series sprint that ships them.

| Sprint / Group | Purpose | Example primitives |
|----------------|---------|--------------------|
| **S2** Tensor algebra | Baseline model graph expressivity | `reshape`, `pad`, `tile`, `dynamic_slice`, `dynamic_update_slice`, `cat`, `stack` |
| **S2** Indexing | Functional updates and retrieval | `scatter_add`, `scatter_reduce`, `top_k`, `sort`, `index_update`, `take` |
| **S2** Reductions | Loss/metric computation | `mean`, `var`, `max`, `min`, `argmax`, `cumsum`, `cumprod`, `cummax`, `cummin` |
| **S2** Stability primitives | Numerically safe ops | `logsumexp`, `log_softmax`, `log1p`, `expm1`, `softplus` |
| **S2** Scalar math | Activation, schedule, transcendental breadth | `exp`, `log`, `sqrt`, `rsqrt`, `pow`, `erf`, `lgamma` |
| **S2** Comparisons / logical | Boolean masking + control predicates | `eq`, `lt`, `logical_and`, `bitwise_xor` |
| **S2** Numeric helpers | Saturating + classification ops | `clamp`, `where`, `isnan`, `isfinite`, `sign` |
| **S3** State trees | Native model/state containers | `tree_flatten`, `tree_unflatten`, `tree_map`, `tree_reduce`, `state_filter`, `state_partition` |
| **S4** RNG | Reproducible stochastic compilation | `rng_key`, `rng_split`, `rng_fold_in`, `rng_bernoulli`, `rng_normal`, `rng_truncated_normal`, `rng_dirichlet` |
| **S5** Control flow + transforms | Recurrent + composable transforms | `scan`, `associative_scan`, `while_loop`, `fori_loop`, `cond`, `switch`, `vmap`, `pmap`, `vjp`, `jvp`, `remat`, `autocast`, `axis_index`, `axis_size` |
| **S6** Sharding + collectives | Compiler-visible SPMD placement and primitive collectives | `shard_map`, `named_sharding`, `partition_spec`, `psum`, `pmean`, `pmax`, `pmin`, `collective_permute`, `broadcast_to_axis` |
| **S7** Model layers | Standalone model authoring | `linear_general`, `conv1d`, `conv_transpose`, `group_norm`, `gru_cell`, `simple_rnn_cell`, `bidirectional_scan`, `max_pool`, `avg_pool`, `lora_linear` |
| **S7** Position encodings + attention | Modern transformer and reasoning-model building blocks | `rope`, `alibi`, `ntk_rope`, `multi_head_attention`, `gqa_attention`, `mqa_attention`, `mla_decode`, `mla_decode_fused`, `deepseek_sparse_attention`, `attn_sliding_window`, `attn_compressed_blocks`, `attn_top_k_blocks`, `lightning_attention`, `gated_attention`, `hybrid_attention`, `gated_deltanet`, `kimi_delta_attention`, `modified_delta_attention` |
| **S7** Memory | Titans/Atlas-style learned memory | `memory_read`, `memory_write`, `memory_evict` |
| **S8** Tiny model conformance | Aggregate model-family suite in `examples.conformance.s8_tiny_models` | `tiny_diffusion_conformance`, `tiny_recurrent_conformance`, `tiny_attention_conformance`, `tiny_training_step_conformance` |
| **S9** Numerics + quantization | Mixed precision + quant flows | `quantize_fp8`, `dequantize_fp8`, `quantize_fp4`, `dequantize_fp4`, `quantize_int8`, `dequantize_int8`, `quantize_int4`, `dequantize_int4`, `fake_quantize`, `calibration_observer`, `grad_scaler_step`, `autocast` |
| **S10** Optimizers + schedules | Functional training step | `sgd`, `momentum`, `adam`, `adamw`, `adafactor`, `lion`, `muon`, `lamb`, `cosine_lr`, `cosine_warmup_lr`, `cyclical_lr`, `chained_schedule`, `clip_grad_norm`, `ema_update`, `polyak_avg` |
| **S11** Losses | Training criteria | `mse_loss`, `mae_loss`, `huber_loss`, `cross_entropy_loss`, `kl_divergence`, `info_nce_loss`, `triplet_loss`, `focal_loss`, `ddpm_noise_pred_loss`, `score_matching_loss`, `ctc_loss`, `seq2seq_loss`, `ppo_policy_loss`, `grpo_policy_loss`, `cispo_policy_loss` |
| **S12** Serialization | Save / load / migrate / shard | `save_state`, `load_state`, `save_sharded`, `load_sharded`, `state_migration`, `partial_state_load` |
| **S13** Custom-primitive API | User-defined primitives + kernels | `custom_primitive`, `custom_call`, `custom_vjp`, `custom_jvp`, `custom_batching`, `custom_lowering` |
| **S14** AOT + cache | Persistent JIT cache + AOT export | `aot_export`, `aot_load`, `stablehlo_export`, `gguf_export`, `safetensors_export`, `compilation_cache` |
| **S15** Data pipeline | Native dataset + tokenizer surface (in-scope per S0) | `dataset_map`, `dataset_filter`, `dataset_batch`, `dataset_shuffle`, `sharded_dataset`, `iterable_dataset`, `dataset_checkpoint`, `tokenizer_byte`, `tokenizer_bpe`, `tokenizer_unigram`, `tokenizer_sentencepiece_compat` |

## Model-Family Coverage Tags

The registry tags primitives by the model families they unblock:

- diffusion / DiT / U-Net
- RNN / xLSTM
- Mamba / SSM
- Hyena / FNet / spectral models
- Linformer / cosFormer
- Griffin / Megalodon
- Titans / Atlas memory
- JEPA
- MoSA / MoE sparse attention
- DeepSeek-style MLA / Native Sparse Attention
- MiniMax Lightning / long-context hybrid attention
- Kimi Delta / Modified Delta Attention
- Ling-style MLA + Lightning hybrid attention
- reasoning RL post-training loops

## S0 Scope Decision (2026-05-10)

The data pipeline is **in scope** (S15). Tessera owns its own dataset,
batching, sharding-of-data, and tokenization surfaces. `tf.data`,
`torch.utils.data`, `grain`, `tiktoken`, `tokenizers`, and `sentencepiece`
are reference vocabularies only — nothing in the runtime imports them.

The training step is **in scope** (S10 optimizers, S11 losses, S12
checkpointing). Custom-primitive authoring is **in scope** (S13). AOT export
and persistent compilation cache are **in scope** (S14).

## Current S1 Result

S1 is complete when the registry and tests exist, not when every primitive is
implemented. The current result is intentionally a mixed dashboard:

- Existing Tessera operators are imported from `OP_SPECS` as partial coverage
  entries; the registry now consults `tessera.autodiff.vjp._VJPS` and
  `tessera.autodiff.jvp._JVPS` so any op with a registered VJP/JVP correctly
  shows `complete`.
- All currently tracked standalone compiler primitives have at least a
  Python-reference surface; none remain falsely listed as planned.
- The snapshot includes lowering/backend gates so reference-only primitives are
  visually distinct from Graph IR-lowered primitives.
- Reasoning-model support now appears as first-class registry coverage for
  RoPE/MLA/NSA, MoE dispatch/combine, quantization STE, optimizers, and
  PPO/GRPO/CISPO helpers.
- Missing contract axes remain visible until each primitive has semantics,
  transform rules, lowering, backend coverage, and tests.

The generated table below is a checked-in snapshot of the registry output for
the high-risk S1 entries and milestone sentinels. Tests compare these rows to
`render_markdown(...)` so the dashboard cannot drift silently.

<!-- BEGIN GENERATED PRIMITIVE COVERAGE SNAPSHOT -->
| Primitive | Category | Status | Existing op | Lowering gate | Backend gate | Missing contracts | Model families |
|-----------|----------|--------|-------------|---------------|--------------|-------------------|----------------|
| `matmul` | loop_nest | partial | yes | registered | partial | sharding_rule, backend_kernel | - |
| `permute` | tensor_algebra | partial | yes | registered | partial | jvp, batching_rule, transpose_rule, sharding_rule, backend_kernel | - |
| `collective_permute` | collective | partial | yes | stub_required | reference_only | batching_rule, backend_kernel | all |
| `scan` | control_flow | partial | yes | stub_required | reference_only | math_semantics, shape_rule, dtype_layout_rule, batching_rule, transpose_rule, sharding_rule, backend_kernel | all |
| `selective_ssm` | state_space | partial | yes | missing | reference_only | math_semantics, shape_rule, dtype_layout_rule, jvp, batching_rule, sharding_rule, backend_kernel | Mamba/SSM |
| `dataset_map` | data | partial | yes | not_applicable | reference_only | batching_rule, sharding_rule, backend_kernel | all |
| `tokenizer_bpe` | tokenizer | partial | yes | not_applicable | reference_only | batching_rule, sharding_rule, backend_kernel | all |
<!-- END GENERATED PRIMITIVE COVERAGE SNAPSHOT -->

Regenerate/check the full table programmatically with:

```python
from tessera.compiler.primitive_coverage import render_markdown

print(render_markdown())
```

## Open Gap Plan

The next work is contract hardening, not broad name collection:

- **Contract hardening:** promote math semantics, shape rules, and dtype/layout
  rules for high-use S2, S5, S7, S10, and S11 primitives. The first focused
  semantic pass now covers `conv1d`, `linear_general`, `sgd`, data/tokenizer
  declarations, reduction aliases, memory primitives, core S11 losses,
  minimum optimizers, and modern attention-family wrappers.
- **Transform coverage:** close VJP/JVP, batching, transpose, and sharding gaps
  starting with reductions, model layers, losses, `memory_read`, scans, and
  collectives. Focused passes now cover `linear_general`, `sgd`, core
  differentiable S11 losses, `memory_read`, `cummax`/`cummin`, RoPE/MLA/NSA,
  quantization STE, MoE dispatch/combine, optimizers, and reasoning RL losses.
- **Lowering gates:** convert `stub_required` S5-S15 Python-reference
  primitives into real Graph IR lowering paths. The first focused pass now
  registers Graph IR names for `linear_general`, `sgd`, core S11 losses,
  optimizer ops, attention-family wrappers, quantization, MoE transport, and
  RL helpers.
- **Backend gates:** distinguish CPU-reference behavior, Graph IR-lowered
  behavior, and backend-kernel-ready behavior for each primitive group. The
  major remaining reasoning-model gap is fused backend kernels for MLA, Native
  Sparse Attention, Lightning, Delta/Kimi recurrence, and optimizer steps.
- **Memory architecture:** extend `memory_read`, `memory_write`, and
  `memory_evict` with sharded layout rules, batching-axis overrides, and a
  persistent memory state ABI; differentiable `memory_read` VJP/JVP is now
  present, while mutation-style writes/evictions remain state effects.
