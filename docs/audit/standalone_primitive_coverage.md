---
status: Informative
classification: Audit Dashboard
authority: Companion dashboard for `docs/audit/roadmap/ROADMAP_AUDIT.md` S1
last_updated: 2026-07-12 (registry actions reconciled)
---

# Standalone Primitive Coverage

This dashboard tracks Tessera-native compiler primitive completeness. PyTorch,
JAX, and Flax are reference vocabularies only; they are not runtime
dependencies and do not define Tessera semantics.

The source of truth for this dashboard is
`python/tessera/compiler/primitive_coverage.py`. `op_catalog.py` remains the
source of truth for currently accepted operators; this dashboard can include
planned primitives without falsely marking them as supported.

For historical audit narrative, see `docs/audit/coverage/COVERAGE_AUDIT.md`.
For current cross-layer support state, use
`docs/audit/generated/support_table.md` and the registry source.

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

The registry is intentionally rendered by code rather than maintained by hand.
Use `python -m tessera.compiler.audit support_table --check` for the
cross-layer support table and `render_markdown()` from
`tessera.compiler.primitive_coverage` when a primitive-only table is needed.

The current truth is that Tessera has a broad Python-reference surface and a
growing set of native/runtime lanes, but primitive rows should not be treated
as compiler-complete until their layered contract is complete. Contract
completion still requires fully specified semantics, transform rules, lowering,
backend kernels or explicit reference-only status, and tests for each
primitive. Avoid copying registry totals into prose unless a test owns the
snapshot.

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
| `matmul` | loop_nest | partial | yes | registered | partial | backend_kernel | - |
| `permute` | tensor_algebra | partial | yes | registered | partial | backend_kernel | - |
| `collective_permute` | collective | partial | yes | registered | reference_only | backend_kernel | all |
| `scan` | control_flow | partial | yes | registered | reference_only | - | all |
| `selective_ssm` | state_space | partial | yes | registered | partial | sharding_rule, backend_kernel | Mamba/SSM |
| `dataset_map` | data | partial | yes | runtime_only | reference_only | - | all |
| `tokenizer_bpe` | tokenizer | partial | yes | runtime_only | reference_only | - | all |
<!-- END GENERATED PRIMITIVE COVERAGE SNAPSHOT -->

Regenerate/check the full table programmatically with:

```python
from tessera.compiler.primitive_coverage import render_markdown

print(render_markdown())
```

## Open Gap Plan

The broad contract/lowering pass is closed. Current registry truth across 480
primitives is:

- **Semantics and basic transforms:** math, shape, dtype/layout, batching,
  transpose, lowering, effect/masking, and tests have zero open rows. The five
  late speculative/packed-attention entries (`spec_accept`, sampled/tree
  variants, `target_verify`, and `varlen_sdpa`) now carry explicit semantics,
  ODS verifier coverage, and focused tests.
- **Graph IR lowering metadata:** **425 registered / 10 host-materialized / 45
  runtime-only / 0 stub-required / 0 missing**. Clifford and EBM dialect/decomposition rows are
  registered; image preprocessing and EDM conditioning lower compositionally;
  scalar diffusion schedule constructors are explicitly `host_materialized`.
- **Autodiff frontier:** 29 VJP and 30 JVP rows remain planned. These are
  concentrated in visual-complex/conformal maps, solver helpers, EDM scalars,
  cache transactions, `moe_swiglu_block`, `score_combine`, and `varlen_sdpa`.
  Add rules only with numerical derivative tests; discrete verification ops
  are explicitly non-differentiable rather than left planned.
- **Sharding frontier:** 43 rows remain partial. Priority groups are
  reasoning/sparse attention and MoE transport, state-space recurrence,
  spectral transforms, sparse COO, and linear-algebra solver/decomposition
  operations. Promotion requires mock-mesh execute/compare evidence or real
  multi-device proof; backend availability alone is insufficient.
- **Backend frontier:** the legacy aggregate axis remains 372 partial, 9
  planned, and 99 `no_kernel_required`. Readiness decisions must use per-target
  `backend_manifest` entries and the op×target proof ladder in
  `op_target_conformance.md`; the aggregate axis is not permission to claim a
  universal kernel. Reasoning-model fused kernels and optimizer steps remain
  important performance work, but several already have executing CPU/x86 or
  ROCm lanes and must be reported per target.
- **Memory architecture:** closed for the current contract. `memory_read`,
  `memory_write`, and `memory_evict` have batching and sharding rules plus the
  persistent state ABI. `memory_read` is differentiable; writes and evictions
  remain intentionally stateful/non-differentiable.
