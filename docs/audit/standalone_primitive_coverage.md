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

## Milestone Groups

The dashboard groups primitives by the S-series sprint that ships them.

| Sprint / Group | Purpose | Example primitives |
|----------------|---------|--------------------|
| **S2** Tensor algebra | Baseline model graph expressivity | `reshape`, `pad`, `tile`, `dynamic_slice`, `dynamic_update_slice`, `cat`, `stack` |
| **S2** Indexing | Functional updates and retrieval | `scatter_add`, `scatter_reduce`, `top_k`, `sort`, `index_update`, `take` |
| **S2** Reductions | Loss/metric computation | `mean`, `var`, `argmax`, `cumsum`, `cumprod` |
| **S2** Stability primitives | Numerically safe ops | `logsumexp`, `log_softmax`, `log1p`, `expm1`, `softplus` |
| **S2** Scalar math | Activation, schedule, transcendental breadth | `exp`, `log`, `sqrt`, `rsqrt`, `pow`, `erf`, `lgamma` |
| **S2** Comparisons / logical | Boolean masking + control predicates | `eq`, `lt`, `logical_and`, `bitwise_xor` |
| **S2** Numeric helpers | Saturating + classification ops | `clamp`, `where`, `isnan`, `isfinite`, `sign` |
| **S3** State trees | Native model/state containers | `tree_flatten`, `tree_unflatten`, `tree_map`, `tree_reduce`, `state_filter`, `state_partition` |
| **S4** RNG | Reproducible stochastic compilation | `rng_key`, `rng_split`, `rng_fold_in`, `rng_bernoulli`, `rng_normal`, `rng_truncated_normal`, `rng_dirichlet` |
| **S5** Control flow + transforms | Recurrent + composable transforms | `scan`, `associative_scan`, `while_loop`, `fori_loop`, `cond`, `switch`, `vmap`, `pmap`, `vjp`, `jvp`, `remat`, `autocast`, `axis_index`, `axis_size` |
| **S6** Sharding + collectives | Compiler-visible SPMD placement and primitive collectives | `shard_map`, `named_sharding`, `partition_spec`, `psum`, `pmean`, `pmax`, `pmin`, `collective_permute`, `broadcast_to_axis` |
| **S7** Model layers | Standalone model authoring | `linear_general`, `conv1d`, `conv_transpose`, `group_norm`, `gru_cell`, `simple_rnn_cell`, `bidirectional_scan`, `max_pool`, `avg_pool`, `lora_linear` |
| **S7** Position encodings + attention | Modern transformer building blocks | `rope`, `alibi`, `ntk_rope`, `multi_head_attention`, `gqa_attention`, `mqa_attention`, `mla_decode` |
| **S7** Memory | Titans/Atlas-style learned memory | `memory_read`, `memory_write`, `memory_evict` |
| **S8** Tiny model conformance | Standalone family smoke tests | `tiny_diffusion_conformance`, `tiny_recurrent_conformance`, `tiny_attention_conformance`, `tiny_training_step_conformance` |
| **S9** Numerics + quantization | Mixed precision + quant flows | `quantize_int8`, `dequantize_int8`, `quantize_int4`, `dequantize_int4`, `fake_quantize`, `calibration_observer`, `grad_scaler_step`, `autocast` |
| **S10** Optimizers + schedules | Functional training step | `sgd`, `adam`, `adamw`, `adafactor`, `lion`, `muon`, `lamb`, `cosine_lr`, `cosine_warmup_lr`, `cyclical_lr`, `chained_schedule`, `clip_grad_norm`, `ema_update`, `polyak_avg` |
| **S11** Losses | Training criteria | `mse_loss`, `mae_loss`, `huber_loss`, `cross_entropy_loss`, `kl_divergence`, `info_nce_loss`, `triplet_loss`, `focal_loss`, `ddpm_noise_pred_loss`, `score_matching_loss`, `ctc_loss`, `seq2seq_loss` |
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
  entries; the registry now consults `tessera.autodiff.vjp._VJPS` so any op
  with a registered VJP correctly shows `vjp = complete`.
- Missing standalone compiler primitives are planned entries.
- Missing contract axes remain visible until each primitive has semantics,
  transform rules, lowering, backend coverage, and tests.

The generated table below is a checked-in snapshot of the registry output for
the high-risk S1 entries and milestone sentinels. Tests compare these rows to
`render_markdown(...)` so the dashboard cannot drift silently.

<!-- BEGIN GENERATED PRIMITIVE COVERAGE SNAPSHOT -->
| Primitive | Category | Status | Existing op | Missing contracts | Model families |
|-----------|----------|--------|-------------|-------------------|----------------|
| `matmul` | loop_nest | partial | yes | math_semantics, shape_rule, dtype_layout_rule, batching_rule, transpose_rule, sharding_rule, backend_kernel, tests | - |
| `permute` | tensor_algebra | partial | yes | math_semantics, shape_rule, dtype_layout_rule, jvp, batching_rule, transpose_rule, sharding_rule, backend_kernel, tests | - |
| `collective_permute` | collective | partial | yes | math_semantics, shape_rule, dtype_layout_rule, vjp, jvp, batching_rule, transpose_rule, sharding_rule, lowering_rule, backend_kernel | all |
| `scan` | control_flow | partial | yes | math_semantics, shape_rule, dtype_layout_rule, vjp, jvp, batching_rule, transpose_rule, sharding_rule, lowering_rule, backend_kernel | all |
| `selective_ssm` | state_space | partial | yes | math_semantics, shape_rule, dtype_layout_rule, jvp, batching_rule, transpose_rule, sharding_rule, masking_effect_rule, backend_kernel, tests | Mamba/SSM |
| `dataset_map` | data | partial | yes | math_semantics, shape_rule, dtype_layout_rule, vjp, jvp, batching_rule, transpose_rule, sharding_rule, lowering_rule, backend_kernel | all |
| `tokenizer_bpe` | tokenizer | partial | yes | math_semantics, shape_rule, dtype_layout_rule, vjp, jvp, batching_rule, transpose_rule, sharding_rule, lowering_rule, backend_kernel | all |
<!-- END GENERATED PRIMITIVE COVERAGE SNAPSHOT -->

Regenerate/check the full table programmatically with:

```python
from tessera.compiler.primitive_coverage import render_markdown

print(render_markdown())
```
