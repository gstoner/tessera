---
status: Normative (development roadmap)
classification: Audit / Plan
authority: Sequences every open capability gap into executable phases
last_updated: 2026-05-10
---

# Tessera Development Roadmap

This document is the **authoritative sequencing plan** for every issue raised
in the May 2026 capability review. Tasks are grouped into phases by dependency
and impact. Each task has explicit acceptance criteria so Claude (or any
contributor) can pick it up and execute without further design conversation.

If a task lists "Open question:", that is a decision the implementer must
surface back to the user before writing code. Otherwise, the task is
self-contained.

## How to use this doc

1. Scan **Phase ordering rationale** to understand dependencies.
2. Pick the lowest-numbered task whose dependencies are all ✅.
3. Read its **Acceptance criteria**; those are the test(s) you must make pass.
4. After landing, mark the task ✅ and update any cross-referencing audits
   (e.g., `advanced_examples_capability_gap.md`).

Status legend: 📋 planned · 🚧 in progress · ✅ done · 🔲 deferred (out of
scope this cycle, with a tracked reason).

## Phase ordering rationale

```
A (quick wins)   ───┐
                    ├──► C (Theme 1 cleanup)
B (protocols)    ───┘                ┐
                                     ├──► D (streaming)
                                     ├──► E (KV-cache)
F (autodiff follow-ups, parallelizable)
G (NVIDIA execution — biggest lift, parallelizable with F)
H (Conv2d + remaining nn cleanup, after G picks the layout)
I (DDP/FSDP — depends on F4+F5 + G)

S0 (standalone gap lock) ─► S1 (primitive contracts) ─► S2 (tensor algebra)
                                                       ├► S3 (state trees)
                                                       ├► S4 (RNG/effects)
                                                       ├► S5 (control flow)
                                                       ├► S6 (sharding + collectives)
                                                       ├► S7 (model primitives)  [depends on S2-S5]
                                                       ├► S9 (numerics + quantization)
                                                       ├► S10 (optimizers)        [depends on S3]
                                                       ├► S11 (losses)            [depends on S2, S9]
                                                       ├► S12 (checkpointing)     [depends on S3]
                                                       ├► S13 (custom-op API)     [depends on S5]
                                                       ├► S14 (cache + AOT)       [depends on S5]
                                                       ├► S15 (data pipeline)     [in-scope per S0]
                                                       └► S8 (tiny model conformance) [depends on S2-S7, S9-S15]
```

**Legacy A-I critical paths (all three chains are now closed as of 2026-05-09):**
- ✅ **B1 (buffer protocol)** unblocked `BatchNorm1d` (C1) + streaming kernels
  with state (D). Closed.
- ✅ **B2 (`KVCacheHandle` value type)** unblocked Theme 4 entirely (E1–E3) +
  the `KVCache` Module wrapper (C2). Closed.
- ✅ **F4 (Graph IR adjoints) → F5 (effect-aware adjoint collectives) → I
  (DDP/FSDP).** This chain — the long pole for distributed training — is
  closed at v1. F4+F5 verified end-to-end on MLIR 21; I1/I2 ship against
  mock_collective.
- 🚧 **G is the highest-leverage remaining phase.** Without NVIDIA execution, the
  autotuner, FA-4 verification, GPU-only tier, and GPU CI are all dark. G1
  audit is done; G2–G7 sequenced in `docs/audit/nvidia_execution_audit.md`.

Estimated remaining runway for the legacy A-I execution track: **4–8 weeks of
focused work on Phase G** (4–6 days to first H100 BF16 GEMM per the G1 audit;
remainder is sweep + verification + CI).

**New standalone compiler track:** The S-series milestones below are the next
roadmap for Tessera as a self-contained model compiler, independent of PyTorch,
JAX, or Flax at runtime. PyTorch/Core ATen and JAX/Flax are used only as
reference vocabularies for missing mathematical and compiler semantics.

---

## Standalone compiler milestone sprints (S-series)

These sprints convert the May 2026 JAX/Flax/PyTorch gap analysis into an
implementation path. They run in parallel with Phase G where possible: S1-S5,
S9-S14 are mostly frontend/compiler/runtime-semantics work; S6 and S8 benefit
from real GPU execution but must still have CPU-reference acceptance; S15 is
runtime-only data plumbing.

**Sprint sequencing:**
- **S0** (lock + scope decisions) gates everything.
- **S1** (registry) gates every other S-sprint because each sprint adds
  contract-complete entries to it.
- **S2-S6, S9-S15** are largely parallelizable behind S1 with the explicit
  dependencies declared on each sprint.
- **S7** depends on S2-S5 because layers compose tensor algebra (S2),
  state (S3), RNG (S4), and transforms (S5).
- **S8** is the conformance gate — it depends on every other S-sprint.

### [S0] Gap lock + sprint documentation ✅

**Scope:** XS (doc/test only). This section is the canonical sprint breakdown
for the standalone compiler goal.

**In-scope decisions locked here:**
- **Tessera remains runtime-independent of PyTorch, JAX, and Flax.** They are
  reference vocabularies only; nothing in the runtime, compiler, or shipped
  artifact may import them.
- **The data pipeline is in scope** (S15). Tessera owns its own dataset,
  batching, sharding-of-data, and tokenization surfaces. The standalone
  compiler claim does not allow punting to `tf.data`, `torch.utils.data`, or
  `grain` at runtime — those are reference vocabularies only.
- **The training step is in scope.** Optimizers (S10), losses (S11), and
  checkpointing (S12) are not optional add-ons; without them S8 cannot
  execute its "one training step on CPU reference" criterion.
- **Custom-primitive authoring is in scope** (S13). Tile-level kernel
  authoring with VJP/JVP/batching/sharding rule registration is a
  differentiator, not an external concern.
- **AOT export and persistent compilation cache are in scope** (S14). A
  JIT-only compiler does not ship.

**Acceptance:**
- `docs/audit/execution_roadmap.md` contains S0-S15 milestone sprints
  (S0-S8 plus S9 numerics/quant, S10 optimizers, S11 losses, S12
  checkpointing, S13 custom-op API, S14 cache + AOT, S15 data pipeline).
- The S-series explicitly states that Tessera remains runtime-independent from
  PyTorch, JAX, and Flax.
- Unit coverage checks that the standalone roadmap names every required
  compiler surface: primitive contracts, tensor algebra, state trees, explicit
  RNG, control flow, sharding, model primitives, tiny model conformance,
  numerics/quantization, optimizers, losses, checkpointing, custom-op API,
  compilation cache + AOT, and data pipeline.

### [S1] Native primitive contract registry ✅

**Scope:** M (~350 LOC code + ~150 LOC tests + generated markdown).

**Status (landed 2026-05-10):** `python/tessera/compiler/primitive_coverage.py`
now provides the standalone primitive coverage registry, imports existing
`OP_SPECS` as partial entries, keeps planned gaps separate from supported ops,
and renders the markdown dashboard contract. Coverage is locked by
`tests/unit/test_standalone_compiler_roadmap.py`.

Build a standalone registry next to, not inside, the supported-op catalog. The
current `op_catalog.py` says what exists; S1 documents what each primitive must
prove before it is compiler-complete.

**Files (new):**
- `python/tessera/compiler/primitive_coverage.py`
- `tests/unit/test_standalone_compiler_roadmap.py`
- `docs/audit/standalone_primitive_coverage.md`

**Acceptance:**
- Each primitive/gap entry records: math semantics, shape rule, dtype/layout
  rule, VJP, JVP, batching rule, transpose rule, sharding rule, masking/effect
  behavior, lowering rule, backend kernel, and tests.
- Existing `OP_SPECS` are imported as implemented-or-partial entries, without
  falsely marking missing rules as complete.
- Missing standalone primitives are represented as planned entries, not as
  supported ops; as of the current snapshot, the known planned reduction and
  memory entries have been promoted to Python-reference primitives.
- The generated markdown dashboard groups entries by model-family relevance:
  diffusion, RNN/xLSTM, SSM/Mamba, Hyena/FNet/spectral, Linformer/cosFormer,
  Griffin/Megalodon, Titans/Atlas memory, and JEPA.

### [S2] Tensor algebra, indexing, and scalar math 🟢 (Python reference + hardening complete)

**Scope:** L (~900 LOC code + ~600 LOC tests). Depends on S1.

**Status (2026-05-10):** S2 is now implemented as a CPU-reference compiler
surface in `python/tessera/__init__.py`, `python/tessera/compiler/op_catalog.py`,
`python/tessera/compiler/primitive_coverage.py`, and
`python/tessera/autodiff/vjp.py`. Covered today: tensor algebra (`reshape`,
`view`, `flatten`, `squeeze`, `unsqueeze`, `permute`, `broadcast`, `expand`,
`cat`, `stack`, `split`, `chunk`, `pad`, `tile`, `repeat`, `roll`, `flip`,
`slice`, `select`, `dynamic_slice`, `dynamic_update_slice`), indexing/update (`take`,
`index_select`, `scatter`, `scatter_add`, `scatter_reduce`, `index_update`,
`nonzero`, `top_k`, `sort`, `argsort`), reductions (`mean`/`prod`/`amax`/
`amin`/`max`/`min`/`var`/`std`/`argmax`/`argmin`/`cumsum`/`cumprod`/
`cummax`/`cummin`), stability primitives
(`logsumexp`, `log_softmax`, `log1p`, `expm1`, `softplus`, `sigmoid_safe`),
scalar math (`sub`, `div`, `floor_div`, `mod`, `exp`, `log`, `sqrt`, `rsqrt`,
`pow`, `cos`, `tan`, `sinh`, `cosh`, `asin`, `acos`, `atan`, `atan2`, `erf`,
`erfc`, `lgamma`, `digamma`), numeric helpers, comparisons, logical ops, and
full bitwise ops. Differentiable entries have VJPs where the local math
contract is unambiguous; discontinuous, boolean/integer-valued, and sort-like
entries intentionally do not. The small reduction alias gaps (`max`, `min`,
`cummax`, `cummin`) are now first-class reference primitives. Remaining S2
hardening is formal shape/dtype promotion rules, JVP/batching/transpose/
sharding rules, and backend kernels.

Complete Tessera-native tensor manipulation and functional indexing. These are
compiler primitives, not PyTorch compatibility wrappers.

**Primitive groups:**
- Shape/view/data movement: `reshape`, `view`, `flatten`, `squeeze`,
  `unsqueeze`, `permute`, `broadcast`, `expand`, `cat`, `stack`, `split`,
  `chunk`, `slice`, `select`, `pad`, `tile`, `repeat`, `roll`, `flip`,
  `dynamic_slice`, `dynamic_update_slice`.
- Functional updates: indexed set/add/min/max without mutation-dependent
  behavior.
- Indexing/sorting: `gather`, `scatter`, `scatter_add`, `scatter_reduce`,
  `nonzero`, `top_k`, `sort`, `argsort`, `take`, `index_select`,
  `index_update`.
- Reductions: `sum`, `mean`, `prod`, `max`, `min`, `var`, `std`, `argmax`,
  `argmin`, `cumsum`, `cumprod`, `cummax`, `cummin`. Each carries an explicit
  `axis`/`keepdims` rule and an fp32-accumulate dtype-promotion policy.
- Numerical-stability primitives: `logsumexp`, `log_softmax`, `log1p`,
  `expm1`, `softplus`, `sigmoid_safe`, `softmax_safe`. These are first-class,
  not derived ops, because they have their own VJPs and lowering rules.
- Scalar math breadth: `sub`, `div`, `floor_div`, `mod`, `exp`, `log`,
  `sqrt`, `rsqrt`, `pow`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`,
  `asin`, `acos`, `atan`, `atan2`, `erf`, `erfc`, `lgamma`, `digamma`.
- Comparisons + logical: `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `logical_and`,
  `logical_or`, `logical_not`, `logical_xor`, `bitwise_and`, `bitwise_or`,
  `bitwise_xor`, `bitwise_not`.
- Numeric helpers: `clamp`, `minimum`, `maximum`, `sign`, `abs`,
  `reciprocal`, `floor`, `ceil`, `round`, `trunc`, `where`, `isnan`,
  `isinf`, `isfinite`.

**Acceptance:**
- Every S2 primitive has shape/dtype rules, CPU reference behavior, VJP/JVP
  status, and Graph IR lowering status in the S1 dashboard.
- Functional update tests prove no caller-visible mutation leaks through the
  API.
- Dynamic-shape tests cover symbolic batch, sequence, image height/width, and
  memory-window dimensions.

### [S3] Pytrees, module state, and model containers 🟢 (Python reference + hardening complete)

**Scope:** M (~500 LOC code + ~350 LOC tests). Can run after S1.

**Status (2026-05-10):** S3 is implemented as a Python/reference compiler
surface in `python/tessera/state/tree.py` with tests in
`tests/unit/test_state_tree.py`. Covered today: `tree_flatten`,
`tree_unflatten`, `tree_map`, `tree_reduce`, `tree_transpose`,
`tree_leaves`, `tree_structure`, `tree_all`, `tree_any`, custom pytree node
registration, typed `STATE_COLLECTION_SPECS`, `empty_state_tree`,
`module_state_tree`, `state_filter`, and `state_partition`. The module-state
projection separates trainable params, persistent buffers, and BatchNorm-style
batch stats without depending on PyTorch/JAX/Flax objects. Remaining S3
hardening is Graph IR state typing, optimizer/recurrent/memory state producers,
and sharded state-tree lowering.

Add a Tessera-native tree/state abstraction inspired by JAX pytrees and Flax
collections, but owned by Tessera semantics.

**Acceptance:**
- Native tree APIs support flatten/unflatten, tree-map, tree-reduce,
  tree-transpose, state filters, and state partitioning.
- State classes distinguish params, buffers, batch stats, optimizer slots,
  RNG state, recurrent state, memory state, and metrics.
- `nn.Module.state_dict()` can be projected into filtered state trees without
  depending on external framework objects.
- Tests cover nested modules, shared containers, mutable/non-mutable buffers,
  optimizer state, recurrent state, memory state, and metrics.

### [S4] Explicit RNG and stochastic effects 🟢 (Python reference + hardening complete)

**Scope:** M (~450 LOC code + ~300 LOC tests). Depends on S1; integrates with
S3 once state trees exist.

**Status (2026-05-10):** S4 is implemented as a typed Python/reference RNG
surface in `python/tessera/rng.py` with tests in `tests/unit/test_rng_keys.py`.
Covered today: `RNGKey.from_seed`, `split`, `fold_in`, `clone`, named streams,
serializable replay metadata (`to_state` / `from_state`), and samplers
`uniform`, `normal`, `truncated_normal`, `bernoulli`, `categorical`,
`multinomial`, `randint`, `permutation`, `gamma`, `beta`, `dirichlet`, and
`poisson`. `ts.ops.dropout(..., rng=RNGKey)` now accepts explicit typed RNG
streams. Remaining S4 hardening is Graph IR RNG-key typing, typed RNG lowering
for all random ops, stochastic-depth/random-mask callsites, and checkpoint /
distributed replay integration through S3/S6.

Make randomness a typed, compiler-visible effect rather than an implicit
process-global source.

**Acceptance:**
- Add typed RNG keys with `split`, `fold_in`, `clone`, named streams, and
  deterministic replay metadata.
- Add samplers (each a separate registry primitive with its own contract):
  `rng_uniform`, `rng_normal`, `rng_truncated_normal`, `rng_bernoulli`,
  `rng_categorical`, `rng_multinomial`, `rng_randint`, `rng_permutation`,
  `rng_gamma`, `rng_beta`, `rng_dirichlet`, `rng_poisson`.
- Dropout, stochastic depth, diffusion noise, random masking, and sampling use
  explicit RNG streams.
- Tests prove deterministic replay across single-device, sharded, checkpointed,
  and resumed runs.

### [S5] Control flow and transform composition 🟢 (Python reference + hardening complete)

**Scope:** XL (~1,200 LOC code + ~800 LOC tests). Depends on S1; uses S2 for
indexing-heavy cases.

**Status (2026-05-10):** S5 is implemented as a CPU/reference compiler
surface in `python/tessera/control.py`, layered over the existing autodiff
transforms in `python/tessera/autodiff/`. Covered today: `scan`,
`associative_scan`, `while_loop`, `fori_loop`, `cond`, `switch`, `map`,
`pmap`, `axis_index`, `axis_size`, `axis_name`, `value_and_grad`, `vjp`,
`jvp`, `vmap`, `remat`, `checkpoint`, and `autocast`. Tests cover structured
control flow, axis context, `grad(vmap(f))`, `vmap(grad(f))`, `grad(scan(f))`,
and `remat(scan(f))`. Remaining S5 hardening is Graph IR / Schedule IR /
Tile IR control-flow lowering, split-transpose scan rules, backend-native
batched lowering, and full sharding-aware transform rules.

Promote JAX-like transforms to first-class Tessera compiler semantics rather
than Python helpers.

**Acceptance:**
- First-class transforms: `jit`, `grad`, `value_and_grad`, `jvp`, `vjp`,
  `vmap`, `pmap`, `map`, `scan`, `associative_scan`, `while_loop`,
  `fori_loop`, `cond`, `switch`, `remat`, and `checkpoint`.
- Mesh-aware transform helpers: `axis_index`, `axis_size`, `axis_name` are
  first-class so functions inside `vmap`/`pmap`/`shard_map` can introspect
  their position. (Listed here, not in S6, because they are transform-time
  semantics, not collective primitives.)
- Transform rules compose through Graph IR, Schedule IR, Tile IR, and backend
  lowering, with explicit fallback diagnostics where a stage is not ready.
- Reverse-mode through scan, split transpose for scans, checkpointed BPTT, and
  stateful streaming scan are covered.
- Tests include `grad(vmap(f))`, `vmap(grad(f))`, `grad(scan(f))`,
  `remat(scan(f))`, and `shard_map(grad(f))` once S6 lands.

### [S6] Native sharding, collectives, and distributed semantics 🟢 (Python reference + hardening complete)

**Scope:** L (~800 LOC code + ~500 LOC tests). Depends on S3-S5; GPU runtime
acceleration depends on Phase G.

**Status (2026-05-10):** S6 is implemented as a CPU/reference sharding surface
in `python/tessera/sharding.py`. Covered today: `NamedMesh`, `PartitionSpec`,
`NamedSharding`, `named_sharding`, `partition_spec`, `shard_map`, and
collective primitives `psum`, `pmean`, `pmax`, `pmin`, `collective_permute`,
and `broadcast_to_axis`. Tests validate mesh metadata, partition specs,
reference `shard_map` split/reassemble behavior, axis context, and stacked
rank-value collectives. Remaining S6 hardening is Graph IR placement typing,
collective transpose/VJP/sharding rules, multi-axis `shard_map`, optimizer /
RNG / checkpoint integration, and hardware runtime lowering.

Add compiler-visible placement, sharding, and SPMD mapping semantics, plus the
collectives primitive library that `shard_map` callees actually invoke.

**Acceptance:**
- Named mesh, partition specs, named sharding, replicated/sharded state,
  addressable-device metadata, and memory-kind placement exist as native
  objects.
- Add a `shard_map`-style API with explicit in/out specs and collective
  behavior.
- **Collectives primitive library** — each is a registry primitive with its
  own VJP/transpose/sharding rule and Graph IR lowering: `psum`, `pmean`,
  `pmax`, `pmin`, `all_gather`, `all_to_all`, `reduce_scatter`,
  `collective_permute`, `broadcast_to_axis`. (`tessera.all_reduce`/`all_gather`/`all_to_all`/
  `reduce_scatter` exist in `op_catalog.py` today; S6 turns them into
  contract-complete entries with batching/transpose/sharding rules.)
- Data, tensor, sequence, expert, pipeline, and memory sharding are expressible
  in the compiler and visible to autodiff/optimizer/RNG/checkpoint logic.
- CPU/mock tests validate semantics; NVIDIA/NCCL tests become active when
  Phase G provides hardware execution.

### [S7] Flax-level model primitive library 🟢 (Python reference + hardening complete)

**Scope:** L (~900 LOC code + ~600 LOC tests). Depends on S2-S5 (recurrent
layers require `scan` from S5; attention layers require batching rules from
S5; normalization layers require reductions from S2; sampling layers require
S4 RNG).

Provide native model-building layers so Tessera users can author models without
PyTorch/JAX/Flax modules.

**Acceptance:**
- Layers: `Linear`, `LinearGeneral`, `Einsum`, `Embed`, `Conv1d`, `Conv2d`,
  `Conv3d`, `ConvTranspose`, `LoRA`, and `LoRALinear`.
- Normalization: BatchNorm, LayerNorm, RMSNorm, GroupNorm, InstanceNorm,
  WeightNorm, SpectralNorm.
- Pooling: `max_pool`, `avg_pool`, `min_pool`, adaptive variants.
- Recurrent layers: `simple_rnn_cell`, `gru_cell`, `lstm_cell`, optimized
  LSTM, bidirectional scan, sequence flip/masking.
- Position encodings: `rope`, `alibi`, `ntk_rope` as registry primitives
  with shape/dtype/VJP rules.
- Attention library as registry primitives (in addition to existing `flash_attn`):
  `multi_head_attention`, `gqa_attention`, `mqa_attention`, `mla_decode`,
  `mla_decode_fused`, `attn_sliding_window`, `attn_compressed_blocks`,
  `attn_top_k_blocks`, `deepseek_sparse_attention`, `gated_attention`,
  `hybrid_attention`, `lightning_attention`, `gated_deltanet`,
  `kimi_delta_attention`, and `modified_delta_attention`.
- Dropout and stochastic-depth modules consume explicit Tessera RNG streams.

**Status 2026-05-10:** partial reference surface landed. `tessera.nn`
now exposes `LinearGeneral`, `Einsum`, `LoRALinear`, `Conv1d`,
`ConvTranspose1d`/`ConvTranspose`, `GroupNorm`, `InstanceNorm`,
`WeightNorm`, `SpectralNorm`, pooling helpers, `SimpleRNNCell`, `GRUCell`,
`bidirectional_scan`, `rope`, `rope_split`, `rope_merge`, `alibi`,
`ntk_rope`, `multi_head_attention`, `gqa_attention`, `mqa_attention`,
`mla_decode`, `mla_decode_fused`, `gated_attention`, `hybrid_attention`,
`deepseek_sparse_attention`, `lightning_attention`, `gated_deltanet`,
`kimi_delta_attention`, and `modified_delta_attention`, with coverage entries
and unit tests. A reference Titans/Atlas memory surface now ships as
`memory_read`, `memory_write`, and `memory_evict`, with explicit
math/shape/dtype contracts and lowering/backend gates in the dashboard.
`conv1d` now has fp64 reference semantics, VJP/JVP coverage, and an
`explicit_semantic` contract; `linear_general` also has an
`explicit_semantic` contract, focused VJP/JVP rules, and a Graph IR catalog
entry. RoPE/MLA/NSA, sparse attention, MoE dispatch/combine, Lightning,
Gated DeltaNet, Kimi Delta, Modified Delta, gated attention, and hybrid
attention have VJP/JVP registry coverage. The normal x86/GPU lowering
pipelines include slots for SwiGLU, MLA fusion, Native Sparse Attention,
Lightning attention fusion, Delta chunking, and hybrid-attention expansion.
Remaining S7 work: Conv3d, sequence flip/masking, stochastic depth, memory
state ABI integration, batching/sharding/transpose rules for reference layers,
and backend fused kernels for the reasoning-model attention families.

### [S9] Numerics, mixed precision, and quantization 🟢 (Python reference + hardening complete)

**Scope:** L (~700 LOC code + ~450 LOC tests). Depends on S1; integrates with
S2 (dtype rules) and S5 (autocast as a transform).

Standalone-compiler numerics policy. Today's `op_catalog.py` ships `cast`,
`quantize_fp{8,6,4}`, and `dequantize_fp{8,6,4,nvfp4}`; S9 turns those into a
contract-complete numerics layer plus the missing mixed-precision and QAT
infrastructure.

**Acceptance:**
- Compiler-visible dtype lattice: `f64`, `f32`, `bf16`, `f16`, `f8e4m3`,
  `f8e5m2`, `f6`, `f4`, `nvfp4`, `i64`, `i32`, `i16`, `i8`, `u8`, `bool`.
  Promotion rules and reduction-precision policy (fp32 accumulate by
  default) documented in the registry.
- `autocast(dtype)` transform composes through `grad`/`vmap`/`scan` and
  rewrites supported ops to the chosen dtype while keeping reductions in
  fp32. Replaces today's ad-hoc `python/tessera/autodiff/mixed_precision.py`
  hooks with a first-class transform.
- Quantization primitives: per-tensor symmetric, per-channel, blockwise,
  GPTQ-style int4, and AWQ-style int4. `quantize_int8`, `dequantize_int8`,
  `quantize_int4`, `dequantize_int4` join the existing fp8/fp6/fp4/nvfp4
  ops with full contracts (VJP via straight-through estimator + transpose
  rule).
- QAT (quantization-aware training) hooks: fake-quantize ops with VJPs,
  observer modules, calibration capture.
- `GradScaler` migrated from `python/tessera/autodiff/mixed_precision.py`
  to expose a contract-complete primitive.
- Tests cover dtype-promotion correctness, autocast under reverse-mode AD,
  per-channel quant round-trip error, and QAT gradient flow.

**Status 2026-05-10:** partial reference API landed in
`python/tessera/quantization.py` and is exported from `tessera`. Covered:
`quantize_int8`, `dequantize_int8`, `quantize_int4`, `dequantize_int4`,
`fake_quantize`, `CalibrationObserver`/`calibration_observer`, and
`grad_scaler_step`, with unit tests and primitive-coverage entries. Existing
fp8/fp6/fp4/nvfp4 ops remain available through `tessera.ops`; `quantize_fp8`,
`dequantize_fp8`, `quantize_fp4`, `dequantize_fp4`, and `fake_quantize` now
have STE-style VJP/JVP coverage for QAT and fp8/fp4-adjacent LLM training.
Remaining S9 work: per-channel/blockwise quantization, GPTQ/AWQ policy hooks,
full dtype-lattice documentation, reduction precision policy, and full
autocast rewrite integration through compiled IR.

### [S10] Optimizer library and training-step primitives 🟢 (Python reference + hardening complete)

**Scope:** M (~600 LOC code + ~400 LOC tests). Depends on S3 (state trees
hold optimizer slots) and S2 (scalar math + reductions).

Today `op_catalog.py` exposes a single `tessera.adam` primitive and
`python/tessera/nn/utils.py` ships `clip_grad_norm_`. S10 promotes the
optimizer surface to a real library.

**Acceptance:**
- Optimizers as functional primitives over state trees: `sgd`, `momentum`,
  `nesterov`, `adam`, `adamw`, `adafactor`, `lion`, `muon`, `lamb`. Each
  registry entry has a state-tree shape contract and a step contract.
- Learning-rate schedules: `constant_lr`, `cosine_lr`, `cosine_warmup`,
  `linear_warmup`, `polynomial_lr`, `inverse_sqrt`, `cyclical`,
  `chained_schedule`.
- Gradient transforms: `clip_grad_norm`, `clip_grad_value`,
  `centralize_grad`, `add_decoupled_weight_decay`, `ema_update`, `polyak`.
- `OptaxStyleChain` — composition of gradient transforms (the reference
  vocabulary is `optax`/`flax.optim`; Tessera owns the implementation).
- Tests cover Rosenbrock convergence, AdamW vs Adam decoupling,
  Adafactor's factored second moments, EMA shadow consistency, and
  optimizer-state pytree round-trip through S3.

**Status 2026-05-10:** partial functional optimizer library landed in
`python/tessera/optim.py` and is exported as `tessera.optim`. Covered:
`sgd`, `momentum`, `nesterov`, `adam`, `adamw`, `adafactor`, `lion`, `muon`,
`lamb`, `constant_lr`, `cosine_lr`, `cosine_warmup_lr`,
`linear_warmup_lr`, `polynomial_lr`, `inverse_sqrt_lr`,
`clip_grad_norm`, `clip_grad_value`, `centralize_grad`,
`add_decoupled_weight_decay`, `ema_update`, `polyak_avg`, and
`OptaxStyleChain`/`chain`. Unit tests cover nested parameter trees,
factored Adafactor state, optimizer-state S3 tree round trip, schedules,
EMA/Polyak updates, gradient-transform composition, and dtype policy behavior.
The minimum LLM optimizer set (`adam`, `adamw`, `momentum`, `adafactor`,
`lion`) is compiler-visible in `op_catalog`, ODS, stubs, primitive coverage,
and VJP/JVP registries. Optimizer math and state default to fp32; fp16/bf16
params cast updates back to storage dtype by default; optional `master_dtype`
supports fp32 master weights for fp16 and quantized-adjacent QAT paths.
Remaining S10 work: sharded optimizer state integration, Rosenbrock/
convergence suites, fused backend optimizer kernels, and larger training-loop
examples.

**Refinement 2026-05-10:** added `adam`, `cyclical_lr`, `chained_schedule`,
mixed-precision optimizer kwargs, and autodiff coverage for `adam`,
`adafactor`, and `lion` to close the remaining minimum optimizer gaps for
MoSA, MoE, DeepSeek-style fp8-adjacent training, MiniMax, and Kimi-style
long-context finetuning.

### [S11] Loss / criterion library 🟢 (Python reference + hardening complete)

**Scope:** M (~400 LOC code + ~350 LOC tests). Depends on S2 (reductions +
stability primitives) and S9 (numerics policy).

Today only `CrossEntropyLoss` exists in `python/tessera/nn/layers.py`. S11
ships the full standalone-compiler loss surface needed for S8's training-step
acceptance.

**Acceptance:**
- Regression: `mse_loss`, `mae_loss`, `huber_loss`, `smooth_l1_loss`,
  `log_cosh_loss`.
- Classification: `cross_entropy_loss` (existing — promoted to registry
  primitive), `binary_cross_entropy_loss`, `focal_loss`,
  `label_smoothed_cross_entropy`.
- Distribution: `kl_divergence`, `js_divergence`, `wasserstein_distance`.
- Contrastive / metric: `nt_xent_loss`, `info_nce_loss`, `triplet_loss`,
  `contrastive_loss`, `cosine_embedding_loss`.
- Diffusion: `ddpm_noise_pred_loss`, `vlb_loss`, `score_matching_loss`.
- Sequence: `ctc_loss`, `seq2seq_loss`.
- Reasoning RL: `normalize_group_advantages`, `ppo_policy_loss`,
  `grpo_policy_loss`, `cispo_policy_loss`, and a `RolloutBatch` container for
  log-probs, old/ref log-probs, rewards, masks, and metadata.
- Each loss registers its `reduction` policy (`none`/`mean`/`sum`),
  numerical-stability path (uses `logsumexp` / `log_softmax` from S2 where
  applicable), and a VJP.

**Status 2026-05-10:** partial functional loss library landed in
`python/tessera/losses.py` and is exported as `tessera.losses`. Covered:
regression (`mse_loss`, `mae_loss`, `huber_loss`, `smooth_l1_loss`,
`log_cosh_loss`), classification (`cross_entropy_loss`,
`binary_cross_entropy_loss`, `focal_loss`,
`label_smoothed_cross_entropy`), distribution (`kl_divergence`,
`js_divergence`, `wasserstein_distance`), contrastive/metric
(`nt_xent_loss`, `info_nce_loss`, `triplet_loss`, `contrastive_loss`,
`cosine_embedding_loss`), diffusion (`ddpm_noise_pred_loss`, `vlb_loss`,
`score_matching_loss`), and sequence (`ctc_loss`, `seq2seq_loss`).
Focused VJP/JVP and Graph IR catalog coverage now exists for MSE/MAE/Huber/
SmoothL1, log-cosh, BCE, DDPM noise-prediction, score matching, and VLB
reducers, and those losses now carry `explicit_semantic` math/shape/dtype
contracts. `python/tessera/rl.py` now exports PPO, GRPO, and CISPO helpers;
CISPO clips importance weights directly and treats the clipped weight as a
constant multiplier on the log-prob objective. The RL helpers are registered in
`op_catalog`, `tessera.ops`, primitive coverage, and VJP/JVP registries.
Remaining S11 work: VJP/JVP registration for the remaining classification,
distribution, contrastive, and sequence losses, reduction-policy registry
metadata, backend kernels, and larger numerical stability goldens. S15
data/tokenizer primitives are now explicitly declared non-differentiable in the
coverage contract instead of appearing as missing autodiff work.

### [S12] State serialization and checkpointing 🟢 (Python reference + hardening complete)

**Scope:** M (~500 LOC code + ~350 LOC tests). Depends on S3 (state trees) and
S6 (sharded state for sharded checkpoints).

Standalone-compiler weights/state must be saveable, restorable, version-able,
and shardable without external framework dependencies.

**Acceptance:**
- On-disk format: typed binary container holding state-tree topology,
  per-leaf metadata (dtype, shape, sharding spec, version), and tensor
  payloads. Self-describing — no external schema needed to read it back.
- API: `save_state(tree, path)`, `load_state(path) -> tree`,
  `save_sharded(tree, path, mesh)`, `load_sharded(path, mesh)`.
- Migration: registered `state_migration` rules so a checkpoint produced
  at version N can be loaded at version N+1 with explicit field renames /
  shape splits / dtype upgrades.
- Partial loading: load only `params` (drop optimizer slots), or merge a
  LoRA adapter into a base checkpoint.
- Resilience: torn-write detection (per-leaf checksums), atomic rename,
  and resumable saves for multi-host training.
- Tests cover round-trip equality on every state class from S3, sharded
  save/load on the mock collective harness, and a forward-compat migration
  example.

**Status 2026-05-10:** real state-tree payload serialization landed in
`python/tessera/checkpoint.py` alongside the existing runtime checkpoint
manifest API. Covered: `save_state`, `load_state`, checksummed leaves,
atomic rename, top-level collection filtering for partial loads,
`register_state_migration`/`state_migration`, and
`save_sharded`/`load_sharded` with mock mesh metadata. Remaining S12 work:
multi-rank shard partitioning, resumable multi-host writes, richer partial
merge policies such as LoRA merge-in, and non-numpy tensor payload adapters.

### [S13] Custom-primitive / extension API 🟢 (Python reference + hardening complete)

**Scope:** L (~700 LOC code + ~500 LOC tests). Depends on S5 (transforms must
see custom primitives) and S1 (registry).

Tessera's tile-level kernel authoring is the standalone differentiator. S13
formalizes how a user-defined kernel registers with the compiler so that
`grad`, `vmap`, `pmap`, `shard_map`, and `autocast` all compose with it.

**Acceptance:**
- `@tessera.custom_primitive(name)` decorator that lets a user define:
  forward implementation (Python or `@tessera.kernel` tile program), shape
  rule, dtype rule, VJP, JVP, batching rule, transpose rule, sharding
  rule, masking/effect declaration, and a per-target lowering hook.
- `@tessera.custom_call` escape hatch for opaque kernels that bypass
  Graph IR optimization but still report shape/sharding/effects to the
  compiler.
- Per-target lowering registration: a custom primitive can supply
  `lower_to_tile`, `lower_to_x86`, `lower_to_apple_cpu`, `lower_to_apple_gpu`,
  `lower_to_rocm`, `lower_to_metalium`, etc.
- Tests cover: a user `softplus_inplace` op with a hand-written VJP that
  composes under `grad(vmap(f))`; a user kernel that registers a
  Tile-IR-only lowering and falls back to a numpy reference on other
  targets; an effect-declaring custom primitive that participates in
  collective-insertion correctly under `shard_map`.

**Status 2026-05-10:** partial custom primitive API landed in
`python/tessera/custom.py` and is exported from `tessera`. Covered:
`custom_primitive`, `custom_call`, `custom_vjp`, `custom_jvp`,
`custom_batching`, metadata-bearing shape/dtype/transpose/sharding/masking
rules, and per-target lowering registration. Custom primitives register as
real `tessera.ops` entries so the existing autodiff tape can see them; tests
cover a hand-written VJP under `grad(vmap(f))`, Tile-only lowering metadata,
and an opaque state-effecting custom call. Remaining S13 work: Graph IR
custom-op nodes, collective insertion/effect lowering beyond metadata,
target-specific custom-call ABI plumbing, and transform-rule enforcement in
compiled pipelines.

### [S14] Compilation cache and AOT export 🟢 (Python reference + hardening complete)

**Scope:** L (~700 LOC code + ~400 LOC tests). Depends on S5 (the transforms
that compile) and S6 (sharding metadata travels with the artifact).

A JIT-only compiler is not deployable. S14 ships persistent compilation
caching and a real AOT export path.

**Acceptance:**
- Persistent JIT cache keyed on
  `hash(graph_ir + target + dtype_policy + mesh_spec + tessera_version)`
  with on-disk artifact storage (extends today's autotuner SQLite cache to
  cover compiled programs, not just kernel configs).
- Cache invalidation rules surface to the user when any keyed input
  changes — never silent staleness.
- AOT export targets:
  - `tessera.aot.export(fn, *example_inputs) -> AOTArtifact` produces a
    self-contained artifact with Graph IR, per-target Tile IR, and the
    minimum runtime metadata to launch.
  - StableHLO export for the JAX/TF interoperability story.
  - GGUF / safetensors export for inference-only deployment.
- Loadable artifacts: `tessera.aot.load(path).run(inputs)` works without
  the original Python source.
- Tests cover cache hit/miss correctness, AOT round-trip equality with JIT
  on every reference op family, and a sharded-AOT artifact loaded onto a
  smaller mesh.

**Status 2026-05-10:** partial AOT/cache surface landed in
`python/tessera/aot.py` and is exported as `tessera.aot`. Covered:
`export`, `load`, `stablehlo_export`, `gguf_export`,
`safetensors_export`, `compilation_cache_key`, and `CompilationCache`.
Reference artifacts carry `RuntimeArtifact` data plus launch metadata; when a
Python callable is picklable, `aot.load(path).run(inputs)` works in the
reference path. Remaining S14 work: native artifact bundles per backend,
source-independent execution for non-picklable callables, cache invalidation
diagnostics in the JIT path, sharded-AOT mesh remapping, and real GGUF /
safetensors binary compatibility.

### [S15] Native data pipeline 🟢 (Python reference + hardening complete)

**Scope:** L (~800 LOC code + ~500 LOC tests). Depends on S3 (datasets are
state trees) and S4 (shuffle uses RNG).

**S0 decision:** the data pipeline is in scope. Tessera owns its own dataset,
batching, sharding-of-data, and tokenization surfaces. `tf.data`,
`torch.utils.data`, and `grain` are reference vocabularies only.

**Acceptance:**
- `Dataset` interface: `map`, `filter`, `batch`, `prefetch`, `shuffle`,
  `interleave`, `take`, `skip`, `repeat`, `concatenate`, `zip`, `unbatch`.
- Source datasets: in-memory tensor source, sharded file source
  (memory-mapped), generator source, deterministic synthetic source.
- Sharding-of-data: a `ShardedDataset` partitions across mesh axes so
  data parallelism is a first-class semantic, not a per-call argument.
- Determinism: shuffle uses S4 RNG keys; `dataset.replay_from(epoch, step)`
  is bit-exact under the same key.
- Tokenization surface: `Tokenizer` interface with `byte`, `bpe`,
  `wordpiece`, `unigram`, and `sentencepiece_compat` implementations
  (the format compatibility layer reads SentencePiece protobufs but the
  tokenizer is implemented in Tessera). Tokenizers expose
  `encode`/`decode`/`vocab_size`/`special_tokens` as compiler-visible state.
- Streaming / iteration: `IterableDataset` for sources too large to
  enumerate; backpressure and prefetch buffer sizing are explicit.
- Resumability: `dataset.checkpoint() / dataset.restore(state)` round-trips
  through S12 so a training run pauses and resumes without re-shuffling
  the same examples.
- Tests cover deterministic replay, sharded data-parallel iteration on the
  mock collective harness, tokenizer round-trip equality, and resumed
  iteration after a checkpoint.

**Status 2026-05-10:** partial CPU-reference data pipeline landed in
`python/tessera/data.py` and is exported as `tessera.data`. Covered:
`Dataset` with `map`, `filter`, `batch`, `prefetch`, `shuffle`,
`interleave`, `take`, `skip`, `repeat`, `concatenate`, `zip`, `unbatch`,
`checkpoint`, `restore`, and `replay_from`; tensor, synthetic, sharded-file,
sharded, and iterable sources; RNGKey-backed deterministic shuffle; and
byte/BPE/WordPiece/unigram/SentencePiece-compatible tokenizer surfaces.
Remaining S15 work: lazy streaming execution, true memory-mapped shard
readers, async prefetch workers, tokenizer training/model-file parsers, and
compiler-visible dataset lowering.

### [S8] Tiny standalone model conformance suite 🟢 (aggregate suite shipped)

**Scope:** XL (~1,000 LOC tests/examples + dashboard integration). Depends on
S2-S7 + S9-S15; performance gates depend on Phase G.

The acceptance target is small standalone Tessera models, not imported
PyTorch/JAX/Flax modules.

**Acceptance:**
- Add tiny standalone models for diffusion/DiT or U-Net, xLSTM, Mamba/SSM,
  Hyena/FNet/spectral, Linformer/cosFormer, Griffin/Megalodon, JEPA, and
  Titans/Atlas memory.
- Every tiny model validates forward correctness, backward correctness, state
  behavior, RNG determinism, shape polymorphism, mixed precision, one training
  step on CPU reference (using S10 optimizers, S11 losses, S15 data pipeline),
  and a checkpoint round-trip (S12).
- Backend gates track CPU reference, Apple/GPU development backends, and
  NVIDIA/H100 optimized execution after Phase G.
- The S1 dashboard reports which model families are blocked by each missing
  primitive contract.

**Status 2026-05-10:** initial conformance smoke tests landed in
`tests/unit/test_s7_s8_s9.py` for recurrent/scan forward+backward, RNG
state replay, diffusion-like Conv1d+GroupNorm, and attention-like
LinearGeneral+MHA fragments. Coverage now tracks
`tiny_diffusion_conformance`, `tiny_recurrent_conformance`, and
`tiny_attention_conformance` as shipped partial targets. Remaining S8 work:
dedicated tiny xLSTM, Mamba/SSM, Hyena/FNet, Linformer/cosFormer,
Griffin/Megalodon, JEPA, and Titans/Atlas examples with optimizer/loss/data
pipeline/checkpoint gates once S10-S15 mature.

**Expansion 2026-05-10:** added
`tiny_training_step_conformance` in `tests/unit/test_s8_training_conformance.py`.
It exercises S15 data batches, S11 loss, S10 optimizer/gradient transforms,
S4 RNG state, and S12 checkpoint round-trip in one CPU-reference training
step. Remaining S8 work now centers on broader model-family coverage and
backend gates rather than missing training-loop primitives.

**Tiny-model suite 2026-05-10:** shipped importable examples under
`examples/conformance/s8_tiny_models/` with a `TinyModelSpec` manifest for
diffusion/DiT, xLSTM/recurrent, Mamba/SSM, Hyena/FNet/spectral,
Linformer/cosFormer, Griffin/Megalodon, JEPA, and Titans/Atlas memory. The
aggregate suite covers S2-S15, validates forward/backward behavior, S4 RNG
replay, S3 state-tree round trip, S6 sharding/collective semantics, S9
quantization, S10 optimizer steps, S11 losses, S12 checkpoints, S13 custom
primitive metadata, S14 AOT/cache export, and S15 data/tokenizer inputs.
Compiler slices are wired into `tessera.testing.compiler_examples` so
foundation targets emit Graph/Schedule/Tile/Target IR today; CUDA/ROCm remain
artifact-only until Phase G runtime execution lands.

---

## Phase A — Quick wins (independent, parallelizable, ~1–2 weeks)

### [A1] Debugging story — env-var IR dumps + per-pass diff + how-to doc ✅

**Scope:** S (~250 LOC code + ~300 LOC docs).

**Files (new):**
- `python/tessera/debug_env.py` — env-var parser (`TESSERA_DEBUG_IR=graph,schedule,tile,target` and `TESSERA_DEBUG_DUMP_DIR=/tmp/tessera`)
- `tests/unit/test_debug_env.py`

**Files (modify):**
- `python/tessera/compiler/jit.py` — emit IR snapshots after each lowering stage when env var is set
- `docs/guides/Tessera_Debugging_Tools_Guide.md` — add "Dumping IR mid-pipeline" section
- `CLAUDE.md` — list the env vars in the Testing/Debug section

**Acceptance:**
- `TESSERA_DEBUG_IR=graph,schedule TESSERA_DEBUG_DUMP_DIR=/tmp/d ./run.py` writes `graph.mlir` and `schedule.mlir` for every JIT artifact emitted in that run.
- Per-pass diff helper: `tessera-mlir diff /tmp/d/graph.mlir /tmp/d/schedule.mlir` prints a textual line-diff (use Python `difflib`).
- Test: env var off → no files written; env var on → files exist + are non-empty MLIR.
- Doc: a recipe titled "kernel ran, results wrong, what now?" walks through `TESSERA_DEBUG_IR`, the diff tool, and `tessera.debug.replay_manifest`.

### [A2] Dynamic shapes — audit + doc + test ✅

**Scope:** S (mostly investigation + ~150 LOC test + doc).

**Files (modify):**
- `docs/spec/SHAPE_SYSTEM.md` — add "Dynamic shape support matrix per backend" section
- `tests/unit/test_dynamic_shapes.py` (new) — every supported backend × symbolic-dim case

**Acceptance:**
- Documented matrix of which `Dim("S")`/symbolic dims actually flow through to which backend (x86 / Apple_cpu / Apple_gpu).
- For each "supported" cell, a test that builds a `@tessera.jit` with a symbolic dim, calls it with two different concrete shapes, and validates correct output.
- For each "unsupported" cell, a test that asserts a clear error message at decoration or first-call time (no silent fallback).
- Update `CANONICAL_API.md` with a one-paragraph dynamic-shape semantics block.

**Audit result (2026-05-09):** dynamic shapes work on CPU reference, Apple
CPU, and Apple GPU — symbolic dims flow through to actual execution.
Call-time constraint enforcement also landed: `JitFn.__call__` resolves
symbolic dims from concrete argument shapes and raises `TesseraConstraintError`
for `Divisible`, `Range`, and `Equal` violations.

### [A2-followup] Call-time constraint enforcement ✅

**Scope:** S (~150 LOC). Lift the constraint check from `@jit` decoration
into `JitFn.__call__` so that constraint violations on real argument shapes
raise `TesseraConstraintError` even when no `bindings=` was supplied.
Acceptance: the `xfail` test in `test_dynamic_shapes.py` flips to `xpass` and
the `xfail` mark is removed.

### [A3] KV-cache lowering coverage matrix ✅

**Scope:** XS (doc-only, ~80 LOC).

**Files (new):**
- `docs/audit/kv_cache_coverage_matrix.md`

**Files (modify):**
- `CLAUDE.md` Architecture Decision #21 — link to the matrix

**Acceptance:**
- Per-target table: rows = `kv_cache_append`, `kv_cache_prune`, FA-4 with cache; columns = x86, Apple_cpu, Apple_gpu, NVIDIA, ROCm, TPU, Cerebras, Metalium, RubinCPX. Cells: ✅ executes / 🟡 lowers but no execution / 🔲 emits diagnostic / ❌ silent no-op (bug).
- Audit method: grep each backend's lowering passes for `kv_cache_*`, verify behavior, document.
- For any 🔲 cells found that turn out to be ❌, file a follow-up task in this roadmap.

### [A4] Theme 1 cleanup — small phantoms that don't need new infrastructure ✅

**Scope:** S (~300 LOC code + ~200 LOC tests).

**Files (modify):**
- `python/tessera/nn/__init__.py` — replace 8 phantoms with real classes/aliases
- `python/tessera/nn/layers.py` — add the new Module wrappers
- `tests/unit/test_nn_module.py` — add coverage

**Per-phantom resolution:**
| Phantom | Resolution |
|---------|-----------|
| `SiLU`, `Sigmoid`, `GELU`, `ReLU`, `Tanh`, `Identity` | Stateless `Module` wrappers — `def forward(self, x): return ts.ops.silu(x)` etc. (`Identity` returns input unchanged) |
| `MultiHeadCrossAttention` | Subclass of `MultiHeadAttention` whose `forward(q, k, v)` requires explicit K/V (no self-attention shortcut) |
| `RotaryEmbedding` | Module owning `theta` (precomputed); `forward(x)` calls `ops.rope(x, self.theta)` |
| `CastedLinear`, `CastedEmbedding` | Subclass `Linear` / `Embedding` with extra `cast_dtype` arg; `forward` does `ops.cast(super().forward(x), self.cast_dtype)` |
| `CrossEntropyLoss` | Functional + Module form: `-mean(log_softmax(logits)[target])`. Composes through `ops.softmax` + `ops.reduce` for autodiff |
| `nn.utils.clip_grad_norm_(params, max_norm)` | Real impl: compute total `||grad||₂`, scale `.grad` in place if above threshold |

**Acceptance:**
- Each phantom in the list above replaced with a real implementation that passes `forward` shape tests + composition tests.
- `CrossEntropyLoss` tested end-to-end through a tape — gradients to logits match numerical Jacobian.
- `clip_grad_norm_` tested: above-threshold case scales correctly; below-threshold leaves grads untouched.
- Update `advanced_examples_capability_gap.md` to mark these ✅.

### [A5] `flash_attn` VJP via `custom_rule` ✅

**Scope:** XS (~50 LOC code + ~60 LOC test).

The numpy-reference VJP is shipped via the built-in `custom_rule` registry so
`MultiHeadAttention`-style code can train end-to-end on the reference path.

**Files (new):**
- `python/tessera/autodiff/_flash_attn_vjp.py` — registered via `custom_rule("flash_attn")`
- Tests: numerical-Jacobian against `flash_attn` reference impl

**Acceptance:**
- VJP computes `dQ`, `dK`, `dV` by recomputing scores + softmax during backward (memory-efficient is out of scope; correctness first).
- Numerical-Jacobian test passes at fp64 with `rtol=1e-5`.
- `MultiHeadAttention` module training step tested end-to-end.

---

## Phase B — Foundational protocols (sequential, ~1 week) — **all three landed 2026-05-09**

### [B1] Module buffer protocol — `register_buffer` + state_dict integration ✅

**Scope:** M (~250 LOC code + ~150 LOC tests).

Buffers are non-trainable named tensors that ride alongside parameters in
`state_dict()` (BatchNorm running stats, RoPE precomputed `theta`, attention
masks, etc.). They differ from parameters in two ways: no `.grad`,
`requires_grad` is meaningless. They're persisted by `state_dict` like
parameters.

**Files (modify):**
- `python/tessera/nn/module.py` — add `_buffers: OrderedDict`, `register_buffer(name, value, persistent=True)`, attribute routing for `Buffer`-tagged tensors, `buffers()` / `named_buffers()` iterators, `state_dict()` includes persistent buffers
- `python/tessera/nn/__init__.py` — export `Buffer` (a thin tagged-ndarray class so `__setattr__` can route)
- `tests/unit/test_nn_module.py` — buffer round-trip, `persistent=False` excluded from state_dict, `to(dtype)` migrates buffers

**Acceptance:**
- `module.register_buffer("running_mean", np.zeros(64))` → `module.running_mean` returns the buffer; `module.named_buffers()` yields it; `module.state_dict()` includes it under the right name.
- `module.register_buffer("temp", arr, persistent=False)` excluded from `state_dict()` but still accessible.
- `module.parameters()` / `named_parameters()` does **not** yield buffers (regression test).
- `Module.zero_grad()` does not touch buffers (no `.grad` slot to reset).

**Decision (locked 2026-05-09):** `Buffer` is a wrapper class analogous to
`Parameter`, with `_data: DistributedArray`, no `.grad` slot, no `requires_grad`,
and `persistent: bool` for state-dict participation.

### [B2] `KVCacheHandle` opaque value type ✅

**Scope:** M (~200 LOC code + ~150 LOC tests).

A handle that flows through Tile IR / Graph IR as a first-class value
representing the state of a paged KV cache. Today, `ops.kv_cache_append`
returns the cache, but there's no formal handle type — passing it through
ops works only by Python convention. Theme 4 needs a real handle.

**Files (new):**
- `python/tessera/cache/__init__.py` — `KVCacheHandle` class (Python-side; opaque to ops)
- `python/tessera/cache/handle.py` — internal storage (paged numpy buffers), `pages`, `current_seq`, `max_seq` attributes

**Files (modify):**
- `python/tessera/__init__.py` — export `cache` namespace
- `python/tessera/ops` callsites — `kv_cache_append`/`kv_cache_prune` accept and return `KVCacheHandle` instances (with backward-compat for the existing `ReferenceKVCache`)

**Acceptance:**
- `cache = ts.cache.KVCacheHandle(num_heads=4, head_dim=64, max_seq=128, dtype="fp32")` constructs.
- `cache = ts.ops.kv_cache_append(cache, k, v)` returns a new handle (functional style).
- `ts.ops.kv_cache_read(cache, slice(0, 64))` returns `(k, v)` tensors. (Adds a new op + VJP-stub; mark `flash_attn` consumers TODO.)
- Round-trip: append then read returns the appended values.
- Test: appending past `max_seq` raises a clear `TesseraAutodiffError`-style error.

### [B3] `Module.to(dtype)` — dtype migration ✅

**Scope:** S (~100 LOC + ~80 LOC tests).

Migrate every `Parameter` and persistent `Buffer` in a module tree to a new
dtype. `to(device)` deferred until a real device handle exists post-Phase G.

**Files (modify):**
- `python/tessera/nn/module.py` — `Module.to(dtype: str) -> Module` (mutates in place + returns self for chaining)
- `tests/unit/test_nn_module.py` — coverage

**Acceptance:**
- `mlp.to("fp16")` migrates every Parameter and persistent Buffer; non-persistent buffers untouched.
- Subsequent `forward(x)` on `fp16` parameters works (uses `_as_array` extraction; numpy handles the cast).
- Round-trip: `to("fp16").to("fp32")` returns to the original dtype shape; values within fp16 quantization noise of the original.
- `to("invalid")` raises `ValueError` with the list of valid dtypes.

---

## Phase C — Theme 1 cleanup that depends on Phase B (parallelizable, ~1 week) — **landed 2026-05-09**

### [C1] `BatchNorm1d` (real Module) ✅

**Scope:** S (~80 LOC + ~80 LOC tests). Depends on **B1**.

**Files (modify):**
- `python/tessera/nn/layers.py` — `BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)`
- `tests/unit/test_nn_module.py`

**Acceptance:**
- `register_buffer("running_mean", zeros)`, `register_buffer("running_var", ones)`, `register_buffer("num_batches_tracked", 0)`.
- Train mode: uses batch stats; updates running stats with `momentum`.
- Eval mode: uses running stats (no update).
- `state_dict()` includes the buffers; `load_state_dict()` restores them.
- Replace the phantom in `nn/__init__.py`.

### [C2] `KVCache` Module wrapper ✅

**Scope:** XS (~60 LOC + ~50 LOC tests). Depends on **B2**.

Module form of the KV cache for layered transformer use:

```python
class KVCache(Module):
    def __init__(self, num_heads, head_dim, max_seq, dtype="fp32"): ...
    def forward(self, k, v):  # appends and returns full (K, V) so far
```

**Acceptance:** transformer block test uses `ts.nn.KVCache` to maintain decoding state across calls; second forward returns concatenated K/V.

---

## Phase D — Theme 3 streaming kernels (~2–3 weeks) — **D1/D2/D4 landed 2026-05-09; D3 deferred**

### [D1] `ops.depthwise_conv1d` ✅

**Scope:** M (~250 LOC + ~150 LOC tests). Depends on **B1** for streaming state.

**Acceptance:**
- `ops.depthwise_conv1d(x, w, *, kernel_size, padding, groups, causal=False, state=None) -> (y, state_out)`.
- Numpy reference matches torch's `F.conv1d(..., groups=in_channels)`.
- Causal version produces no future leakage (test with one-hot inputs).
- VJP for autodiff (Tier 2 v1 op set extended).
- Streaming variant: passing `state=` into a sequence of length-1 calls produces the same output as one length-N call.

### [D2] `ops.online_softmax` (streaming, numerically stable) ✅

**Scope:** M (~150 LOC + ~100 LOC tests).

**Acceptance:** matches naive `softmax` to fp32 precision while accepting one chunk at a time + carry state. Required for FA-4 reference path; useful standalone.

### [D3] `ops.selective_ssm` (Mamba2 selective state-space op) ✅ (forward; VJP is follow-up)

**Scope:** L (~400 LOC + ~200 LOC tests). Depends on **D1** + **B1**.

**Acceptance:**
- Mamba2 algorithm: A/B/C/Δ projections, chunked scan (size 128), output gate.
- Replaces the placeholder reference inside `examples/advanced/Nemotron_Nano_12B_v2/`.
- VJP shipped (chunked scan adjoint is the interesting part).

### [D4] `nn.DynamicDepthwiseConv1d` Module ✅

**Scope:** XS (~50 LOC). Depends on **D1** + **B1**.

Replaces the phantom; wraps `ops.depthwise_conv1d` with state buffer.

---

## Phase E — Theme 4 KV-cache + block quantization (~1–2 weeks) — **landed 2026-05-09**

### [E1] `ops.quantize_kv` / `ops.dequantize_kv` ✅

**Scope:** S (~150 LOC + ~80 LOC tests).

**Acceptance:**
- `quantize_kv(k, v, bits=4) -> (k_q, v_q, scale, residual_bits)` — matches the algorithm sketched in `examples/advanced/kv_cache_serving/`.
- `dequantize_kv(...)` round-trips with bounded error.
- Numerical: max relative error ≤ `2^(-bits)` on N(0, 1) inputs.

### [E2] `ops.kv_cache_update` / `ops.kv_cache_read` (with `KVCacheHandle`) ✅

**Scope:** M (~200 LOC + ~150 LOC tests). Depends on **B2** + **E1**.

Functional API on `KVCacheHandle`. Replaces the legacy `kv_cache_append`/`kv_cache_prune` (which become thin shims).

### [E3] Rolling-window KV-cache state machine ✅

**Scope:** S (~150 LOC + ~100 LOC tests). Depends on **E2**.

**Acceptance:** Cache supports `evict_oldest(n)`; auto-eviction when `current_seq == max_seq`; tracked entries ≤ window size.

---

## Phase F — Tier 2 autodiff follow-ups (parallelizable, ~2–4 weeks) — **F1–F7 landed; +Phase F-MoR shipped 2026-05-10**

### [F1] Mixed-precision: autocast context + GradScaler ✅

**Scope:** M (~300 LOC + ~150 LOC tests). Independent.

**Acceptance:**
- `with ts.autodiff.autocast("fp16"): y = model(x)` casts forward inputs to fp16, accumulates in fp32 for matmul, casts back.
- `GradScaler.scale(loss); scaler.step(optimizer)` follows the standard fp32 master-copy pattern.

### [F2] Activation checkpointing (`rematerialize`) ✅

**Scope:** M (~250 LOC + ~150 LOC tests). Independent.

**Acceptance:**
- `with ts.autodiff.rematerialize(): y = expensive_block(x)` — forward stores only the recipe, recomputes during backward.
- Memory test: peak resident activations during backward < forward-only peak by a measurable margin on a 4-layer MLP.

### [F3] Custom kernel adjoints — `flash_attn` ✅, spectral ops ✅, `moe` ✅, `selective_ssm` ✅, linear / sparse / hybrid attention ✅, optimizers ✅

**Scope:** S each (~80 LOC + ~80 LOC tests per op). Independent.

Sized as A5 — derive standard analytical VJP, register via `custom_rule`.

**Status (verified 2026-05-10):** all originally-listed F3 ops + the
post-F3 follow-ups have analytical VJPs registered in
`python/tessera/autodiff/vjp.py`:

- `flash_attn` — line 590 (recompute-scores adjoint)
- `fft` / `ifft` / `rfft` / `irfft` — lines 637–667 (spectral adjoints)
- **`moe`** — line 219 (per-token routed-matmul adjoint, accumulating
  per-expert weight gradient when multiple tokens share an expert)
- **`selective_ssm`** — line 680 (Mamba2 chunked-scan adjoint that
  recomputes the forward trajectory and walks ``t = S-1 → 0``)
- `linear_attn` / `linear_attn_state` — lines 407 / 524
- `silu_mul` / `gather` / `clip` / `masked_fill` — extras shipped
  with Theme 9 / SwiGLU work
- RoPE/MLA/NSA and attention-family follow-ups — `rope`, `rope_split`,
  `rope_merge`, `ntk_rope`, `latent_kv_compress`, `latent_kv_expand_k`,
  `latent_kv_expand_v`, `mla_decode`, `mla_decode_fused`,
  `attn_sliding_window`, `attn_compressed_blocks`, `attn_top_k_blocks`,
  `deepseek_sparse_attention`, `multi_head_attention`, `gqa_attention`,
  `mqa_attention`, `gated_attention`, `hybrid_attention`,
  `lightning_attention`, `gated_deltanet`, `kimi_delta_attention`,
  `modified_delta_attention`, `power_attn`, and `retention`
- MoE transport and optimizer/RL follow-ups — `moe_dispatch`, `moe_combine`,
  `adam`, `adamw`, `momentum`, `adafactor`, `lion`,
  `ppo_policy_loss`, `grpo_policy_loss`, and `cispo_policy_loss`
- Quantization STE follow-ups — `quantize_fp8`, `dequantize_fp8`,
  `quantize_fp4`, `dequantize_fp4`, and `fake_quantize`

The original `moe 🔲` and "D3 VJP" follow-ups are both ✅ closed —
both predate this 2026-05-10 update; the doc is being refreshed to
reflect that.

### [F4] Graph IR adjoint ops ✅ — verified end-to-end on MLIR 21
ODS + pass body + per-op `buildAdjoint` impls + CMake tablegen target +
multi-output rewrite that exposes argument cotangents as additional
function outputs. `tessera-opt --tessera-autodiff` builds clean against
`/opt/homebrew/opt/llvm@21` and the lit fixture
`tests/tessera-ir/phase_f4/autodiff_pass_smoke.mlir` passes FileCheck
showing: cotangent seed (constant tensor of 1.0), two transposed matmuls
(dA = seed @ B^T, dB = A^T @ seed), multi-result return signature, and
`tessera.autodiff.arg_cotangents` annotation. Build recipe:
```bash
cmake .. -DLLVM_DIR=/opt/homebrew/opt/llvm@21/lib/cmake/llvm \
         -DMLIR_DIR=/opt/homebrew/opt/llvm@21/lib/cmake/mlir
make -j tessera-opt
./tools/tessera-opt/tessera-opt --tessera-autodiff <input.mlir> | FileCheck <input.mlir>
```

**Scope:** L (~600 LOC code + ~300 LOC tests). Foundational.

Move autodiff from numpy-reference (Tier 2 v1) to IR-level. Adds adjoint ops
to `TesseraOps.td`; teaches the lowering pipeline to materialize backward
computations as Graph IR rather than tape-walked numpy.

**Files (modify):**
- `src/compiler/ir/TesseraOps.td` — adjoint ops
- `src/transforms/lib/AutodiffPass.cpp` (new) — Graph IR adjoint generation
- `tools/tessera-opt` — register the pass + a `tessera-autodiff` pipeline

**Acceptance:**
- A `@jit` function with `@tessera.autodiff.reverse` lowers to Graph IR with adjoint ops.
- `tessera-opt --tessera-autodiff` on a forward IR produces a forward+backward IR.
- Numerical equivalence with Tier 2 v1 numpy reference.

**Decision (locked 2026-05-09):** `AdjointInterface` op trait. Each
differentiable op declares an `adjoint` method on its ODS interface; the
`AutodiffPass` walks the IR and inserts adjoint ops by interface dispatch.
Avoids doubling the op count and keeps lowering tables small. Custom adjoints
register via the same `tessera.autodiff.custom_rule(name)` Python API used in
the v1 numpy reference, with the registration also visible to the IR pass.

### [F5] Effect-aware adjoint collective insertion ✅ — full rewrite landed
Real `tessera.collective.{reduce_scatter, all_gather, all_reduce}` ops
emitted on cotangent SSA values from F4's multi-output rewrite. Per-arg
`tessera.adjoint_collective_plan` attribute records the choice. Pipeline
alias `tessera-autodiff-pipeline` runs F4+F5 together. Compiles clean.

**Scope:** M (~250 LOC). Depends on **F4**.

Extends `GPUCollectiveInsertionPass` to insert `reduce_scatter` / `all_gather`
on adjoint paths for distributed parameters.

**Acceptance:** A 2-rank mock-collective test of MLP training shows correct
gradient aggregation across ranks.

### [F6] JAX-style transforms — `vmap`, `jacrev`, `jacfwd` ✅

**Scope:** L (~600 LOC code + ~400 LOC tests). Independent.

**Status (landed 2026-05-09 via deferred-items plan Item 5):**
- `tessera.autodiff.vmap(fn, in_axes=0, out_axes=0)` — naive scan-then-stack;
  per-arg `in_axes` (int / sequence / None); `out_axes=None` returns the
  per-element list as-is.
- `tessera.autodiff.jacrev(fn, argnums=0)` — reverse-mode Jacobian; one
  backward pass per output dim. Uses Item 4's `retain_graph=True`
  re-runnable tape.
- `tessera.autodiff.jacfwd(fn, argnums=0)` — forward-mode Jacobian via the
  JVP engine in `python/tessera/autodiff/jvp.py`. v1 implementation uses
  central FD (`eps=1e-6`) for the `jvp(fn, primals, tangents)` entry
  point; the parallel JVP-rule registry is in place for 12 core ops.
  A true tape-based dual-number propagation is a Phase G perf
  follow-up that won't change the API contract.
- 15 unit tests in `tests/unit/test_jax_transforms.py` (vmap × 6,
  jacrev × 3, jacfwd × 3, JVP engine × 2, composability × 1).

The canonical `vmap(grad(fn))` per-sample-gradient pattern works as
written.

### [F7] Higher-order derivatives ✅

**Scope:** M (~400 LOC code + ~250 LOC tests).

**Status (landed 2026-05-09 via deferred-items plan Item 4):**
- `tessera.autodiff.grad(fn, argnums=0)` — JAX-style gradient
  transform. Returns ndarray for int argnums, tuple for sequence
  argnums; uses `accumulate_param_grad=False` so it doesn't leak into
  caller `Parameter.grad` slots.
- `tessera.autodiff.hvp(fn, primals, tangents, eps=1e-4)` —
  Hessian-vector product via central finite difference of `grad`.
  ~1e-6 accuracy at fp64.
- `tessera.autodiff.elementwise_grad(fn)` — per-element derivative
  for vector → vector elementwise ops; convenient for inspecting
  activation derivatives.
- `tape.backward(target, *, retain_graph=False, accumulate_param_grad=True)`
  — re-runnable tape with explicit opt-in via `retain_graph=True`.
  Required for `jacrev` (F6).
- 15 unit tests in `tests/unit/test_higher_order_autodiff.py`
  (`grad` × 6, `hvp` × 3, `elementwise_grad` × 3, `retain_graph` × 3).

True forward-over-reverse HVP (`jvp(grad(fn), x, v)`) lands when the
F6 forward-mode tape's analytical-rule path matures; the FD path is
correct for L-BFGS / natural-gradient / GAN-penalty workloads today.

### [F-MoR] Mixture of Recursions ✅ — landed 2026-05-10

**Scope:** M (~400 LOC code + ~280 LOC tests). Independent.

Bae et al. 2025 "Mixture-of-Recursions" — adaptive computation by
routing tokens through different numbers of recursive layer
applications. A learned per-token router assigns each token to a
target depth d ∈ [1, max_depth]; the layer is applied recursively to
each token until it hits its target depth, then the token's hidden
state freezes for the rest of the loop. Computational savings follow
from "easy" tokens routing to lower depths.

**Shipped 2026-05-10:**

- **Three primitive ops** in `python/tessera/__init__.py`:
  - `ops.mor_router(x, w_router, *, max_depth)` — argmax-based
    token-choice depth router; returns ``(B, S)`` int64 in
    ``[1, max_depth]``.
  - `ops.mor_partition(x, depth, *, step)` — bool mask of tokens
    whose target depth ≥ step (1-indexed).
  - `ops.mor_scatter(full, updated, mask)` — write `updated` values
    into `full` at masked positions (frozen-token semantics for the
    unselected rows).
- **VJPs** in `python/tessera/autodiff/vjp.py`:
  - `mor_router` returns zero gradients (argmax is non-differentiable;
    real router-training uses auxiliary load-balance / utilization
    losses the user adds explicitly).
  - `mor_partition` zero-grad on the int-valued depth and the
    real-valued x.
  - `mor_scatter` is linear in `updated`; gradients flow through
    `full` on the False positions and through `updated` on the True
    positions.
- **`nn.MixtureOfRecursions(layer, *, embed_dim, max_depth)`** Module —
  composes the router + recursion loop. The wrapped `layer`'s
  parameters are shared across all recursion steps (the canonical
  MoR contract).
- **ODS ops** `tessera.mor_router` / `tessera.mor_partition` /
  `tessera.mor_scatter` in `src/compiler/ir/TesseraOps.td`.
- **Lit fixture** `tests/tessera-ir/phase8/mor_primitives.mlir` —
  ODS verifier + assembly-format roundtrip.
- **17 unit tests** in `tests/unit/test_mor.py` covering forward
  correctness of all three ops + the Module + per-token recursion
  depth verified end-to-end + VJP shape contracts.

**Acceptance:**
- Per-token recursion depth is honored: with a layer that adds a
  constant, output for token i increases by exactly `depth_i * Δ`.
- `mor_partition(depth=[1,2,3,2,1], step=2)` returns
  `[F, T, T, T, F]`.
- `mor_scatter(full, updated, mask)` writes `updated` only where
  `mask` is True; the rest of `full` is preserved bit-equivalent.
- `nn.MixtureOfRecursions` rejects rank-2 inputs and `max_depth=0`.

**Phase G follow-ups** (not gating the v1 surface):
- Token-active-only kernel: gather active tokens before the layer
  call and scatter back after, instead of the v1 reference's
  apply-to-full-then-mask approach. Saves compute proportional to
  token-depth utilization.
- KV-cache-share-first / KV-cache-recursion policies for attention
  inside the inner layer (the example folder at
  `archive/examples/advanced/Tessera_MoR/` sketches the design).

---

## Phase G — NVIDIA execution path (THE BIG ONE, ~4–8 weeks) — **G1 audit landed 2026-05-09; G2–G8 sized**

The single highest-leverage block. Until this lands, the autotuner is dark,
FA-4 is unverified, the GPU-only tier is theoretical, and GPU CI is impossible.

### [G1] Audit current state — what's actually missing? ✅ (delivered at `docs/audit/nvidia_execution_audit.md`)

Per-component audit + 8-task punch list (G1-1 through G1-8). Critical path
to first H100 BF16 GEMM 128×128×128: **4–6 days** of focused work, of which
only G1-5/G1-6/G1-8 require real H100 hardware. G1-2/G1-3/G1-4/G1-7 can
land on a CUDA-only-no-H100 dev box.

### [G2] CUDA runtime backend wiring verification 📋

**Scope:** M. Likely audit shows `cuda_backend.cpp` is partially wired; finish what's missing.

### [G3] WGMMA SM_90 BF16 GEMM end-to-end 📋

**Scope:** L (~500 LOC + tests). The first real GPU execution. Pick one shape, drive it through the whole stack: Graph IR → Schedule IR → Tile IR → NVIDIA Target IR → PTX → cuBin → launch.

**Acceptance:** `@jit(target=GPUTargetProfile(isa=ISA.SM_90))` on a fixed-shape BF16 GEMM produces correct output (within fp32 tolerance) when run on a real H100.

### [G4] TMA descriptor wiring 📋

**Scope:** M. `NVTMADescriptorPass` exists but its descriptors must reach the runtime launch. Depends on G3.

### [G5] FA-4 forward verification on H100 📋

**Scope:** M. With G3+G4, run FA-4 forward on real hardware; compare against the numpy reference; iterate on tile-size tuning.

### [G6] Autotuner sweep against cuBLAS baseline 📋

**Scope:** S. With G3 done, the Bayesian autotuner finally has something to time. Run a sweep over `tile_q`/`tile_kv`/`pipeline_stages`; produce a JSON of best configs per shape.

### [G7] CI: GPU-spine equivalent of `validate.sh` 📋

**Scope:** M. CUDA-required tests gated; `scripts/validate.sh --gpu` runs the GPU subset.

---

## Phase H — Conv2d Module + remaining nn cleanup (~1 week)

### [H1] Conv2d Module — layout NHWC (decision locked) ✅

**Scope:** S.

**Decision (locked 2026-05-09):** **NHWC default** — matches existing
`tessera.ops.conv2d`. Ship a `tessera.nn.Conv2dNCHW` shim that does
`x.transpose(0, 2, 3, 1)` → `Conv2d(...)` → `out.transpose(0, 3, 1, 2)` for
torch-port code. Both forms share weight storage in HWIO (`(kH, kW, in, out)`).

**Acceptance:** `Linear`-shaped Module wrapper; `register_buffer("bias", ...)` if `bias=True`; tested forward shape.

### [H2] `LSTM` Module ✅ (state-propagation primitive shipped)
`ops.lstm_cell` returns packed `[h_t, c_t]` (single-output for v1 tape
compatibility); `ops.lstm_state_h`/`lstm_state_c` extract parts under
autodiff. VJPs registered for all three; BPTT through 2+ steps verified
against numerical Jacobian to 1e-11 at fp64. `nn.LSTMCell` (single-step
Module wrapping the primitive) and `nn.LSTM` (multi-step unroll) both
ship.

---

## Phase I — DDP / FSDP wrappers (post-F4 + F5 + G)

### [I1] `tessera.distributed.DDP(module, mesh_axis="dp")` ✅

**Depends on F4 + F5 + G.** All-reduce on adjoint path; backward triggers gradient sync.

### [I2] `tessera.distributed.FSDP(module, mesh_axis="dp")` ✅ (v1 — per-rank Module instances, sharded leading-dim, mock_collective tested)

**Depends on I1.** Sharded parameters + gather-on-forward + reduce-scatter-on-backward. `OptimizerShardPass` (Phase 5) is the underlying machinery.

---

## Out-of-scope / consciously deferred 🔲

| Item | Reason |
|------|--------|
| Module device migration (`to("cuda")`) | Requires a real device handle; tied to Phase G. |
| AIR bitcode codegen on Apple GPU | MPS+MSL covers everything we need; revisit only if a perf wall demands it. |

F6 and F7 are no longer out of scope: the reference implementations shipped
2026-05-09. Remaining work for those areas is performance/IR maturation, not
API support.

---

## Cross-references

- `docs/audit/advanced_examples_capability_gap.md` — per-example status tied
  to these phases (Theme 3 = Phase D, Theme 4 = Phase E, etc.)
- `docs/spec/AUTODIFF_SPEC.md` — Tier 2 v1 spec; Phase F lands the follow-ups
- `docs/CANONICAL_API.md` — public surface; update as each task lands
- `CLAUDE.md` Architecture Decisions #19, #21, #22 — relevant invariants
- `examples/advanced/README.md` — honest per-example status; refresh after C/D/E
- `tests/unit/test_nn_module.py`, `tests/unit/test_autodiff.py` — current test
  surface that future phases extend

---

## Phase summary (for at-a-glance scope)

| Phase | Status | LOC est | Wks | Independent? | Unblocks |
|-------|--------|---------|-----|--------------|----------|
| A — quick wins | ✅ complete | ~1,000 + 600 docs | 1–2 | ✅ all 5 tasks | Theme 1 90% closed; debugging story; autograd-via-flash_attn |
| B — protocols | ✅ complete | ~700 | 1 | sequential within | C, D, E |
| C — Theme 1 cleanup | ✅ complete | ~250 | 0.5 | ✅ within phase | BatchNorm1d, KVCache module |
| D — streaming | ✅ (D3 VJP open) | ~1,000 | 2–3 | partly sequential | Jet_nemotron, Nemotron_Nano forward |
| E — KV-cache | ✅ complete | ~700 | 1–2 | sequential within | kv_cache_serving, Fast_dLLM_v2, paged-MLA |
| F — autodiff follow-ups | ✅ all closed (F1–F7 + F3-moe + D3-VJP + Phase F-MoR) | ~2,500 | 2–4 | F1+F2+F3 ‖, F4→F5 | distributed training, mixed precision, checkpointing, JAX-style transforms, Mixture of Recursions |
| G — NVIDIA execution | 🚧 G1 audit only; G2–G7 open | ~2,000 | 4–8 | sequential within | autotuner, FA-4 verification, GPU CI |
| H — Conv2d Module + LSTM | ✅ complete | ~150 | 0.5 | indep | torch-port examples |
| I — DDP/FSDP | ✅ v1 complete | ~600 | 2 | post-F+G | distributed training at scale |

**Total ~7,900 LOC code + ~3,000 LOC tests + ~1,000 LOC docs over 12–20 weeks.**

---

## Status snapshot — 2026-05-10

**Done:** A, B, C, D (forward + D3 VJP), E, F (F1–F7 + F3-moe + Phase F-MoR), H, I.

**Remaining frontier:** **Phase G (NVIDIA execution)** — the only long pole. G1 audit is complete (`docs/audit/nvidia_execution_audit.md`); G2–G7 are open. Per the audit: 4–6 days of focused work to first H100 BF16 GEMM, of which only G1-5/G1-6/G1-8 need real H100 hardware.

**Sequenced next steps for G:**
1. **G2** — finish wiring `cuda_backend.cpp` per the audit (CUDA-only-no-H100 dev box is sufficient).
2. **G3** — first WGMMA SM_90 BF16 GEMM end-to-end (Graph IR → Schedule IR → Tile IR → NVIDIA Target IR → PTX → cuBin → launch). This is the unlock; everything downstream is sweep-on-top.
3. **G4** — TMA descriptors reach runtime launch (depends on G3).
4. **G5** — FA-4 forward verification on real H100 (needs hardware).
5. **G6** — autotuner sweep vs cuBLAS baseline (needs hardware).
6. **G7** — `validate.sh --gpu` CI spine.

**Cleanup items closed 2026-05-10:**
- ✅ **D3 VJP** — `selective_ssm` chunked-scan adjoint shipped; registered at `python/tessera/autodiff/vjp.py:680` (`vjp_selective_ssm` recomputes the forward trajectory and walks ``t = S-1 → 0`` accumulating gradients).
- ✅ **F3-moe** — MoE router `custom_rule` VJP shipped at `python/tessera/autodiff/vjp.py:219` (per-token routed-matmul adjoint with per-expert gradient accumulation).
- ✅ **F6** — `vmap` / `jacrev` / `jacfwd` shipped (deferred-items plan Item 5).
- ✅ **F7** — `grad` / `hvp` / `elementwise_grad` + re-runnable tape shipped (deferred-items plan Item 4).
- ✅ **Phase F-MoR** — Mixture of Recursions primitives + `nn.MixtureOfRecursions` Module + ODS ops + lit fixture + 17 unit tests (2026-05-10).

**Three critical chains called out in the rationale at the top of this doc — all closed:**
- B1 → C1 + D: ✅ complete (D3 VJP shipped)
- B2 → E1–E3 + C2: ✅ complete
- F4 → F5 → I (DDP/FSDP): ✅ v1 complete
