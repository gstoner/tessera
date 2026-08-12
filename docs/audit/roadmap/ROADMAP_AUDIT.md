---
last_updated: 2026-08-11
audit_role: theme
---

# Roadmap Audit

This document consolidates execution-roadmap, deferred-item, and sprint/crosscut
planning material.

> **Status authority.** Roadmap docs are sequencing + rationale, never status
> truth (see *Still Open* below). Live status traces to the generated
> dashboards (Decision #26): cross-surface state in
> [../generated/surface_status.md](../generated/surface_status.md), per-axis
> S-series state in [../generated/s_series_status.md](../generated/s_series_status.md),
> executable runtime pathways in
> [../generated/runtime_execution_matrix.md](../generated/runtime_execution_matrix.md),
> and the all-up open-work queue in [../MASTER_AUDIT.md](../MASTER_AUDIT.md).

## Finished

- The old broad execution roadmap established phase ordering and acceptance
  criteria.
- Deferred-item plans captured rationale for scope decisions instead of leaving
  them implicit.
- Sprint/crosscut docs identified recurring issues such as status strictness,
  backend kernel gates, tensor attributes, and GA6 complexity.

## Still Open

The remaining roadmap is narrower than the historical sprint plans imply:

1. **E2E-REAL-6 compiler authority.** Promote the tracer to the sole general
   frontend, move family planning out of `JitFn`, require every physical package
   to consume its content-addressed Schedule/Tile parent, and delete retained
   Graph-to-backend reconstruction only after differential and exact-target
   proof. The first native-JVP and scheduled-attention cohorts have landed; this
   is not yet a whole-compiler closeout.
2. **General structured programs.** Bounded `if`, counted `for`, canonical
   bounded `while`, and forward `control_scan` are real. Open work is general
   source-CFG recovery, multi-block regions, a typed Presburger carrier,
   region-level scan products, and lowering measured SAVE/RECOMPUTE/HYBRID
   choices into generated adjoints.
3. **Distributed execution and propagation.** The core collective Target rows
   remain reference until native multi-rank transport executes. The generated
   domain-sharding queue needs local-layout, propagation, mock-mesh, or native
   multi-rank proof; it is not one undifferentiated kernel backlog.
4. **Measured architecture promotion.** Clean target clocks, resource vectors,
   and exact-device packets remain prerequisites for selector changes. Evidence
   does not transfer between x86, Apple, ROCm, and NVIDIA architectures.
5. **Tiled SSD.** The shared Tile schedule described below is still absent even
   though existing sequence-mixer runtime lanes execute.
6. **Block-local training.** Decoupled scheduling and memory accounting landed
   as hardware-free scaffolding. Real block-parallel placement, local-objective
   verification, and trainer integration remain open; see the design note below.

## Standalone Compiler Roadmap Baseline

The archived execution roadmap established the **Standalone compiler milestone
sprints (S-series)**. The list below is the retained scope baseline, not the
active status or ownership contract; use the routing table in *Active ownership*
for current work.

S-series checkpoints:

- [S0] Scope lock and standalone compiler boundaries.
- [S1] Native primitive contract registry.
- [S2] Tensor algebra, indexing, and scalar math.
- [S3] Pytrees, module state, and model containers.
- [S4] Explicit RNG and stochastic effects.
- [S5] Control flow and transform composition.
- [S6] Native sharding, collectives, and distributed semantics.
- [S7] Flax-level model primitive library.
- [S8] Tiny standalone model conformance suite.
- [S9] Numerics, mixed precision, and quantization.
- [S10] Optimizer library and training-step primitives.
- [S11] Loss / criterion library.
- [S12] State serialization and checkpointing.
- [S13] Custom-primitive / extension API.
- [S14] Compilation cache and AOT export.
- [S15] Native data pipeline.

Scope decisions from S0 remain active:

- Tessera is runtime-independent of PyTorch, JAX, and Flax. They are reference vocabularies
  for supported ops, not runtime dependencies.
- The data pipeline is in scope, including compatibility vocabulary from
  `tf.data`, `torch.utils.data`, and `grain`.
- The training step is in scope.
- Custom-primitive authoring is in scope.
- AOT export and persistent compilation cache are in scope.

The roadmap explicitly covers broad model families including diffusion, xLSTM,
Mamba, Hyena, Linformer, cosFormer, Griffin, Megalodon, JEPA, and Titans/Atlas.

## Active ownership

| Need | Owner | Live evidence |
|---|---|---|
| Compiler sequencing and authority consolidation | [`../compiler/README.md`](../compiler/README.md) then [`../compiler/INTEGRATED_COMPILER_PLAN.md`](../compiler/INTEGRATED_COMPILER_PLAN.md) | [`../generated/compiler_progress.md`](../generated/compiler_progress.md) |
| Per-operation phase state | generated support and single-GPU triage | [`../generated/support_table.md`](../generated/support_table.md) and [`../generated/single_gpu_closeout.md`](../generated/single_gpu_closeout.md) |
| Primitive transform/sharding contracts | integrated compiler plan plus the domain owner | [`../generated/s_series_status.md`](../generated/s_series_status.md) |
| Executable runtime and ABI evidence | applicable backend `todo.md` | [`../generated/runtime_execution_matrix.md`](../generated/runtime_execution_matrix.md) and [`../generated/runtime_abi.md`](../generated/runtime_abi.md) |
| Apple exact-device work | [`../backend/apple/todo.md`](../backend/apple/todo.md) | Apple target map and packets |
| NVIDIA exact-device work | [`../backend/nvidia/todo.md`](../backend/nvidia/todo.md) | NVIDIA target maps and packets |
| ROCm exact-device work | [`../backend/rocm/todo.md`](../backend/rocm/todo.md) | ROCm target map and packets |
| x86/AVX-512 and AMX work | [`../backend/x86/todo.md`](../backend/x86/todo.md) | x86 target map and packets |
| Cross-surface narrative | [`../MASTER_AUDIT.md`](../MASTER_AUDIT.md) | generated files remain count authority |

Keep this document as a routing and decision-provenance hub. Historical
S-series, contract-pass, ReplaySSM, and single-GPU closeout plans must not remain
active owners after their implementation boundary has closed. Archive a roadmap
fragment once its durable decision and any genuine follow-up have been routed to
the owners above; do not duplicate status counts or runtime claims here.

Archived in this cleanup:

- [`archive/CONTRACT_PASS_PLAN.md`](archive/CONTRACT_PASS_PLAN.md) — all
  registered consumers are live;
- [`archive/CONTROL_FLOW_AND_DEEPSEEK_ACCELERATION_PLAN.md`](archive/CONTROL_FLOW_AND_DEEPSEEK_ACCELERATION_PLAN.md)
  — structured execution and model physical closure now have separate owners;
- [`archive/REPLAYSSM_PLAN.md`](archive/REPLAYSSM_PLAN.md) — the serving ABI is
  complete; it remains an oracle/candidate for tiled SSD;
- [`archive/SINGLE_GPU_CLOSEOUT_PLAN.md`](archive/SINGLE_GPU_CLOSEOUT_PLAN.md) —
  its generated triage survives under current compiler/backend ownership;
- [`archive/S_SERIES_ENABLEMENT_MAP.md`](archive/S_SERIES_ENABLEMENT_MAP.md) and
  [`archive/S_SERIES_GAP_CLOSURE_PLAN.md`](archive/S_SERIES_GAP_CLOSURE_PLAN.md)
  — historical taxonomy and enablement execution.

## Active design — tiled SSD Schedule→Tile family

- **Tiled fused SSD (Mamba-2) as a Schedule→Tile family** — activated
  2026-08-11. SSD is matmul-dominant by construction, so its fusion belongs at
  Tile IR as a tiled GEMM schedule with the matmul intrinsic selected per backend
  (`simdgroup_matrix` / WGMMA / WMMA/MFMA / AVX-512), **not** as a one-off
  backend kernel.
  The current Apple `selective_ssm` (chunked-parallel, 3 MPS-`bmm` + host) stays
  as the functional reference; the naive per-channel Apple fused kernel is an
  explicit anti-pattern (loses the cross-channel gram sharing → slower than the
  3-`bmm` path). The **tiled SSD schedule itself is not implemented on any
  target**: existing x86/ROCm `selective_ssm` execution is evidence for those
  existing lanes, not proof for this new schedule. Each eventual Apple, ROCm,
  and NVIDIA lowering needs its own artifact, native-execution, and oracle proof.
  Full design + sequencing + acceptance criteria:
  [`docs/architecture/proposals/tiled_ssd_tile_ir_schedule.md`](../../architecture/proposals/tiled_ssd_tile_ir_schedule.md).

## Source Material Consolidated

- `archive/execution_roadmap.md`
- `archive/deferred_items_plan.md`
- `archive/sprint_plan_task4_and_crosscuts.md`

---

## Design note — Decoupled-stage pipeline + decoupled-block memory

> Consolidated from the former standalone `decoupled_stage_pipeline.md` (Decision #26 — fewer audit entry points; merged 2026-06-26).

**Status:** experimental scaffolding landed (planner + memory-model modes,
hardware-free, test-gated). Real block-parallel device execution is **not**
proven — see *Still Open*. Theme owner: `compiler` + `roadmap`. Date: 2026-06-21.

Provenance: ideas harvested from *DiffusionBlocks: Blockwise Training for
Generative Models* (Shing/Koyama/Akiba, arXiv:2506.14202, ICLR 2026). This note
covers the two scheduling/memory levers (review items #2 and #3); the numerical
primitives (#1 equi-probability band partition, #4 EDM preconditioning) and the
#5 conformance fixture landed separately in
`python/tessera/compiler/diffusion_schedule.py`,
`python/tessera/compiler/denoise_reference.py`, and the primitive-coverage
registry.

### The idea

End-to-end backprop couples pipeline stages: stage *k+1*'s forward consumes
stage *k*'s activations, and the backward pass must wait for the whole forward
to finish — this is what forces the 1F1B fill/drain **bubble** and the
activation stash across stages.

DiffusionBlocks observes that a residual network is an Euler discretization of a
reverse-diffusion ODE, so each **block** can be reinterpreted as one denoising
step with a **self-contained, data-derived target** (the clean signal), trained
against a local EDM denoising objective. Blocks then have **no cross-stage
forward/backward activation dependency**. Two consequences a compiler can model:

1. **Scheduling (item #2):** stages can run fully asynchronously — no fill,
   no drain, **zero bubble**. This is a genuinely different schedule mode from
   1F1B / interleaved-1F1B.
2. **Memory (item #3):** only one block's activations are ever live, so peak
   training activation memory is ≈ `total / num_blocks` — a *structural* lever,
   **orthogonal to gradient checkpointing** (which trades compute for the
   backward live-set; this trades nothing, it partitions which block trains).

The paper's *proven* win is the memory reduction (B× less, matching/beating
end-to-end on CIFAR/ImageNet/LM1B). The paper does **not** demonstrate an actual
multi-device block-parallel run — it trains one randomly-chosen block per step
sequentially. That gap is exactly where Tessera, as a compiler with an explicit
pipeline/distribution planner, has something to add.

### What landed (hardware-free, test-gated)

- **`pipeline_planner.PipelinePlan(decoupled=True)`** — a decoupled-stage
  schedule: `bubble_fraction == 0`, `warmup_steps == 0`, `total_clocks == 2·m`,
  and `_build_decoupled()` emits a schedule where every rank runs F then B on
  its own micro-batch with no cross-stage handoff (asserted in
  `tests/unit/test_pipeline_stage_insertion.py::TestDecoupledStage`). Mutually
  exclusive with `interleaved`. Serialized via `to_mlir_attrs()` (`decoupled =
  true`).
- **`checkpoint.CheckpointPolicy.DECOUPLED_BLOCK` + `num_blocks`** —
  `peak_activation_fraction() == 1/num_blocks` and
  `estimated_peak_activation_bytes(total)`, the B=2/4/6 trade the paper ablates
  (`tests/unit/test_checkpoint_decorator.py::TestDecoupledBlockPolicy`). The
  saving is structural (no recompute markers emitted), explicitly distinct from
  the SELECTIVE/FULL recompute levers.

These are the *scheduling/memory duals* of the same block-local-training regime:
`decoupled=True` (when does each block run) and `DECOUPLED_BLOCK` (how much
activation memory it costs).

### Still open (not proven)

- **No real block-parallel device mapping.** The planner emits a zero-bubble
  schedule and the MLIR attr, but no `PipelineStageInsertionPass` /
  distribution-planner path yet places decoupled stages on distinct devices and
  runs them concurrently. `distributed_planner` stage assignment (`pp_stage`) is
  unchanged; the decoupled mode is a *schedule-level* claim only.
- **No training-loop integration.** The per-block local objective (EDM denoise
  target + equi-probability σ-band → block dispatch) is available as primitives
  but is not wired into an end-to-end trainer; correctness of block-local
  training on a real Tessera model is unproven.
- **Validity gate.** `decoupled=True` / `DECOUPLED_BLOCK` are only sound for
  genuinely decoupled per-stage objectives (the diffusion/denoising reframing or
  another local objective) — **not** standard end-to-end backprop, where the
  cross-stage dependency is real. Nothing enforces this precondition yet; it is
  the user's contract. A verifier check ("is this region trained with a local
  objective?") is the natural next gate.

### Suggested next steps (priority order)

1. Wire the equi-probability band schedule + EDM denoise target into a small
   reference block-local trainer over `denoise_reference.py`, and show per-block
   loss converges independently (extends the #5 conformance fixture from
   inference to training).
2. Teach `distributed_planner` / `PipelineStageInsertionPass` to place decoupled
   stages on distinct ranks and prove (lit) that no cross-stage activation
   collective is inserted.
3. Add the local-objective precondition verifier so `decoupled=True` cannot be
   silently applied to a backprop-coupled graph (Decision #21 stable-diagnostic
   pattern).
