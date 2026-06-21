# Roadmap Audit

This document consolidates execution-roadmap, deferred-item, and sprint/crosscut
planning material.

## Finished

- The old broad execution roadmap established phase ordering and acceptance
  criteria.
- Deferred-item plans captured rationale for scope decisions instead of leaving
  them implicit.
- Sprint/crosscut docs identified recurring issues such as status strictness,
  backend kernel gates, tensor attributes, and GA6 complexity.

## Still Open

- Roadmap docs should no longer be used as status truth.
- Deferred items need to be reattached to the canonical theme that owns the
  proof.
- Long historical sprint documents should not compete with generated
  dashboards or theme audits.
- **Block-local training (DiffusionBlocks, arXiv:2506.14202):** decoupled-stage
  pipeline scheduling + decoupled-block memory accounting landed as
  hardware-free, test-gated planner/checkpoint modes; real block-parallel device
  execution and training-loop integration are still open. See
  [`decoupled_stage_pipeline.md`](decoupled_stage_pipeline.md).

## Standalone Compiler Roadmap Contract

The archived execution roadmap established the **Standalone compiler milestone sprints (S-series)**.
The active contract is preserved here so tests and readers do not need a root
redirect to the historical file.

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

## Next Work

1. Keep roadmap material as planning provenance.
2. Move active work into compiler/backend/coverage/domain audit docs.
3. Delete or archive roadmap fragments once their decisions are represented in
   the owning theme audit.

## Deferred — sequenced to the NVIDIA/AMD backend timeline

- **Tiled fused SSD (Mamba-2) as a Tile-IR schedule** — decided 2026-06-07,
  deferred. SSD is matmul-dominant by construction, so its fusion belongs at
  Tile IR as a tiled GEMM schedule with the matmul intrinsic selected per backend
  (`simdgroup_matrix` / WGMMA / MFMA), **not** as a one-off Apple Metal kernel.
  The current Apple `selective_ssm` (chunked-parallel, 3 MPS-`bmm` + host) stays
  as the functional reference; the naive per-channel Apple fused kernel is an
  explicit anti-pattern (loses the cross-channel gram sharing → slower than the
  3-`bmm` path). Apple is the *executable validation backend* for the schedule;
  NVIDIA/AMD lowerings inherit it. Full design + sequencing + acceptance criteria:
  [`docs/architecture/proposals/tiled_ssd_tile_ir_schedule.md`](../../architecture/proposals/tiled_ssd_tile_ir_schedule.md).

## Source Material Consolidated

- `archive/execution_roadmap.md`
- `archive/deferred_items_plan.md`
- `archive/sprint_plan_task4_and_crosscuts.md`
