# Decoupled-Stage Pipeline + Decoupled-Block Memory (design note)

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

## The idea

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

## What landed (hardware-free, test-gated)

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

## Still open (not proven)

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

## Suggested next steps (priority order)

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
