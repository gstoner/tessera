<!-- MERGE-START: Reasoning_Pattern_Library.md -->
# Reasoning Pattern Library

Patterns encoded from your CoT document (resource-constrained, performance-driven, mixed precision) and domain modeling (Doc‑2):

## Memory-Bound Pattern (Preset: `--pddl-cot-preset=memory-bound`)
- Diagnose: low arithmetic intensity, high bytes/FLOP
- Actions: `apply-async-copy-pipeline`, `apply-shared-memory-tiling`, `optimize-layout`
- CoT Hints:
  - Echo `shared_memory_bytes` and compare to `limit_shared_memory_bytes=228000` (Hopper) ￼
  - Prefer streaming K dimension; double-buffer loads (Ampere cp.async / Hopper TMA)
  - Track occupancy and bank conflicts

## Compute-Bound Pattern (Preset: `--pddl-cot-preset=compute-bound`)
- Diagnose: high arithmetic intensity
- Actions: `apply-tensor-core-mapping (WMMA/WGMMA)`, `increase-tiles`, `unroll`
- CoT Hints:
  - Emit feasibility: shape divisibility checks for WMMA/WGMMA
  - Track `registers_per_thread <= 255` and occupancy targets

## Mixed-Precision Pattern (Preset: `--pddl-cot-preset=mixed-precision`)
- Storage vs Accumulation tradeoffs (FP8/BF16/FP16 vs FP32)
- Actions: `select-precision(storage,accum)`, `insert-quant-dequant`, `loss-scaling` (training)
- CoT Hints:
  - Record numeric targets: `error <= 1e-6`, track overflow/underflow flags
  - Prefer FP32 accumulation for stability; compress KV/cache tensors

Each preset has a pass flag example:
```
-tessera-pddl-infer-plan='mode=prove-as-you-go,preset=memory-bound'
```
<!-- MERGE-END: Reasoning_Pattern_Library.md -->
