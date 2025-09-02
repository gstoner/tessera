
<!-- MERGE:BEGIN Tessera_Autotune_Architecture.md -->
# Tessera Autotune Architecture (Part 1/2)

## Overview
Tessera Autotune is a hybrid system:
- **Compiler-side (MLIR)**: tunable attributes on ops, transform-dialect meta-schedules, static legality/cost checks.
- **Runtime-side (Service)**: search algorithms (grid/random/hyperband), on-device measurement, and a **SQLite cache** keyed by schedule.

This split lets us reuse MLIR analyses while iterating quickly on search and profiling.

## IR-Tunable Contract
Each tunable kernel carries a **TunableSet** of named knobs (e.g., `BLOCK_M/N/K`, `num_warps`, `num_stages`, `vector_bytes`). Ranges live as attributes on the op. The autotuner chooses a **concrete assignment**; the pass materializes tiling/pipeline choices from them.

### Example (conceptual)
```
%gemm = tessera.gemm ... {
  tessera.tunable_list = ["BLOCK_M","BLOCK_N","BLOCK_K","num_warps","num_stages"],
  tessera.BLOCK_M = [64,128,256],
  tessera.BLOCK_N = [64,128,256],
  tessera.BLOCK_K = [32,64,128],
  tessera.num_warps = [4,8,16],
  tessera.num_stages = [1,2,3]
}
```
