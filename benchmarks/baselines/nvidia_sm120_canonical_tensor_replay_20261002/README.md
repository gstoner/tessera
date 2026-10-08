# SM120 canonical tensor reduction migration

Owner: W1.1. Sync: NVIDIA-W1.1-CANONICAL-TENSOR-REPLAY-2026-10-02.

TileIRLoweringPass now recovers an explicitly targeted SM120 generic tensor
M/N/K matmul loop nest as one Graph contraction only after whole-function
equivalence to a fresh generic tiling replay. Replay includes the actual
zero accumulator, padding, slices, loop bounds, pipeline tokens, yields,
return lineage and bias/activation/residual epilogue. Target/architecture,
unique epilogue roles and activation sites are checked before recovery. Markers alone do not
authorize recovery. A detached replay module has its own verified MLIR pass
manager. The live Graph then descends through the registered native Schedule
and Tile passes; no Python shader, tensor-to-pointer cast or per-step package
launch is introduced.

The recovered Tile is byte-identical to the direct Graph/Schedule replay.
It has pointer-backed tile.view, typed pack/zero/MMA/unpack/store and native
masked K loads, without tensor.extract_slice, tensor padding allocation or
tile.async_copy. The scheduled checked ABI owns external resident buffers;
the Tensor reduction intermediate is eliminated before executable Tile IR.
The symbol and function signature are preserved through recovery; native
Schedule owns the subsequent entry projection and accumulator lineage.

[rtx5070.json](rtx5070.json) queries the NVIDIA RTX 5070 UUID, driver and
compute capability 12.0. Twelve fp16/BF16 rows cover 16x32x16, ragged
17x35x19 and 64x256x64 (M,K,N), both plain and bias/ReLU/residual.
Independent float32 NumPy references pass before and after timing, with
maximum absolute error below 1.6e-6. Host and resident package calls require
execution_kind=native_gpu. Source, compiler, Tensor/Tile/Target/image hashes,
entry symbols and ABI IDs are retained.

## Validation

- producer-tests.txt: 65 passes, including existing native Graph delegation,
  the recovered canonical loops, exact-device packages and tampered
  accumulator/bounds/return arithmetic refusals.
- lit.txt: the changed generic tiling fixture passes all RUN lines, including
  older SM90 tensor construction and the explicit-target SM120 gate.
- registry and adjacent scheduled routes are recorded separately.

## Timings

Each resident sample uses a C++ 200-launch loop bracketed by CUDA events,
after 20 warmup launches. Five samples are retained. This event window
includes any C++/driver dispatch gaps; it is not isolated instruction time.
Host package wall time includes transfers, allocations and module lifecycle.
Initial direct Graph-to-Tile, generic tiling, verified recovery-to-Tile and
native packaging costs are separate. These characterize the migration;
there is no speedup comparison or throughput promotion.

| MxKxN | Storage | Epilogue | Resident event-window us/launch | Package wall ms | Recovery-to-Tile ms |
|---|---|---|---:|---:|---:|
| 16x32x16 | fp16 | none | 11.228 | 0.344 | 10.199 |
| 16x32x16 | fp16 | bias/ReLU/residual | 11.084 | 0.464 | 11.344 |
| 16x32x16 | bf16 | none | 11.898 | 0.375 | 10.798 |
| 16x32x16 | bf16 | bias/ReLU/residual | 8.104 | 0.451 | 11.074 |
| 17x35x19 | fp16 | none | 10.693 | 0.339 | 11.202 |
| 17x35x19 | fp16 | bias/ReLU/residual | 13.656 | 0.473 | 11.147 |
| 17x35x19 | bf16 | none | 10.173 | 0.356 | 10.800 |
| 17x35x19 | bf16 | bias/ReLU/residual | 17.084 | 0.563 | 11.039 |
| 64x256x64 | fp16 | none | 8.734 | 0.366 | 10.815 |
| 64x256x64 | fp16 | bias/ReLU/residual | 7.924 | 0.442 | 10.758 |
| 64x256x64 | bf16 | none | 9.179 | 0.335 | 11.259 |
| 64x256x64 | bf16 | bias/ReLU/residual | 17.398 | 0.553 | 10.654 |

## Reproduce

On the owning Super-Bear WSL checkout with matching compiler/runtime:

    source .build-sm120-w1-1/validation-env.sh
    .venv/bin/python benchmarks/nvidia/benchmark_canonical_tensor_replay.py --samples 5 --reps 200 --output benchmarks/baselines/nvidia_sm120_canonical_tensor_replay_20261002/rtx5070.json

## Remaining

This migration proves a replay-equivalent static single matmul function,
including the tested ReLU epilogue. Arbitrary tensor producer graphs,
noncanonical reductions/initial accumulators, other activation device proof,
dynamic generic loop reconstruction and older SM architectures remain open.
The two historical tensor-valued constructors remain only for sm<120;
their older-target artifact fixtures do not establish hardware execution.
Sibling backends have no parity from RTX 5070 evidence.
