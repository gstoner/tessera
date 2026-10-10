# SM120 row-major RHS typed fragment materialization

Owner W1.1; sync NVIDIA-ROW-MAJOR-B-CORE-2026-10-03.

A native row-major tensor producer can use explicit transpose intent to gather its logical KxN storage into the typed column-major B register contract. Each fp16/BF16 register gathers two K elements separately at the physical leading dimension. Complete views support unbounded gathers; ragged views keep bounds, safe-address-before-load and zero selection. The typed K-loop accumulator and output store remain unchanged. Missing transpose intent and incompatible physical storage refuse.

## Evidence boundary

The recorder derives a seed native Tile function, changes only physical B view strides/memory and transpose intent, and removes the inherited Schedule hash. It compiles native Tile/NVIDIA Target/NVVM/LLVM/PTX and launches resident GPU pointers. This is a raw physical experiment, with no replay-certified production Schedule or checked row-major RHS package claim.

Sixteen exact RTX 5070 cases cover fp16/BF16, contiguous and five-element padded pitches, complete fragments and ragged shapes with multi-panel K. Independent fp32 matmul checks run before and after three CUDA-event dispatch windows. Poison values fill padding to expose pitch errors. No producer stage is timed here; only consumer dispatch windows are recorded. There is no speedup or selector promotion claim.

The source census confirms LowerMatmulToTileMMA and LowerKReductionAddToTileMMA are registered only for sm<120. Existing canonical SM120 recovery delegates through native Graph/Schedule/Tile typed views and fragments. This does not close arbitrary producers, noncanonical accumulators or older architecture consumers.

Next: bind row-major RHS selection in native Schedule identity/replay, extend checked descriptor/host ingress, and connect one named normalization RHS producer with buffer lifetime/stream checks and separate producer/consumer timing. FP8/MXFP8/MXFP4 remain required gates before any final strategy decision; this core change admits only fp16/BF16.

Files: rtx5070.json, run.txt, build-final.txt, contracts.txt, census.json, final validation and drift logs.

## Recorded result

16 physical cases passed, with maximum absolute error 3.87430191e-07. Compiler/producer/registry gates passed 367 tests; existing native GPU routes passed 86 tests; audit gates passed 11 tests. Both native tools were rebuilt. All 32 generated documents were regenerated, with compiler-plan, Ruff and diff gates passing. Graphify remains unavailable in WSL.
