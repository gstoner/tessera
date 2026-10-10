# Current W1.1 producer census and exact-device replay

Owner W1.1 / FRONTEND-IR-MEDIUM-1.
Sync NVIDIA-W11-CENSUS-2026-10-06.

The two historical tensor-valued MMA constructors are in
src/transforms/lib/TileIRLoweringPass.cpp. Their rewrite registration is
restricted to sm<120. For the supported SM120 plain/canonical tensor matmul
envelope, the positive route delegates/reconstructs verified Graph then
native Schedule/Tile; it materializes pointer-backed tile.view and typed
pack/zero/MMA/unpack/store. The scheduled producer is in PMPasses.cpp.
This census does not establish generic tensor-producer closure.

[census.json](census.json) retains the named source inventory and hashes.
[producer-tests.txt](producer-tests.txt): 127 passed on current matching
LLVM/MLIR 23.1.1/CUDA 13.3 compiler and RTX 5070 sm_120. Positive device
tests cover native package execution, ragged dimensions, FP16/BF16, canonical
accumulator recovery, operand permutations, epilogues and typed Tile parity;
guard fixtures remain distinct.

[rtx5070.json](rtx5070.json): twelve current canonical tensor replay rows
pass independent references before and after five 200-launch event windows.
Compilation, recovery, package wall and device dispatch measurements remain
separate. Identical compiler lineage does not establish universal performance.

Arbitrary tensor composition, noncanonical accumulators, generic dynamic
reconstruction and older-SM physical execution remain open. ROCm, Apple and
x86 physical routes require their own architecture evidence.
