# Native nested NVFP4 leading maps on RTX 5070

Owner E2E-REAL-6. Sync NVIDIA-NVFP4-NESTED-MAPS-2026-10-08.

## Route and exact-device evidence

Public scalar JIT plus repeated leading vmap -> logical NVFP4 Graph MLIR ->
native Schedule flattening -> Tile -> NVIDIA Target -> PTX -> checked native
ABI -> execution. Native verification checks each logical prefix extent and
scale orientation before flattening. Descriptors retain the full logical tuple;
equal-product tuples cannot be rebound or reuse another Graph's Schedule.

Device: NVIDIA GeForce RTX 5070, GPU-cba12639-821a-7a10-4cd3-f918f9c0a545, 610.88, 12.0. LLVM/MLIR 23.1.1 assertions, CUDA 13.3.73.
The packet binds 17 source files and three compiler/runtime
binaries; hashes were rechecked after recording. The checkout is unpublished
and dirty; HEAD alone is not its source identity.

## Verification

- device.log: 53 tests pass, including all 24 owning nested cases.
- regression.log: 198 scalar/orientation/package/capability tests pass.
- prefix.log: 31 tests pass, including direct native malformed-prefix checks.
- registry.log: 490 operator/dtype/diagnostic/pass/capability checks pass.

Cases cover three coupled policies, all four operand orientations, prefix
[2,3] with M7/N5/K31 and prefix [1,2,3] with M17/N19/K129.
Changed codes/scales reuse the warm image with compiler subprocesses and
eager execution forbidden. Each public call uses one launch; retained outputs
and the original scalar Graph remain intact.

## Timing characterization

All 24 profiles pass independent decoded fp64 numerics before
timing, each public sample and final portable replay. Maximum absolute error:
0.
Seven samples per profile; 100 resident launches per CUDA-event sample.

Warm public median range: 1.313967–1.909268 ms.
Resident CUDA-event median range: 0.008635–0.012723 ms.

Public wall time includes validation, allocation, transfer and synchronized
launch. CUDA events exclude upload/readback but include dispatch gaps in the
100-launch window. These are distinct domains, not isolated kernel latency.
No counterbalanced baseline or speedup/selector-promotion claim is made.

## Remaining scope

Dynamic prefixes, mixed nested policies, nonleading axes, general composition,
encoded-storage derivatives and generic scaled-matmul batching/transpose
closure remain open. Apple/x86 lack this named contract; gfx1201 retains its
separate typed FP8 physical route. No sibling physical proof is inferred.
The full aggregate suite remains unproved green.

## Documentation delivery gates

All 32 generated documents are in sync after canonical regeneration. Eleven audit lifecycle tests pass. Graphify refresh completed from the authoritative project root: 231,907 nodes and 405,904 edges. Logs are preserved in this packet. These gates do not replace the outstanding generic full-unit closure checks.
