# Native lossless MXFP4 storage bridge

Owner: ROCM-NVFP4-INGEST-1. Sync: ROCM-INGEST-STORAGE-2026-10-05.

## Proven route and envelope

The ordinary Python frontend traces `mxfp4_folded_storage` into typed Graph
MLIR without calling the host oracle. Verified Schedule/Tile passes retain a
hashed storage/ownership contract. ROCm Target IR materializes integer GPU
operations, lowered through LLVM to a checked gfx1201 HSACO image. Common
runtime execution, image reuse and serialized descriptor replay are proved.

The named static envelope is positive N divisible by 16 and K divisible by 64.
Checkpoint-order packed E2M1 bytes become fragment-order bytes; group-major
E8M0 bytes are copied and an unsigned per-column maximum is appended. This
is a lossless byte transformation with no additional quantization. Bitwise
tests cover zero scales and all 256 byte patterns; reserved scale patterns
are storage tests, not admission evidence for numerical consumers.

Private outputs, input immutability, alias rejection, retained IR digest
checks and the complete raw descriptor geometry/direction/ordering contract
are checked before HIP submission. General uint8 remains planned/gated;
the named operation is ready only on gfx1201. Sibling execution is unsupported.

## Exact-device measurements

Tajasaurus WSL: AMD Radeon RX 9070 XT, verified gfx1201. Matching compiler
builds passed on Tajasaurus and Super-Bear. [Packet](gfx1201.json) records
compiler/source hashes, image digests, checked ABI and IR provenance.
Every timing row passes independent bitwise comparison before and after timing.

| N x K | Cold trace/compile/call wall ms | Warm checked JIT wall ms | Resident event window us |
| --- | ---: | ---: | ---: |
| 16 x 64 | 287.542 | 2.791 | 2.484 |
| 32 x 256 | 177.722 | 3.196 | 3.040 |
| 64 x 1024 | 185.404 | 3.431 | 5.316 |

Cold wall includes compilation and execution. Warm wall includes validation,
allocation, upload, module load, launch, completion and output readback.
Resident event windows exclude allocation/modules/transfers but include host
dispatch gaps; they are not isolated kernel timings or a speedup claim.

## Validation and remaining integration

- Matching gfx1201 and NVIDIA compiler rebuilds passed.
- All 32 generated documents are in sync; compiler-plan and diff checks pass.
- 40 gfx1201 storage/conversion package and JIT tests passed.
- 692 focused registry/shape/dtype/diagnostic/pass/manifest/capability tests
  passed; one skipped.
- 32 final audit/manifest/host tests passed; six exact-device cases were
  skipped on the NVIDIA-only compiler host.
- 11 NVIDIA packaging/contract regressions and 39 exact RTX 5070 device JIT
  and portable replay regressions passed with the rebuilt compiler.

The converter, bridge and packed consumer are separately compiler-owned.
An owned resident three-stage stream/arena handoff and combined timing remain
open: the current bridge API reads outputs back to the host. Shape-only
consumer packaging must preserve the physical storage contract without
inventing host payload hashes. Broader dynamic/layout/AD support and model
quality acceptance remain open.

FP8, MXFP8 and MXFP4 remain independent mandatory correctness, quality and
performance gates before a final/default choice. No selector is promoted.
Apple, gfx1151 and x86 require their own physical routes and device evidence;
RTX 5070 regression does not establish conversion parity.

Graphify is unavailable in the remote WSL checkout; no refreshed graph is
claimed. No current commit, push or PR has been made for this increment.
