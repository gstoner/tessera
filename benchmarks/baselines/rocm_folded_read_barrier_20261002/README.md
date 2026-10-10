# gfx1201 folded typed-fragment read-barrier candidate — removed

Owner: ROCM-MXFP4-W4A8-1. Sync: ROCM-FOLDED-READ-BARRIER-2026-10-02.

The [candidate.patch](candidate.patch) materializes a full K64 slab of typed
LDS fragments before the read-completion workgroup barrier, then runs WMMA
using registers. Every wave loads before the barrier, including inactive
edge waves; the compute-only wave guard stays after it. The write-completion
barrier remains after draining the prefetched slab. No barrier is removed,
and no view is consumed after an LDS overwrite.

The candidate builds. [compiler-tests.txt](compiler-tests.txt) records
64 compiler/frontend/verifier passes. [device-tests.txt](device-tests.txt)
records 38 exact gfx1201 package numerical passes, including ragged shapes,
runtime-K image reuse and overflow/underflow/zero-scale recovery.

## Instruction-bound result

[pressure/pressure.json](pressure/pressure.json) uses explicit gfx1201 LLVM
target flags. Its diagnostic instruction stream matches the actual native
HSACO exactly. The candidate changes the selected stream versus the preserved
native reference, but instruction count remains 3911, virtual scheduled peak
165, physical 177 VGPRs/29 SGPRs, LDS 25,600 bytes, and no scratch/spills.
This provides no register-pressure improvement.

## Seven-trial graph measurements

The reference compiler is the prior active compiler, including the retained
LLVM branch hint. Its expected stale-source warning is preserved; it is an
intentional historical control. All candidate/reference/HIP package outputs
match bitwise and an independent sampled oracle before timing. Both graph
modes overwrite all three poisoned output copies. Capture/instantiation cost
is outside the graph replay windows; device-clock/event witnesses are checked.

| MxNxK | Candidate graph us/launch | Reference graph us/launch | Candidate/reference |
|---|---:|---:|---:|
| 256x4096x1024 | 23.117 | 23.015 | 1.0045 |
| 256x4096x2048 | 34.607 | 34.590 | 1.0005 |
| 256x4096x5120 | 78.823 | 77.399 | 1.0184 |
| 256x8192x5120 | 149.871 | 150.342 | 0.9969 |

The long-K N4096 row regresses about 1.8%; other rows are within about 0.5%.
The instruction changes produce no consistent gain. The candidate is removed
from active native source and its patch/pressure/timing receipts remain here.
Graph replay includes GPU graph dispatch and markers, not isolated profiler
kernel time. No Radiance comparison or hardware-counter attribution follows.

## Remaining

The previously proved producer and LLVM branch likelihood remain active.
Register live-range/code generation investigation, exact per-K32 migration,
wider short/ragged-K and model/layout coverage remain open.
gfx1151 and sibling backends receive no execution parity from this experiment.

[restored-pressure/pressure.json](restored-pressure/pressure.json) verifies the rebuilt active compiler has exactly the preserved native-reference selected instruction stream. Graphify refresh was attempted; the WSL command is unavailable.
