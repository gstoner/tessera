# gfx1201 native folded runtime-M/N/K image identity

Owner: ROCM-MXFP4-W4A8-1 / E2E-REAL-6-ROCM-MATMUL-CACHE.
Sync: ROCM-FOLDED-RUNTIME-K-2026-10-02.

The native folded frontend now defaults to runtime M/N/K image reuse within
its checked positive K64 and whole/partial M/N panel classes. Graph, Schedule
and Tile remain launch-owned and shape/payload-bound. The native Target
projection retains every physical schedule key and sets k/scale_k/macro_k to
zero under a unit runtime_k marker, meaning the full runtime K extent; the
physical stage remains K64. The typed LDS loop carries one f32 accumulator
through all stages and applies the row-reference epilogue once after full K.
An LLVM assume exposes K>=64 only after the runtime ABI checks admit positive
K64 multiples. Buffer dimensions, storage, payload hashes, output capacity and
launch geometry are checked before device probing.

[gfx1201.json](gfx1201.json) records actual RX 9070 XT/gfx1201, native compiler
and source fingerprints, matched physical schedules, output hashes and seven
alternating device-clock windows. It was regenerated through the default route.
Three rotating inputs, bitwise full-output parity and an independent sampled
oracle precede timing. Public runtime wall time includes allocation, transfers,
module lifecycle and completion; it is separate from device-window time.
There are no profiler counters or Radiance comparison.

| MxNxK | Runtime native us | Static native us | HIP us | Runtime/static | Runtime/HIP |
|---|---:|---:|---:|---:|---:|
| 256x4096x1024 | 25.753 | 25.768 | 21.198 | 0.9994 | 1.2148 |
| 256x4096x2048 | 36.659 | 36.491 | 31.921 | 1.0046 | 1.1484 |
| 256x4096x5120 | 78.649 | 77.216 | 72.460 | 1.0185 | 1.0854 |
| 256x8192x5120 | 150.033 | 150.688 | 146.742 | 0.9957 | 1.0224 |

All four cases reuse one HSACO payload:
71172227d6182508bad3bc423cbf2e4c444cd6ca47994fb85a2261546f3818cf.
The first package is cold and later cross-K/N packages hit the native image
cache; frontend lowering and payload binding still take time. The packet
records their costs, so warm image reuse is not a zero-cost package claim.

The runtime image uses 177 VGPRs/29 SGPRs, static-native 177/24 and HIP 123/54;
each uses 25600 LDS bytes and no scratch/spills. Runtime K does not resolve the
native/HIP performance gap, especially short K. Resource counts do not locate
peak register liveness. Keep register/column-cost attribution, exact per-K32
native migration, wider layouts and model-quality evidence open.

[device-tests.txt](device-tests.txt): 91 passed before default-route promotion.
[default-device-tests.txt](default-device-tests.txt): 91 passed after promotion.
These overlap and must not be added. They include seven random/ragged shapes
in both fixed/runtime-K modes, overflow/underflow/zero-partial recovery,
cross-M/N/K image identity versus static controls, invalid K64 extents and
insufficient capacity rejected before HIP. [shared-identity-tests.txt](shared-identity-tests.txt): 72 passed, 9 skipped
(other-architecture envelopes); [physical-key-tests.txt](physical-key-tests.txt):
10 passed in fixed/runtime-K modes. [host-tests.txt](host-tests.txt): 329 passed.
[final-gates.txt](final-gates.txt): 311 audit/navigation/registry gates passed.
These suites overlap and are not a combined total.

Fixed-K and static native controls remain available using runtime_k=False and
runtime_mn=False/runtime_k=False, respectively. Explicit approximate policy is
still required; this is not an exact per-K32 MXFP4 contract or a default public
logical MXFP4 route. gfx1151 is not applicable: RDNA3.5 lacks FP8 WMMA. No sibling
backend physical or numerical parity is inferred.

## Reproduce in the owning WSL checkout

Use the matching native compiler and LLVM tools after sourcing scripts/_rocm_env.sh.
Set TESSERA_ROCM_CHIP=gfx1201, TESSERA_OPT to the built compiler, and
PYTHONPATH to the repository python directory and root. Run the recorder with:

    --tessera-opt "$TESSERA_OPT" --llvm-bin "$LLVM_BIN"
    --native-static-control --trials 7
    --case prefill:256x4096x1024 --case prefill:256x4096x2048
    --case prefill:256x4096x5120 --case prefill:256x8192x5120
    --output benchmarks/baselines/rocm_folded_native_runtime_k_20261002/gfx1201.json

Recorder: benchmarks/rocm/record_gfx1201_folded_native_package.py.
Use --no-native-runtime-k to reproduce the prior fixed-K image contract.

Graphify update was attempted in the authoritative WSL checkout; the command is unavailable there. This does not change the compiler/device validation receipts.
