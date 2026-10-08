# Native ROCm math Schedule foundation — 2026-10-06

Owner: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Synchronization: ROCM-MATH-NATIVE-SCHEDULE-2026-10-06.

## Implemented and proved

Six original textual Graph operations (sqrt, exp, add, div, cumsum, cummax)
retain authored SSA roles in a content-addressed native Schedule record.
Replay recomputes the Graph contract before building typed elementwise/scan
Tile kernels. ROCm Target lowering checks the sealed contract, operand roles,
arithmetic attributes and architecture before invoking existing native
generators and ROCDL/LLVM HSACO production. Python does not author Schedule
decisions or Tile/kernel bodies.

Current envelope: isolated static plain row-major f32 tensors; elementwise
shape preservation; inclusive last-axis scans. No new operation, dtype, pass
or native image ABI is introduced. Existing pass metadata is updated.

Matching compiler builds and 65 checks pass independently on Princess-Luna
(gfx1151) and Tajasaurus (gfx1201): 53 compiler/replay/Target checks and 12
owning-device IEEE cases. The latter compare NaN/infinity classifications
and signed zero for sqrt/div against NumPy. Changed Schedule fields, changed
Tile operands/policies, unknown arithmetic attributes and sibling targets are
rejected. Shared host diagnostic/pass/op/dtype gates: 358 pass, 17 ROCm Target
checks skipped on the NVIDIA-only build.

Each architecture has a 24-row, seven-sample/100-repeat packet: six operations,
shapes [3,17], [2,3,257], [256,1024], and reversed input roles for binary
operations. All rows pass an independent NumPy oracle before resident-event
timing and after each allocating end-to-end call. Packets record live GPU,
opaque HIP UUID, compiler binary/source hashes, image and stage IR digests
and device-library identities. Source hashes identify the actual owning
scratch checkout; pre-existing unrelated source differs across those checkouts.

## Timing domains

HIP events surround resident kernel launches. Allocating end-to-end wall time
includes module load, allocation, upload, launch, completion, readback, free
and unload; it excludes compilation. These are characterization packets,
not a matched optimization speedup. The separate package packets below measure ordinary JIT.

| Architecture | Operation, 256x1024 | Device median (us) | Allocating wall median (ms) |
| --- | --- | ---: | ---: |
| gfx1151 | sqrt | 5.248 | 3.611 |
| gfx1151 | exp | 4.860 | 4.448 |
| gfx1151 | add | 5.921 | 5.059 |
| gfx1151 | div | 5.212 | 4.415 |
| gfx1151 | cumsum | 8.511 | 3.902 |
| gfx1151 | cummax | 10.141 | 2.943 |
| gfx1201 | sqrt | 8.213 | 4.069 |
| gfx1201 | exp | 7.551 | 4.493 |
| gfx1201 | add | 4.950 | 4.959 |
| gfx1201 | div | 7.821 | 5.580 |
| gfx1201 | cumsum | 18.105 | 3.717 |
| gfx1201 | cummax | 15.959 | 4.021 |

## Ordinary JIT and portable packages

Synchronization: ROCM-MATH-NATIVE-PACKAGE-2026-10-06.

The same original frontend Graph now drives compiler-owned packages. Three
explicit image ABI IDs describe unary, binary and last-axis scan calls. The
native compiler validates role/policy/shape metadata before removing it from
physical image identity; shape-specific descriptors retain guards and bindings.
Six image families reuse byte-identical HSACO/Target products across the three
shapes and reversed binary input roles on each architecture.

48 package/JIT checks pass independently on each owning host. Warm changed
inputs forbid compiler and legacy metadata executor calls. Portable JSON replay
uses persisted original Graph aliases and the matrix adapter invokes the same
checked launch bridge. Default dtype and kind attributes omitted by the native
ODS printer retain their declared semantics. Invalid stages, shapes, scalar
types, aliases, output ownership and contracts are rejected before HIP.

Each package packet has 24 numerical rows and seven samples. Resident device
events exclude allocation/transfers. Warm JIT and portable host-array calls
include validation, allocation/upload/launch/completion/readback; compilation
and JSON deserialization are excluded. Changed values at the same input
addresses are checked after every wall measurement. These are characterization
results; the earlier direct-image timings are not a matched A/B control.

| Architecture | Operation, 256x1024 | Device median (us) | Warm JIT median (ms) | Portable median (ms) |
| --- | --- | ---: | ---: | ---: |
| gfx1151 | sqrt | 5.163 | 3.541 | 2.776 |
| gfx1151 | exp | 5.217 | 3.057 | 2.853 |
| gfx1151 | add | 5.702 | 4.702 | 4.552 |
| gfx1151 | div | 5.747 | 4.924 | 4.517 |
| gfx1151 | cumsum | 8.798 | 3.103 | 2.879 |
| gfx1151 | cummax | 10.037 | 3.170 | 2.926 |
| gfx1201 | sqrt | 5.988 | 2.894 | 2.644 |
| gfx1201 | exp | 6.277 | 2.717 | 2.476 |
| gfx1201 | add | 5.693 | 4.516 | 4.054 |
| gfx1201 | div | 4.986 | 4.305 | 4.170 |
| gfx1201 | cumsum | 8.948 | 2.973 | 2.527 |
| gfx1201 | cummax | 10.358 | 2.909 | 2.594 |

Shared diagnostic/pass/op/dtype/capability/manifest/execution/ABI/frontend/audit gates: 759 passed, 34 owning-target/package checks skipped on the NVIDIA-only host.

Evidence: package-gfx1151.json / package-gfx1201.json,
package-gfx1151-tests.txt / package-gfx1201-tests.txt,
package-shared-gates.txt and package-build logs.

## Warm-call attribution

package-gfx1201-warm-profile.txt records 30 warm ordinary JIT calls for sqrt,
add and cumsum, with an independent output check. cProfile instrumentation is
diagnostic and is not compared to uninstrumented timings. Most cumulative time
is in the HIP submission boundary (allocation/transfers/launch/completion),
rather than descriptor construction or compiler work. A next performance
experiment should isolate checked native buffer reuse and retain complete
input updates, exact image/device identity and lifetime checks.

## Exact widening and metadata-row retirement

Synchronization: ROCM-MATH-WIDENING-2026-10-06.

Original Graph casts express exact f16/bf16 widening before the math operation.
Schedule replay retains the authored casts; Tile/native loads perform widening
without allocating an intermediate tensor. Graph math shape/dtype verifiers
are unchanged. Narrow Tile input storage is admitted only under the owning
ROCm contract, so sibling lowerings keep their existing f32 envelope. Target
output_dtype is optional: absence preserves legacy same-storage output, while
checked math packages require explicit f32 output and storage-specific ABIs.

Both matching compiler builds pass. 212 checks pass on each architecture,
including changed-input ordinary JIT, portable replay, reversed binary roles,
24 IEEE classification/signed-zero cases, cast policy/shape/producer rejection
and shape-independent image checks. Positional cast dtype is a static frontend
attribute; it is never a tensor operand. Each owning architecture has a 72-row,
seven-sample/100-repeat independent-oracle packet across three input storage
types, three shapes and both binary operand orders. Image reuse is proved
within each operation/storage; storage variants retain distinct images.

Shared registry/frontend gates: 663 passed, 30 owning-backend skips.
Focused frontend/Tile gates: 73 passed, 23 tool/backend skips, including
explicit rejection of narrow ROCm Tile products under Apple/NVIDIA/x86
target labels and changed output-storage contracts.

The gfx1151 physical-math recorder now reloads native packages for all 21 rows.
Output is f32 and correctness is checked before timing. Zero metadata probes
remain in this recorder; its metadata constructor was removed. Its native
packages do not enter a comparison controlled by the legacy module cache.
36 recorder tests pass on Princess-Luna, with seven x86 checks skipped. The
physical recorder is synchronized host-wall characterization, not a kernel
speedup or promotion packet. Native kernel events are in the separate owning
72-row packets. Historical September metadata packets are preserved.

Evidence: widen-gfx1151.json / widen-gfx1201.json, widen-*-tests.txt,
widen-build-*.txt, physical-math-gfx1151.json,
widen-physical-math-tests.txt and widen-shared-gates.txt.

## Still open

General composition, broadcasting, dynamic layouts and AD remain open.
Standalone narrow-output math requires its own contract; this slice proves
explicit f16/bf16 widening followed by f32 computation/output. Host-array
allocation/transfer/completion overhead and the quantized performance programs
remain separate obligations. The historical metadata-row obligation below
was retired by the widening slice.

Apple/x86/NVIDIA physical execution is not applicable to this ROCm-owned
recipe. Their existing consumers and broader migrations retain their own
obligations; no ROCm physical schedule or timing transfers.

## Validation files

- build-gfx1151.txt / build-gfx1201.txt
- final-gfx1151-tests.txt / final-gfx1201-tests.txt
- shared-gates.txt
- gfx1151.json / gfx1201.json and device-benchmark logs

Graphify update was attempted on the authoritative WSL host; the CLI is
unavailable there (exit 127). No Graphify refresh is claimed.


## Warm host-call attribution

Reproduce with `python -m benchmarks.rocm.benchmark_math_launch_attribution
--architecture gfx1151 --output packet.json` (use gfx1201 on its owning host).
`launch-attribution-gfx1151.json` and `launch-attribution-gfx1201.json` record
18 cases each: f32/f16/bf16 inputs, sqrt/add/cumsum, shapes 3x17 and 256x1024,
and nine calls per arm. Every input changes at the same address on every call;
independent f32 oracles pass before and after every measured launch. The
recorder asserts allocation, free, copy, launch and image-lease call counts.

Median per-call instrumented API durations across the nine cases per shape:

| GPU | Shape | Allocation + free | Copies | Kernel submission | Completion | Uninstrumented portable wall |
| --- | --- | --- | --- | --- | --- | --- |
| gfx1151 | 3x17 | 0.0828 ms | 0.1656 ms | 0.0277 ms | 0.0586 ms | 0.6404 ms |
| gfx1201 | 3x17 | 0.0989 ms | 0.2148 ms | 0.0257 ms | 0.0717 ms | 0.6977 ms |
| gfx1151 | 256x1024 | 0.0209 ms | 2.5525 ms | 0.0379 ms | 0.1177 ms | 3.1432 ms |
| gfx1201 | 256x1024 | 0.0170 ms | 2.6940 ms | 0.0274 ms | 0.1059 ms | 3.1548 ms |

Wrapper durations include instrumentation overhead and are diagnostic only.
Category medians are separate summaries, not an additive wall-time model.
No optimization or kernel speedup is established. This evidence supports
checked native allocation reuse for small calls and compiler-owned resident
producer/consumer execution to reduce large transfers. Both implementations
and their matched A/B validation remain open.


## Native math allocation ownership and matched host-wall comparison

`native_movement_runtime.cpp` now supplies `tessera_rocm_math_launch` for
existing compiler-generated images. It reuses checked capacity in the same
native owner service; no input content is cached and no arithmetic is added.
Source widths, exact byte lengths, output non-overlap, alignment, checked
shape products/grid, live architecture, device/context and PID are validated.
Every input is reuploaded before every launch. Pending completion/lease errors
quarantine the owner; explicit clear establishes completion before freeing.
Retention is capped at 128 MiB per device/context arena, including unused
capacity slots. Existing explicit clear before context teardown applies.
`TESSERA_ROCM_NATIVE_MATH=0` selects the original Python launcher;
`TESSERA_ROCM_MATH_STAGING_REUSE=0` selects native allocation-per-call control.
Missing older library symbols retain the existing launcher.

The controlled HIP lifetime harness verifies growth, reuse, changed inputs,
span/alias/overflow/architecture rejection, allocation/copy failure, completion
quarantine/recovery and fork rejection. Both GPUs pass 99 numerical/staging
checks (including all three input storages, IEEE behavior and portable/JIT
execution), followed by nine ownership checks each for the final service
without redundant pre-launch synchronization. Shared runtime/ABI/diagnostic/
pass gates: 356 pass, 12 owning-device skips. Op/dtype/movement binding gates:
36 pass. The generated runtime ABI inventory is regenerated by its owner.

Reproduce: `python -m benchmarks.rocm.benchmark_native_math_staging
--architecture gfx1151 --output packet.json` (gfx1201 on its owning host).
`staging-three-arm-gfx*.json` compares native reuse, native allocation-per-call
and the original Python launcher within each process, with counterbalanced
arm order. Seven samples contain five changed-input JIT and portable calls
each. Oracles run outside timing after every call; allocation/free/reuse/
launch counters are asserted. The native and Python paths use identical
compiler images. All 18 image digests match the corresponding earlier
`widen-gfx*.json` resident HIP-event packet, keeping device timing separate.
No kernel speedup is claimed.

First final three-arm run: median per-row reuse/baseline ratios across nine
operation/storage cases per shape (lower is better):

| GPU | Shape | JIT vs native control | Portable vs native control | JIT vs Python | Portable vs Python |
| --- | --- | --- | --- | --- | --- |
| gfx1151 | 3x17 | 0.902 | 0.876 | 0.859 | 0.814 |
| gfx1201 | 3x17 | 0.887 | 0.871 | 0.858 | 0.833 |
| gfx1151 | 256x1024 | 0.992 | 0.980 | 0.966 | 0.974 |
| gfx1201 | 256x1024 | 0.998 | 0.998 | 0.988 | 0.985 |

`staging-gfx*.json` preserves the first experiment with redundant presync,
which showed negligible allocation-reuse gains; `staging-no-presync-gfx*.json`
records the subsequent two-arm experiment. Historical packets retain their
actual source/runtime hashes. Large shapes still need resident producer/
consumer integration to reduce transfer costs; these timings do not prove
that broader composition, AD or quantized performance programs are closed.

Fresh-process repeat (`staging-repeat-gfx*.json`): all 18 rows per chip
pass numerical and allocation-count checks again. Median per-row ratios:

| GPU | Shape | JIT vs native control | Portable vs native control | JIT vs Python | Portable vs Python |
| --- | --- | --- | --- | --- | --- |
| gfx1151 | 3x17 | 0.896 | 0.866 | 0.854 | 0.804 |
| gfx1151 | 256x1024 | 0.989 | 0.987 | 0.976 | 0.970 |
| gfx1201 | 3x17 | 0.895 | 0.824 | 0.859 | 0.796 |
| gfx1201 | 256x1024 | 0.987 | 0.995 | 0.977 | 0.975 |
