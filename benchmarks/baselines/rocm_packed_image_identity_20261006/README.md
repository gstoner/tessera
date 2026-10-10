# Packed gfx1201 shape-independent image proof

Owner ROCM-NVFP4-INGEST-1 / ROCM-MXFP4-W4A8-1.
Synchronization key ROCM-PACKED-IMAGE-IDENTITY-2026-10-06.

Exact device: AMD Radeon RX 9070 XT / gfx1201, Tajasaurus.
Matching native compiler: LLVM/MLIR 23.1.1.
The rebuilt ROCm header admits native fixed-K runtime-M/N packed projection.
Original Graph/Schedule/Tile shapes remain static and separately certified;
the image digest binds the projected Target actually compiled.
Two shape pairs share entry symbols, Target text, image digest and HSACO bytes.
Whole/partial M/N classes and K remain distinct image identities.

## Validation

- compiler-tests.txt: five native projection tests.
- device-tests.txt: eleven exact-device cases.
- regressions.txt: 37 combined compiler/device/existing resident regressions.
- Independent folded BF16 oracle: bitwise agreement before and after timing.
- Invalid image K, shape policy, tile-class booleans and storage/decode contracts
  refuse launch; changed packed scale bytes also refuse.
- Existing static three-stage resident program regression remains intact.

## Matched characterization

Seven alternating trials per shape/arm; 1,024 resident launches per
marker-bracketed HIP graph window. Allocating checked-runtime wall time is
recorded independently and includes copies/module lifecycle.
Ratios are projected divided by static; lower is faster.

| M x N x K | Device ratio | End-to-end ratio |
| --- | ---: | ---: |
| 128 x 32 x 256 | 0.9998 | 0.9895 |
| 200 x 80 x 256 | 0.9948 | 1.0058 |
| 256 x 64 x 256 | 0.9973 | 1.0013 |
| 512 x 128 x 256 | 0.9947 | 1.0042 |
| 200 x 80 x 2048 | 0.9966 | 0.9964 |
| 256 x 80 x 2048 | 0.9884 | 0.9891 |

This single-session packet establishes correctness and image identity reuse,
not performance/default promotion or dynamic semantic Graph support.
The packed consumer is the measured stage; native ingest/storage are covered
only by their existing static regression here.

## Remaining integration

Authored versus projected lineage is now integrated into the three-stage
resident and portable program. General dynamic Graph/packing, wider layouts,
native runtime ownership below Python and full five-slice closure remain open. No gfx1151, NVIDIA, Apple or x86 device evidence is inferred.

## Reproduction

On Tajasaurus in scratch/next-five-compiler-slices-rocm, source
.build-gfx1201-current/validation-env.sh. Run the three named pytest files
with .venv-movement-capture/bin/python.
Run benchmarks.rocm.benchmark_packed_image_identity with --output packet.json
and --llvm-bin /home/angstorms/scratch/gfx1201-scheduled-norm-edge/.toolchain/llvm-23/bin.

## Resident and portable integration follow-through

52 combined compiler/device/frontend/resident tests pass
(resident-integrated-regressions.txt). Native image projection is now the
resident consumer default. Complete original Target attributes are compared
with projected attributes, allowing exactly runtime M/N and whole/partial
classes, name/hash normalization and the compiler's pipeline metadata.
The static Graph/Schedule/Tile certificate remains per shape.
Tampering with authored stage-K while recomputing its hash is rejected.
The existing portable v1 stage schema carries the retained authored Target
in descriptor provenance, so replay requires neither compiler nor host weight
conversion. Historical static-image construction and replay remain tested.

resident-projected.json executes three shapes through resident stages,
ordinary JIT and portable replay on RX 9070 XT. Per-stage converter, storage,
consumer and combined graph/event times stay separate from activation-reuse,
allocating combined and restore/replay wall times. The lossless storage bridge
and conversion statistics are numerically checked alongside output.

resident-static-control.json retains static consumer images for the same
seeded operands. Runs are sequential characterization, not alternating matched
performance admission. Compiler image reuse is the established architectural
result; no GPU/default performance promotion is claimed.

## Native lifecycle follow-through

Synchronization key ROCM-NVFP4-NATIVE-OWNER-2026-10-06.
Five C ABI exports own preparation, updates, invocation, read and close.
The explicit native_session adapter consumes the same compiler images as the
Python resident control. C++ owns all 11 allocations, three module leases,
private stream, input snapshots, argument binding and timing events.
Ordinary invocation queues ordered work; read/update/close establish completion.
Readback uses native staging so failed asynchronous completion never leaves HIP
writing into a released Python result. Failed completion and partial cleanup
retain native ownership and block further invocation until cleanup succeeds.

47 exact-device/frontend/resident and controlled-HIP failure regressions pass.
Failure injection covers pending completion, copy/event/partial-free failures,
close retry, owning context, fork with an inherited locked mutex and stale
output generation. Quarantined storage is process-lived and never implicitly
freed during static/library teardown. The adapter validates a private program
snapshot before native preparation.
The ABI scan registers all five exports; 21 focused gates pass, one environment
skip. Native source and library hashes are recorded in native-owner-packet.json.

Three exact-device cases compare identical stage images. Independent conversion
statistics, bitwise conversion/storage and folded output are checked before and
after timing. Seven alternating owner trials measure common resident combined,
weight reuse and allocating combined walls. Separate stage events include
submission gaps; none are an isolated GPU instruction timing claim.

| M x N x K | Native/Python warm combined | Weight reuse | Allocating combined |
| --- | ---: | ---: | ---: |
| 128 x 32 x 256 | 1.0949 | 0.7811 | 0.5540 |
| 257 x 80 x 1024 | 1.0137 | 1.0716 | 0.5656 |
| 256 x 64 x 64 | 0.9613 | 1.0130 | 0.5066 |

Allocating walls include preparation, all uploads, three stages, readback and
close. Native image cache leases can be reused across preparations; the Python
control directly loads/unloads modules. Warm results are mixed; no default,
GPU algorithm or broad workload promotion follows.

Still open: ordinary native-session/common-runtime admission, native HIP graph
replay, warm-path attribution/retuning, general dynamic packing and the complete
five-slice program. The native owner is currently explicit via native_session.

Reproduce with the matching gfx1201 environment:
python -m pytest tests/unit/test_rocm_native_nvfp4_owner.py tests/unit/test_rocm_native_nvfp4_lifetime.py
python -m benchmarks.rocm.benchmark_native_nvfp4_owner --output native-owner-packet.json

## Native graph and ordinary JIT follow-through

The native C++ owner now executes ordinary traced NVFP4 JIT and common-runtime portable replay. Fresh-process replay explicitly supplies the matching runtime libraries and disables compiler calls and the legacy Python owner. Exact RX 9070 XT/gfx1201 regression: **58 passed**; native graph focused checks: **13 passed**; shared ABI/audit checks: **32 passed, 1 skipped**.

[Six-case benchmark packet](native-graph-jit-packet.json) records independent numerical checks before/after timing, three windows per case and separate device graph dispatch measurements. Warm allocating JIT wall medians range from 15.34 to 22.03 ms. These walls include host validation, resource ownership, uploads, readback and cleanup; they do not establish kernel speedup. Graph replay caches at most eight stage/repetition keys. Dynamic packing, additional layouts, warm-call tuning and broader AD remain open.

## Matched native ownership and graph attribution

| M/N/K | Native/Python allocating wall | Graph/native warm combined wall |
|---|---:|---:|
| 128 / 32 / 256 | 0.563 | 1.088 |
| 257 / 80 / 1024 | 0.577 | 1.019 |
| 256 / 64 / 64 | 0.510 | 1.096 |

[Matched packet](native-graph-owner-packet.json): seven alternating trials per arm, identical compiler images and inputs; conversion, lossless storage and folded output verified before/after timing. Allocating graph walls include capture/instantiate, while warm graph walls reuse cached graphs. Direct native submission remains the ordinary route; graph capability has no performance promotion. Event samples are per iteration and retain GPU dispatch cost.

## Warm artifact retention

The ordinary gfx1201 JIT retains its compiler-produced RuntimeArtifact with
the verified semantic Graph specialization, bounded to 24 entries. Each launch
still uses common-runtime manifest/Graph validation and native ABI checks.
Inspection artifacts remain separate, and input values are uploaded anew.

[Matched six-case packet](native-artifact-cache-packet.json) alternates seven
retained/reconstructed calls per arm with identical images and an independent
folded numerical check after every call. Retained/reconstructed wall ratios
are 0.727-0.777 (22-27% reduction); these allocating host walls establish no
GPU algorithm gain. The control reconstructs the same manifest before the
same checked runtime launch. Exact-device regressions: 58 passed; the full
10-case frontend device lane additionally proves inspection mutation isolation
and changed activation scales. The earlier profile attributes repeated
manifest construction/validation separately from native preparation.

General dynamic packing/layout/AD integration and warm native preparation/
cleanup overhead remain open.
