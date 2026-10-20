---
last_updated: 2026-10-20
---

# Native SM120 tensor frame owner — integration candidate

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync: NATIVE-TENSOR-PROGRAM-PIPELINE-20261019.

## Authority and scope

The owner consumes the compiler-projected typed buffer/launch frame and emitted
PTX members. It selects no operation family, numerical formula, layout schedule
or fragment implementation. Every launch uses the frame's buffer IDs, positive
i64 scalar arguments and explicit grid/block dimensions.

The current compiler envelope is static rank-two compact FP16/BF16 producer
DAGs ending in a plain matmul. This runtime milestone does not establish
public JIT integration, portable admission, arbitrary producers/layouts,
quantized program support, general AD or performance promotion.

## C ABI

tessera_nvidia_native_frame.h declares the version-one step and six entry
points: prepare, invoke, resident, profile, stats and close. Errors use the
existing same-provider tessera_nvidia_matmul_last_error.

Buffer descriptors use the existing HostView storage IDs (f32=1, f16=2,
bf16=3), rank two, exact physical bytes and compact byte strides. Role bytes
are input=0, private=1 and returned=2. The native owner verifies extents before
byte multiplication, unique writers, initialized reads, consumed private
buffers, launch bounds and the actual driver's pointer/i64 parameter ABI.
Malformed frames produce no public handle.

Each owner retains modules, a nonblocking stream and one dedicated device arena
plus pinned host staging. It reuses the existing allocation and event-timing
helpers. Identical member images within one frame share a loaded module;
this is separate from packet-level content-addressed image storage.
Different owners have independent storage and completed-frame generations.

Host invocation snapshots all roots, enqueues the native program, stages every
result and commits caller outputs only after successful completion. Host result
aliases with roots or other outputs are rejected. Private allocations remain
owned until close; logical temporaries can be reused only after the completed
frame has retired its readers.

## Resident reads and lifetime

Resident-to-host execution requires an explicit producer stream per root,
including the CUDA legacy/per-thread sentinels 1/2. The native owner queries live
allocation context/capacity, rejects spans overlapping its private arena and
orders different producers with recorded events and stream waits. The result
is independent completed host storage; no private device pointer is exported.

Handles bind the process, context identity and a per-thread generation token,
so recycled OS thread IDs cannot adopt an old handle. Profiling borrowed roots
also checks live allocation IDs to reject retired/replaced storage. Changed
producer streams require a new invocation. Python bindings retain borrowed
providers while a frame may be profiled; failed close quarantines them until
context teardown permits explicit retirement.

CUDA execution failures drain before storage can be reused. Failed completion
poisons the owner and requires context teardown; a failed close retains native
ownership. Static argument rejection preserves the previous completed frame.

## Timing contract

Completed host-call measurements include argument marshaling, host result
allocation, snapshot, transfers, all kernels and completion. A paired prebound
C-ABI measurement executes the identical native work with prebuilt views and
output; the difference includes adapter validation and output allocation.

CUDA-event profiles independently measure a full program and grouped stages.
Captured profiles exclude graph setup, priming and retirement, transfers and
readback. Stage windows are independent and must not be summed as program time.
Cold owner preparation includes ABI binding/module JIT/allocation on an existing
context; it is not frontend/compiler cost.

## Next integration gates

The named whole-program Target controller is now verified in
NATIVE_TENSOR_PROGRAM_PIPELINE.md and the native_tensor_target_controller_20261019
packet. NATIVE_TENSOR_PROGRAM_PACKAGE.md now records source-to-program replay and
the explicit compile API. Ordinary JIT dispatch and portable admission remain
required before full public integration. Preserve the native
Graph/Schedule/Tile/Target ancestry; do not replace the remaining integration
with Python operation dispatch or reconstructed kernels. Then prove
compiler-free warm public replay, portable reconstruction, broader source and
layout envelopes, and separate exact-device benchmarking before delivery.


## Positive i64 extent follow-through — 2026-10-20

Sync: NATIVE-BINARY-PRODUCERS-20261020.
The existing v1 step wire fields and LLVM parameters are signed i64. Python
transport and native metadata planning now preserve positive values through
INT64_MAX instead of imposing an unrelated INT32_MAX scalar cap. Buffer shape
axes, byte products/capacities, selected kernel semantics and CUDA grid/block
limits retain their independent checks. This does not widen an operation's
semantic envelope or assert that large buffers fit a particular GPU.

The same pure C++ metadata planner feeds production prepare and a test-only
observer translation unit. The shipped provider still exports the same 102
public entrypoints. The observer lives under tests/device/nvidia and is absent
from the shipped ABI inventory; no generator exemption was added. Production
prepare still creates its module/stream and allocates its arenas eagerly.

Compiler-projected add members with element count 2^31 preserve the pointer/i64
PTX signature and pass the shared native metadata probe for FP16/BF16/FP32.
These probes allocate no CUDA arena and make no large numerical-execution
claim. Normal-sized ordinary/resident/rounding tests separately prove execution.
A test that initially assumed prepare was allocation-free was corrected; its
assertion failures are not evidence of a numerical failure or lazy allocation.

The timing recorder requires K <2^24 before any hardware probe because its
FP32 gamma-K oracle requires K*u <1. The oracle checks that condition directly.
This is a measurement-envelope constraint, not a backend matmul limit.
