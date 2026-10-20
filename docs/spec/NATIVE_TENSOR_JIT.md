---
last_updated: 2026-10-20
audit_role: reference
---

# Ordinary JIT native tensor program integration

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync: NATIVE-TENSOR-JIT-20261020.
Status: private integration candidate; final gates remain open.

## Compiler and runtime boundary

Ordinary primal SM120 calls with two compact half-storage roots query the
native compiler. Python describes tensor metadata and binds arguments; native
Graph/Schedule/Tile passes select operations, arithmetic, policies and kernels.
An admitted product retains actual Graph, Schedule, Tile and NVIDIA Target IR.
Construction checks the original source projection at each intermediate stage,
complete Target replay, projected native members and emitted image bytes.

The JIT keeps at most 24 signature products and corresponding retained C++
owners. Unsupported profiles may continue through their existing routes before
admission. Once admitted, compiler and execution errors propagate. Warm calls
and close/rebind reuse the verified product without tracing or compiler tools.
Native owned input generations remain pinned through synchronous completion.
Process, thread, context, stream and allocation checks remain owned by the
native frame ABI.

IR inspection and compile reports expose the actual adjacent stages.
JIT execution reports require a successful completed native call. The canonical
execution matrix registers the native program executor; RuntimeArtifact launch
uses the same checked C++ frame after restoring the compiler product.

## Restoration boundary

Artifact restoration requires the trusted configured matching compiler.
Serialized payloads cannot select an executable compiler path. The source,
Schedule, Tile, complete Target, member IR and image bytes are revalidated.
This is not compiler-free portable restoration. Retained preparation and calls
use the previously verified values without tools.

## Capacity and evidence

Native admission, typed invocation verification and frame export enforce the
published 64-stage/128-buffer limits before runtime preparation. Aggregate
256-byte-aligned arena arithmetic uses a wider intermediate and rejects
overflow of the 64-bit frame capacity. Python transport checks the same
geometry and storage envelope. Boundary metadata tests allocate no described
memory; a 64-stage source program also executes numerically on RTX5070.

Matching RTX5070 compiler proof passes all 53 ordinary/artifact/boundary/
resident cases. The separate native-frame owner suite passes 45 cases, the
legacy/adjacent suite passes 118, and eight explicit source-package tests pass.
ABI-specific legacy fixtures explicitly select the predecessor; the new suite
uses unmodified ordinary dispatch. Initial fixture and selector failures remain
in the packet. Source/Target/images for all eight original programs are unchanged.

Focused host gates pass 602 tests; final production compiler proof passes 2594
with zero skips. Resident admission passes 19 cases; Ruff and zero-baseline Mypy
pass. The full CPU run passes 30783 with 3710 declared skips, plus one recorder
naming failure resolved by indexing the recorder and a 319-test focused follow-up gate.
Counts from overlapping runs are not added. Evidence and exact identities:
benchmarks/baselines/native_tensor_jit_20261020/README.md.

Paired tiny-program completed-public measurements improve 1.8853-2.4551x
over the merged-main legacy control, across six comparable cases and both
orders. Native cold calls still cost 1.9-3.2 seconds plus separately measured
decoration. Two-sided captured device windows are slower despite identical
producer images and consumer instruction/directive streams after entry
normalization. Runtime/capture/device attribution remains open; no physical
schedule or pure-kernel promotion is claimed.

Exact-head delivery CI and review remain required before merge. Wider producer/layout/dynamic/AD/encoded envelopes, compiler-free
portable restoration and all five aggregate programs remain open.
