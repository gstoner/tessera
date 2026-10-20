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

## Evidence and remaining gates

Initial owning RTX5070 overlay proof passes twelve tests: eight source/storage
JIT cases and four RuntimeArtifact executions. Changed input generations,
independent outputs, warm tracing/tool prohibition, close/rebind and owned
stream refusal are covered. This uses the preserved source-package compiler;
matching final compiler and broader exact-device regressions remain required.

Focused package/registry/audit checks pass 343 cases. Updated transport,
restoration, executor and inspection tests pass 52 cases; overlapping tests
must not be added together. Owned-input scope and tool-free cache/rebind have
a host transport regression.

Before publication: align native admission with all frame capacity limits;
run matching-toolchain resident/stream/lifetime and adjacent route tests;
record paired public/device timings; finish documentation, type/lint, full
unit and production compiler-route gates. Wider producer, layout, dynamic,
AD, encoded storage and compiler-free portable work remain open. No aggregate
five-slice closure or physical schedule/performance promotion follows.
