---
last_updated: 2026-10-20
audit_role: reference
---

# Ordinary JIT native tensor program integration

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync: NATIVE-TENSOR-JIT-20261020.
Status: PR #993 delivered as 1b80c6db1ce2d43fbab27b60f4c4dd6df9aa6b31 after required CI and resolved review. The MLP extension below remains an unmerged candidate.

## Compiler and runtime boundary

Ordinary primal SM120 calls with compact FP16/BF16/FP32 roots query the
native compiler. Python describes tensor metadata and binds arguments; native
Graph/Schedule/Tile passes select operations, arithmetic, policies and kernels.
An admitted product retains actual Graph, Schedule, Tile and NVIDIA Target IR.
Construction checks the original source projection at each intermediate stage,
complete Target replay, projected native members and emitted image bytes.

The JIT shares at most 24 immutable compiler products. Each calling thread
keeps at most 24 corresponding C++ owners, its completion receipt and native
inspection artifact. No frame is borrowed from or closed by another thread.
A thread rebinds if its owner belongs to a retired compiler product. Shared
product eviction never closes another thread's live frame.

close_native_storage() releases the calling thread's frames. Worker code must
call it in a finally block before leaving that thread; cleanup from a different
thread cannot substitute for this. Native process/thread/context guards and
failed-close quarantine remain enforced. This provides independent ownership,
not concurrent GPU execution: native dispatch remains serialized by the lock.

Unsupported profiles may continue through existing routes before admission.
Once admitted, compiler and execution errors propagate. Warm calls and
close/rebind reuse a cached verified product without tracing or compiler tools.
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

Matching RTX5070 compiler proof passes all 57 ordinary/artifact/boundary/
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

Paired tiny-program completed-public measurements improve 1.7488-2.4490x
over the merged-main legacy control, across six comparable cases and both
orders. Native cold calls still cost 1.9-3.2 seconds plus separately measured
decoration. Two-sided captured device windows are mixed across the reviewed-source rerun
despite identical IR/images. The initial packet was uniformly slower. Runtime/capture/device attribution remains open; no physical
schedule or pure-kernel promotion is claimed.

Exact-head delivery CI and review remain required before merge. Wider producer/layout/dynamic/AD/encoded envelopes, compiler-free
portable restoration and all five aggregate programs remain open.

Final reviewed-source evidence: benchmarks/baselines/native_tensor_jit_20261020/thread-review/README.md. All175 device cases pass and focused host gates pass445; exact-head CI/review remains required.

## Native multi-matmul extension candidate

Sync: NATIVE-TENSOR-MLP-20261020.

The private extension admits two or more compact typed roots within the same
64-stage/128-buffer capacity. Nonterminal plain matmuls are retained as actual
members, and every matrix checks its own operand pair and numerical policy.
The named three-root MLP explicitly selects FP16 or FP32 intermediate output.
FP32 ReLU preserves eager maximum(0, x) signed-zero and loaded NaN payload
semantics, uses four-byte storage, and feeds the existing ordered FP32 scalar
contraction with the original half RHS. No implicit tensor-core downcast occurs.
Each native scalar member inherits its selected buffer/scalar/geometry contract.

2619 production compiler cases, 25 focused MLP/oracle checks, 470 shared drift
gates and 87 RTX5070 device cases pass. Owning gfx1201 scalar regressions pass
67 cases. Twelve correctness-gated timing rows retain separate native and public
windows. Artifact restoration still requires the matching compiler. Full CPU passes 30785 with 3710 declared skips; exact-head CI/review remain open. Evidence:
benchmarks/baselines/native_tensor_mlp_20261020/README.md.

## Standalone activation integration review follow-up

Sync: NATIVE-TENSOR-MLP-20261020.

Standalone static compact rank-two SM120 ReLU now uses the same native
Graph/Schedule/Tile activation as composed programs. Ordinary host JIT and
serialized packages support FP16/BF16/FP32 with storage-preserving outputs.
Serial and explicit flat_128 policies carry distinct hashes/entries/geometry;
the default physical policy remains serial. FP32 packages additionally have
resident execution proof. All three typed ABIs bind exact buffer storage,
rank, layout, direction and scalar order before launch. Invalid rank guards
and narrower forged storage are rejected before memory access.

FP32 uses tessera.nvidia.relu.x_o_rows_columns.f32.v1; the production CUDA
bridge recognizes its exact hashed serial/flat entry names and four-byte
storage. Existing half ABIs and their zero policies remain unchanged.
Ordinary public host and resident-package proof are separate; this does not
admit arbitrary public resident unary graphs or broaden differentiation.
Numerical fixture: tests/device/nvidia/test_public_standalone_relu.py.
Legacy half producer and fused accumulator epilogue fixtures remain distinct.

The review follow-up passes 50 combined standalone/composed device cases
and 511 applicable host checks (8 declared non-SM120 skips). Final full CPU
passes 30,788 cases with 3,710 declared skips; 2,640 production compiler
cases pass without skips. All 33 generated documents and five surface audits
pass. Updated-commit CI/review remain required.

Standalone native-image packages also pass fresh-process execution without
compiler tools. This is a separate serialized descriptor contract; generic
multi-member native source-program restoration retains its matching-compiler
requirement. Evidence: benchmarks/baselines/native_tensor_mlp_20261020/review/README.md.


Standalone activation dispatch checks the specialized traced Graph with its
complete activation predicate before selecting native descriptor compilation.
Rank-one/rank-three, FP64 and empty-extent inputs remain outside this bounded
route; the operation-name request alone cannot claim their support.
Host-free negative tests in tests/unit/test_sm120_activation_dispatch.py
retain other JIT paths, while positive FP16/BF16/FP32 controls still select
canonical native compilation. Once selected, compilation/launch errors propagate.

## FP32 original root integration candidate — 2026-10-20

Owner W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync NATIVE-TENSOR-FP32-ROOTS-20261020.

Original compact rank-two FP32 roots now pass through native program admission
and host/resident signature transport. Each outlined member still must pass
its own native storage-pair and numerical-policy selection. FP32 matmul inputs
retain the existing ordered FP32 scalar-FMA strategy; no half or TF32
conversion is introduced. The existing frame ABI already represents four-byte
FP32 roots and checks byte capacity, strides, initialized storage and retirement.

The initial eight positive compiler/signature cases fail on the frozen
pre-change source. The matching candidate passes 170 focused cases, 481 shared
registry/lifecycle gates and 18 RTX5070 host/resident cases. Nine timing rows
separate producer/consumer event windows from completed public calls. Full
delivery and publication gates remain active. This candidate does not
establish pitched/dynamic roots, arbitrary operators or general AD closure.

Evidence: benchmarks/baselines/native_tensor_fp32_roots_20261020/README.md.
