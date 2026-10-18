---
last_updated: 2026-10-18
audit_role: reference
---

# Mixed-storage scalar contraction — implementation contract

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Synchronization key: W11-MIXED-ADJOINT-CONTRACTION-20261018.

## Required semantics

The source-active reverse Graph multiplies an FP32 output cotangent by
FP16/BF16 primal operands. Converting the cotangent to half to obtain a
hardware MMA operand pair changes the Graph contract.

tile.scalar_matmul_kernel carries A/B/D pointers followed by logical
M/N/K i64 dimensions. Mandatory type attributes name each operand, output,
and FP32 accumulator. Mandatory boolean transpose flags describe compact
row-major physical allocations. Transposed A indexes k*M+m; transposed B
indexes n*K+k. There is no host transpose or implicit input narrowing.

Each output starts at positive FP32 zero and visits contraction indices
in increasing order with FP32 fused multiply-add. Half operands widen
exactly to FP32; an FP32 operand stays FP32. Only the final store may round
to FP16/BF16. The operation declares memory reads and writes.

The initial NVIDIA materializer assigns a 16x8 output tile to exactly
32 threads, four outputs per lane, with bounds checked before input loads
or output stores. It emits native arith/SCF/LLVM/NVVM IR, without Python
numerical code or runtime kernel source generation. This scalar strategy
claims no hardware MMA or TF32 semantics.

## Remaining integration and evidence

The named requested-source program below now executes through public VJP and
portable replay. Generic reverse, other requested gradients, batching/dynamic
shapes and final publication gates remain open. The recorded benchmark keeps
native event intervals separate from completed public calls. Exact-device SM120
proof does not transfer to AMD, Apple or x86.

The aggregate five-slice goal remains open.

## Checked native transport and current proof

The additive scalar prepare/resident entry points use the existing HostView
dtype codes independently for A/B/output. Native preparation seals compact
physical shapes and the six-parameter pointer/dimension ABI. Host invocation
uses retained native staging; resident invocation checks live allocation
context/capacity, disjoint output, explicit producer events and synchronous
completion. Native default-stream sentinels retain their existing guards.
Host binding rejects non-native-endian/noncompact arrays and read-only output.

Portable descriptors bind each role's dtype/alignment/shape and the exact
image. Ordinary mixed FP32/FP16 or FP32/BF16 JIT has a native prepared owner;
warm calls neither retrace nor compile. Public FP32/FP16 output selection
retains its existing API. BF16 output is proved on typed Graph packages.
Uniform-FP32 default strategy selection is unchanged.

The earlier standalone source freeze passed 396 tests with zero skips, including native
compiler boundaries, raw numerical fixtures, checked package replay, public
calls, native event-profile output storage and legacy prepared-owner controls.
The isolated no-wait provider fails both pending-root controls before any
download or free can rescue their ordering. Source/tool/provider hashes and
raw receipts remain in the owned scratch until the benchmark/publication
packet is complete. Earlier separate shared and legacy lanes retain their
own source scopes; their passes are not added to the combined total.

This earlier lane proves the standalone contraction prerequisite. The later
sections describe composed reverse integration and its separate source scope.
Broader delivery gates remain active.

Four additional owning tests connect an independently checked native
RMSNorm/ReLU/matmul primal/JVP generation to the compiled scalar consumer.
They retain the FP32 result after producer-owner closure, use independently
chosen consumer half storage, forbid downloads or compilation during
consumer replay, and compare both local and propagated numerical budgets.
This is a separate edge between compiled programs; the composed reverse proof
is recorded below.

## Reverse row integration


The additive SM120 row schema 6 now carries two readonly half-storage inputs
and one half-storage result through native Graph/Schedule/Tile/Target lowering.
Schema 1 retains its original two-result JVP ABI; mismatched schema/arity is
rejected. Eight compiler tests and two RTX5070 numerical/store-canary tests pass,
including the actual row SSA extracted from paired reverse differentiation.
The later named source-reverse exporter and owner consume this row ABI.
Generic reverse envelopes remain open.
This changes no sibling target's physical admission or exact-device status.

## Native source reverse program

The opt-in unit marker `tessera.native.sm120_reverse_request` on a fresh
static SM120 reverse Graph requests native paired-AD outlining. The named
profile is RMSNorm → ReLU → plain FP32-output matmul with one requested source
gradient, FP16/BF16 primal storage and an FP32 output cotangent. Argument
permutation is explicit in the plan. Existing forward numerical/orientation
policies must match; conflicting policies are refused before member execution.

The exporter parses and verifies the exact input before paired AD and binds the
returned plan to that canonical source. Reusing an already outlined module,
including an edited outline carrying stale program metadata, is rejected.
Compiler identity is checked across export and member lowering. Portable replay
uses its separately validated frozen witnesses and does not rerun this compiler
export boundary.

The compiler selects actual reverse SSA, cutting at the mixed matmul result.
It emits two member Graphs and five allocation records: three readonly roots,
the half-storage matmul adjoint edge and the returned half source gradient.
The edge's first write is step 0 and final read step 1; the result is produced
at step 1 and escapes the completed program. Each member crosses native
Schedule, Tile, NVIDIA Target and NVVM lowering independently. Outline metadata
continues to say `artifact_only`; it does not assert execution.

The additive `tessera_nvidia_source_reverse_*_v1` C ABI consumes the compiled
images and checked typed frame. Its native owner loads both kernels, retains
scratch and a private stream, orders resident producer streams, submits both
members, and synchronizes before committing the independent host result.
The Python binding projects metadata and allocates that result; it performs no
derivative math or individual member launches. Profiling refreshes roots and
keeps the two stage intervals, composed CUDA interval and completed host wall
time separate.

The public `native_backward` dispatcher selects a declared composite family,
rather than claiming this multi-op Graph as a single normalization operation.
Host inputs receive the existing numerical frontend differential certificate;
resident inputs receive the structural metadata certificate without downloading
or numerically reexecuting their source. A bounded JIT owner cache reuses the
same native images on warm calls, and closes with the JIT storage lifetime.

The portable schema `tessera.nvidia.source_reverse_program.v1` retains the
source and paired Graphs, native allocation/SSA plan, both member Graph/Schedule/
Tile/Target/native witnesses, per-member ABIs and image identities. Restore
validates metadata before provider access and never invokes a compiler.
Hashes establish internal consistency, not authenticity of arbitrary supplied
native code. Runtime launch produces the physical execution attestation after
both native members complete. The named requested-source profile remains
distinct from generic reverse, other requested gradients, batching and dynamic
shape support.

## Warm metadata validation and measured scope

Only module-level attributes change during public reverse projection; the
verified function SSA is shared without mutation. The runtime caches at most
eight validated portable snapshots, keyed by a hash of the complete current
payload rather than its displayed digest. A miss validates an owned JSON
snapshot; restored owners and receipts consume that snapshot. Returned receipt
ABIs are copied so caller mutation cannot poison a cache entry. Native frame,
context, allocation, stream and completion checks still execute on every call.

Evidence and reproducible standalone recorder:
benchmarks/baselines/sm120_source_reverse_20261018/README.md and
benchmarks/nvidia/benchmark_source_reverse.py. The counterbalanced control
retains repeated validation and Graph copying while borrowing the identical
native owner, images, roots and execution certificate. This measures completed
public metadata overhead; it establishes no kernel speedup.
