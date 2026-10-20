---
last_updated: 2026-10-20
---

# Native bounded tensor program candidate

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync: NATIVE-BOUNDED-BINARY-PRODUCERS-20261020.

## Source and typed ownership

The first bounded profile is compact rank-two matching FP16/BF16 A and
residual, a same-storage `add`, and a plain row-major matmul with FP32 output.
M varies within an explicit positive capacity; K and N remain fixed. Both
original add inputs must have equal active M. No broadcast, pitched layout,
fused epilogue, dynamic K/N, general producer DAG or AD expansion is admitted
by this increment. Existing static FP32/unary/MLP routes remain separate.

A textual Graph may already use `tensor<?xKxhalf>` with explicit matmul shape
bounds. The ordinary Python frontend uses the existing `shape_bounds={"M": B}`
policy: native C++ projects the traced original source onto dynamic member
types after checking that the trace fits B. Python transmits this declaration;
it does not rebuild Graph arithmetic, choose numerical kernels or calculate
operation-specific launch geometry. Conflicting existing bounds and encoded
layouts cannot be silently replaced. Original source and policy remain bound
by the source digest and complete Target replay.

The typed host controller takes actual dynamic input memrefs. It derives M
from the declared original root using memref.dim. Private and result storage
use static capacity allocations, with explicit dynamic memref.subview values
passed to native invocations and returned to the caller. A smaller resident
input is never typed as a full-capacity allocation. Each member separately
keeps its original active memref types. Row flags, view sizes, zero offsets,
unit strides and base-allocation aliases are checked against capacity, input
initialization, unique writers and private retirement. Standard metadata-op
invariants are verified before walking later operations, so malformed views
must diagnose rather than crash. Schedule binary subjects/results retain their dynamic Graph type;
`row_bound` is a separately verified capacity in the content-addressed decision.
The matmul keeps its existing typed dynamic leading-dimension ABI, Tile views,
fragment packs, accumulator lineage and NVIDIA lowering.

## Frame v2 and native ABI

`tessera.native.tensor_program_frame.v2` extends v1 with `row_bound`,
`row_root` and `row_buffers`. Buffer shapes/bytes describe immutable capacity.
Flagged active buffer rows equal the original root's current M. Each stage
carries capacity scalars/geometry plus `scalar_row_multipliers`,
`grid_row_multipliers` and `grid_row_divisors` selected by the compiler.

A scalar multiplier of zero retains its constant; otherwise its value is
M times the multiplier. A grid multiplier of zero retains the constant with
divisor one; otherwise its value is ceil(M times multiplier divided by divisor).
The compiler and native metadata planner check positive divisors, valid root
bindings, signed-i64 overflow, capacity equality and CUDA launch limits.
Native execution interprets these expressions without numerical operation tags.

The additive preparation ABI is `tessera_nvidia_native_frame_prepare_v2`, using
a separate `TesseraNvidiaNativeFrameRowPlanV2`. Existing v1 step/host-view layouts
are unchanged. Both prepared owner types use the existing checked synchronous
invoke, resident, profile, stats and close handle API. V1 preparation cannot
implicitly acquire bounded metadata. This version is unrelated to attention's
integer schema 5 for saved-LSE plus bounded sequences.

## Active binding and lifetime

Preparation allocates one capacity arena. For each call C++ constructs candidate
active descriptors, scalars and grids without changing the retained frame.
Actual input/output spans, compact strides, storage, M equality, capacity,
non-overlap, context, borrowed allocation identities and producer ordering must
pass before snapshots or active launch metadata are committed. Private buffers
remain native-owned. Resident profiling uses the last successful active byte
spans and retained producer streams, not the maximum capacity spans.

Python performs generic view marshaling and allocates an appropriately shaped
host output. The reusable JIT cache key erases only axes declared variable by
the compiler frame; other dimensions, storage, strides and host/resident scope
remain part of the key. Warm matching calls do not parse IR, trace, compile or
execute eager arithmetic. Portable artifact restoration still requires the
matching trusted compiler for source replay, as on the existing static route.

## Initial capacity-model evidence boundary

The initial four acceptance failures locate the static-only ownership gap.
The candidate passes 95 focused public-policy/source/package/static-regression
cases and 43 host-only transport cases. Frozen compiler identity is
f1ce99e7b0cc6cd94159f566ebecc40ad0cef6ce5226cb0119e84ef072269160.

Eight fresh-provider RTX5070 cases pass: ordinary public shrink/grow, independent
stored-add/matmul comparison, serialized replay at a different M, direct native
refusal without changing generation/output, and pending resident producer
streams. All 120 static owner/MLP/FP32 controls pass. The gfx1201 exact composed
MXFP4 control passes and all five neighboring quantized image controls remain
byte-identical. No NVIDIA proof transfers to sibling architectures.

The native provider is 06fcdb02b05e52f681d35427afc76a8b3a6c772d4fa88a281b69fb70006c9465.
It adds exactly one public export (prepare_v2); the original 102 exports remain.
The missing-include build failure is retained separately from the successful
fresh build. Separate producer/consumer/program events and completed-public
benchmarks and broad gates from that initial model are retained in the packet;
the current active-memory proof is below. The five
aggregate programs remain open.

Evidence: benchmarks/baselines/native_tensor_bounded_binary_20261020/README.md.

## Active memory SSA refinement

The first numerically validated capacity model is retained as provenance.
Two new typing assertions fail against its capacity-typed root signatures.
The refined 73739736aa3449078c9c9d0c5a508ca365bf71b0bfd22605556b12f83ef8c4cc
compiler passes 176 focused controller cases, including explicit dynamic
roots/subviews, alias retirement, malformed later views and static-schema
separation. The complete selected registry/public/transport cohort passes 487.
The owning RTX5070 cohort passes 32 cases: 24 compiler cases and eight device
cases. These are not 32 independent numerical workloads.

Both storage profiles retain byte-identical original source, native frames,
member IR and PTX. Only host Schedule/Tile/Target memory SSA changes. Provider
06fcdb02 remains unchanged. Current full compiler: 2,779 passed, zero skipped.
First active-SSA CPU run: 30,902 passed, 3,721 declared skips; three audit-routing failures
were corrected with all 19 routing/lifecycle checks passing. Twelve fresh
RTX5070 timing rows retain one allocation per storage across M=17/1/256/65/5/200.
24/72 captured windows meet the 10ms duration gate; short producer windows
remain diagnostic. Separate producer, consumer, native-program and completed
public timing domains are retained without a speedup claim. Current gfx1201
numerics/five image controls pass. Final delivery gates remain.
The initial candidate's full proof is retained separately.

## Final local delivery

Final local delivery: the CI-equivalent CPU lane passes **30,894** with
**3,721 declared skips**, zero failures/errors. The 11 cases omitted versus
the first CPU run all have explicit performance markers, matching CI's selector;
no other test IDs differ. The initial log-field failure and correction are
retained. All **2,779** compiler cases execute with zero skips. The 323 final
documentation/registry cases, 790 ODS cases, all five surfaces, 33 generated
documents, Ruff, zero-error mypy and Graphify pass. Exact-head public CI/review
and delivery remain pending; all five aggregate programs remain open.
