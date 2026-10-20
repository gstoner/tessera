---
last_updated: 2026-10-20
---

# Native tensor program pipeline — integration candidate

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync: NATIVE-TENSOR-PROGRAM-PIPELINE-20261019.

## Failure and intended boundary

The general SM120 pipeline previously replaced a composed tensor Graph function
with its matmul kernel, discarding upstream producer arithmetic. Six recorded
FP16/BF16 RTX5070 counterexamples produce -16 for sources whose correct results
are 0 (ReLU and two-sided ReLU/softmax) or -32 (RMSNorm).
The existing explicit native member route supplies correct numerical controls;
that does not establish correctness of the general pipeline.

## Typed program candidate

The native Graph-to-Schedule boundary outlines the original static rank-two
producer SSA into isolated member modules. Existing native Schedule/Tile passes
own their arithmetic and physical schedule. The root becomes a typed memref
program: original arguments retain their order; every stage writes a fresh
allocation; tile.program_invoke carries actual buffer SSA and a nested member
symbol. Private intermediates retire after their final readers. The returned
allocation belongs to the result consumer.

The initial candidate covers storage-preserving ReLU, RMSNorm, layer norm and
last-axis softmax chains on either side of a plain FP16/BF16 matmul. A common
compact row-major physical contract connects the buffers. Explicit conflicting
layouts, unsupported producers and incompatible numerical policies require a
separate implementation and may not be silently normalized.

The invocation verifier checks initialization, output aliasing and retirement,
typed member arguments/results, native pointer/scalar ABI and selected launch
geometry. Native Schedule-to-Tile supplies geometry and scalar arguments from
the actual kernel selection. Original semantic policy attributes remain visible
at the program boundary. Python must not manufacture numerical members.

## Completed named milestones and remaining gates

The named static FP16/BF16 programs have positive standard/explicit compiler
routes, reversed-root and two-sided coverage, typed lifetime/symbol/geometry
checks, native member images, a retained C++ owner and RTX5070 numerical/stream
proof. The Target controller increment below records the current evidence.

The original-source replay and explicit compile API are now covered by
NATIVE_TENSOR_PROGRAM_PACKAGE.md. Before public delivery, finish ordinary JIT/portable
admission, compiler-free warm calls, broader numerical-policy/layout/AD
coverage, and completed public-call/device benchmarks with unchanged
FP8/MXFP8/MXFP4 controls where applicable. Run shared registries, four-backend
assessments and final full validation/CI/review gates before one PR.

This remains a private implementation with no general W1.1 closure or
performance promotion.

## Initial diagnostic proof

The compiler-route cohort passes 27 positive/projection/adversarial cases; the
shared audit/diagnostic/pass cohort brings the focused total to 342. Twelve
RTX5070 generic-driver replays pass the six original counterexamples plus six
changed-input cases. That initial packet proves emitted-member arithmetic and guarded buffers;
the subsequent native-owner and Target packets establish retained execution.
Public package admission remains open. See
benchmarks/baselines/native_tensor_program_pipeline_20261019/README.md.

## NVIDIA Target controller increment

The complete typed host controller now lowers from tile.program_invoke to the
registered tessera_nvidia.program_invoke command. Both boundaries share
NativeTensorProgramContract.h for buffer lifetime, exact SM120 architecture,
symbol, storage, native parameter and selected launch checks. The module
schema must be 1 and references must have exactly one member::kernel level
inside an owned native member module; malformed flat references are rejected
before frame export. Every invocation's arity is checked before cross-operation
lifetime traversal, including operations whose own verifier has not run. All
roots and allocations must use positive compact rank-two f32/f16/bf16 physical
storage within the ABI extent/byte limits. Empty or external hosts are
diagnosed before frame export. The Target form
must reference a lowered LLVM/NVVM kernel; semantic Graph members are invalid.
The device NVVM pass preserves these effectful host commands for the checked
native-frame runtime instead of erasing them as numerical artifact markers.

The tensor-member projection option accepts either verified Tile or Target
programs and re-exports the actual controller frame before isolating a member.
Target members are not sent through Schedule-to-Tile again. Pending AD intent
on the original module/function is rejected before primal projection; the AD
transform must consume it first.

Evidence: benchmarks/baselines/native_tensor_target_controller_20261019/.
The later NATIVE_TENSOR_PROGRAM_PACKAGE.md increment adds source replay and an
explicit compile API; ordinary JIT/portable and general AD closure remain open.

Review evidence: benchmarks/baselines/native_tensor_review_closure_20261020/README.md.

## Native launch admission

The native program uses the same selected matmul and semantic-kernel geometry
for admission and emission. The frame v1 CUDA limits are grid X <= 2^31-1,
grid Y/Z <= 65535, block X/Y <= 1024, block Z <= 64, and total block threads
<= 1024; every extent is positive. Typed invocations validate these limits
even when the call and kernel metadata agree. An unsupported grid is rejected
before packaging or device allocation. This does not flatten or otherwise
change the arithmetic schedule for large matrices.

## Native binary producer contract — 2026-10-20

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync: NATIVE-BINARY-PRODUCERS-20261020.

The SM120 native tensor-program owner admits same-shape, same-storage rank-two
`add` producers on either matmul operand, both operands, and in supported
producer chains. Storage is FP16, BF16 or FP32; all buffers are compact and
row-major. This does not admit broadcasting, pitched views, dynamic extents,
fused matmul epilogues or arbitrary AD reconstruction. The existing public
Graph add catalog and native-frame v1 ABI are unchanged.

Graph-to-Schedule retains both semantic operand positions, including repeated
SSA references. Only isolated native tensor-program member modules collect the
new `schedule.binary_kernel`; unrelated owners retain their existing lowering.
The decision binds the semantic hash, target, storage, FP32 accumulation,
propagating NaNs and a flat 128-thread schedule. Schedule-to-Tile emits
`tile.elementwise_kernel` with `(A, B, output, element_count)` native arguments.
NVIDIA lowering selects only kernels marked by the native elementwise contract.
Both inputs load before the store. Half inputs widen to FP32, add once and round
once to their declared output storage. FP32 inputs retain FP32 arithmetic.

The Tile verifier checks the native function contract, actual argument order,
positive element count, exact compact buffer extent, scalar bindings and
128x1x1 launch block. The grid is ceil(element_count/128). Each lane checks its
index against the exact element count before loading or storing. Existing typed
program initialization, retirement and fresh-output invariants remain required.
FP16/BF16 matmul consumers use typed fragment views; FP32 uses the existing
FP32 scalar-FMA consumer. Conflicting layout or FP64 accumulation requests are
preserved and declined, rather than rewritten to this physical contract.

Final frozen compiler 68358e312c1cd8623ebf6ba3b07aec0d0bd4e4dfb16298888474b8d491788312
passes 60 focused compiler checks, 701 adjacent/registry checks and the owning
RTX5070 99-case compiler/device cohort (60 compiler plus 39 device, no skips).
Three additional diagnostic projected-member cases expose producer bits before
matmul: ties-to-even, signed zero, subnormals, cancellation, overflow, NaN/Inf
classification, immutable inputs and output canaries pass across all storages.
The shortened diagnostic frame is not a public standalone add package.

Nine small timing profiles pass independent numerical checks; their captured
producer regions do not reach the 10 ms duration gate at the native capture
repeat limit. They remain diagnostics with no performance promotion. At M256/K4096/N256, six FP16/BF16 profiles pass all 36 captured
window duration gates, independently checked before and after timing. Eight
neighboring ReLU/RMSNorm/two-sided/reversed-root programs retain byte-identical
source, frame, projected members and 18 PTX images versus merged compiler
01922c4a. The gfx1201 exact MXFP4 and four neighboring quantized image controls pass;
all images are unchanged. The public evidence packet and local delivery gates
are complete; exact-head CI and review remain required before merge. Generic W1.1 and all five aggregate programs remain
open.

Final source-bound binary cohort: 117 RTX5070-host cases pass without skips
(60 compiler, 42 device, 15 recorder). The packet binds their source hashes.
Evidence: benchmarks/baselines/native_tensor_binary_producers_20261020/README.md.

Local binary delivery: 30,842 CPU passes / 3,721 declared skips;
2,754 production compiler cases execute without skips, 117 final source-bound
RTX5070-host cases pass, and the ODS inventory/consumer gate passes 783 cases.
All five surfaces, claim lint, 33 generated docs, zero-error typing, Ruff and
Graphify pass. Publication/CI/review remain the delivery boundary.
