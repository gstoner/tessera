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
