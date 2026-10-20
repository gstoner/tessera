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

## Gates still required

- Positive standard and explicit Schedule/Tile routes for FP16/BF16, two-sided
  producers and reversed source argument order.
- Adversarial buffer, lifetime, symbol, numerical-policy and layout checks.
- Native member projection, image packaging and retained C++ owner execution.
- Exact RTX5070 numerical comparisons, stream ordering and guarded allocations.
- Separate completed public-call and device timing with unchanged primitive
  FP8/MXFP8/MXFP4 controls where applicable.
- Shared registry, diagnostic/pass, four-backend, generated-document and full
  validation gates before one reviewed PR.

This document records an unfinished private implementation. No executable
whole-program route, general W1.1 closure or performance promotion is claimed.

## Current diagnostic proof

The compiler-route cohort passes 27 positive/projection/adversarial cases; the
shared audit/diagnostic/pass cohort brings the focused total to 342. Twelve
RTX5070 generic-driver replays pass the six original counterexamples plus six
changed-input cases. These are emitted-member arithmetic and guarded-buffer
proof, not a retained C++ runtime or public package admission. See
benchmarks/baselines/native_tensor_program_pipeline_20261019/README.md.
