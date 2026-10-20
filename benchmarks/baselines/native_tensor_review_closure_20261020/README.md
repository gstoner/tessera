# Native tensor program delivery review closure

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync: NATIVE-TENSOR-PROGRAM-PIPELINE-20261019.

## Reproduced compiler defects and corrections

Four malformed-input controls exposed unchecked verifier/export assumptions:

| Control | Before correction | Corrected contract |
| --- | --- | --- |
| Flat kernel reference | Verification passed; projection aborted with exit -6 | Exactly one owned member::kernel reference |
| Later invocation missing input_count | Lifetime traversal segfaulted, exit -11 | Check all invocation arities before cross-operation use |
| Unused scalar host argument | Exporter cast to memref aborted, exit -6 | Check every root's physical type, including unused arguments |
| External host declaration | Exporter accessed an empty region, exit -6 | Defined single-block host with at least one invocation |

The shared Tile/Target verifier also checks module schema 1, f32/f16/bf16
storage, positive compact rank-two shapes and ABI byte/extent capacity for all
roots, operands and allocations. The exporter diagnoses empty/non-program
hosts before reading their blocks or building frame metadata.

The focused compiler cohort passes **207 cases**, including **18 added negative
cases**: flat references with/without projection, member ownership/schema,
four malformed later arities, eight unsupported/invalid root types/shapes and
two empty/external host forms. Wrong-width arity is rejected by the ODS parser;
all controls return ordinary errors rather than assertions/segfaults.

The original/fixed IR, crash logs, diagnostics and exit codes are retained.
control-compilers.json records which predecessor binary reproduced each crash.

## Delivery integration corrections

The first full CPU lane exposed five failures: undated new specs (three
assertions), archived recorder names and the two-operation ODS inventory.
The specs now carry date frontmatter. Archive Python capture files use
underscore filenames named by their packet READMEs, with unchanged code bytes.
The ODS inventory explicitly pins both new operations and 697 declarations.
No allow-list, refusal state or test gate was weakened. All **800 focused CPU
integration cases** pass. The mixed scalar refusal now identifies Graph->Schedule
without changing its numerical policy or admission.

Full-package Ruff passes; the mypy ratchet remains errors 0 / baseline 0.

## Numerical and performance scope

All eight previously executed RTX5070 source programs reproduce byte-identical
original Graph, complete Target and device images with the final corrected
compiler. review-positive-equivalence.json records each case and compiler hash.
The production native owner/provider is unchanged. Exact-device numerics and
separate timings remain those of native_tensor_source_package_20261019.
There is no new hardware performance claim.

Reproduce equivalence by compiling each compressed original Graph in that
packet with compile_native_tensor_program, then comparing source/Target SHA256
and every ordered image SHA256 to source-package-final-timings.json. This
exercises native admission, replay, member projection and PTX packaging.

Full final unit/compiler-route/CI/review gates remain required before merge.
Ordinary JIT, compiler-free portable loading and the full five-slice goal remain
open; this packet does not narrow their completion requirements.

## PR992 launch-geometry review correction

The corrected control reproduces seven failures and two valid grid-Y boundary
cases. Native admission and emission now share kernel selection/geometry, and
the typed verifier checks the CUDA frame's per-axis and total-thread limits
even when invocation and kernel metadata match. Focused compiler/package/
scalar gates pass 216 cases; diagnostic/pass/audit gates pass 315 cases.
All eight original executed source/Target/image programs remain byte-identical.
No new physical schedule, device execution or performance claim is made.

Receipts: review-geometry-receipt.json, review-geometry-red-corrected.txt,
review-geometry-focused.txt, review-geometry-shared.txt and
review-geometry-equivalence.json.

The complete production compiler-route lane passes 2,567 tests with zero skips
(review-geometry-compiler-route.txt). Six raw FP16/BF16 macro, typed-tail and
fused matmul controls retain identical Tile output against the immutable
pre-change compiler (review-geometry-schedule-equivalence.json).
