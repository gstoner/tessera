# Nested public typed maps on gfx1201

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Sync ROCM-NESTED-TYPED-VMAP-2026-10-07.
Implementation remains in the unpublished aggregate.

## Frontend and native execution

Two public leading vmap transformations now preserve explicit map depth,
scalar semantic capture, independent owner caches and original source bounds.
The tracer removes both mapped dimensions for the scalar semantic capture;
typed frontend projection restores the full logical prefix and batch intent.
Native Graph/Schedule/Tile/ROCm Target/LLVM lowering owns geometry and arithmetic.
Public compiled execution performs no Python numerical map loop, input replication
or per-plane launch loop. The eager map loop is reference-only certification.

Three matching inner/outer policies execute: shared RHS, independent matrices,
and shared LHS. Matrices/scales retain both leading dimensions, with original
rank-two shared storage. Output shape is [B0,B1,M,N]. Each primal package has one
native step. FP32 scale-JVP preserves native differentiation request and program
ownership; E8M0 storage derivatives remain gated. Mixed sharing, nonleading axes,
different output-axis placement and deeper maps remain explicit open work.

Original scalar and inner-map caches/Graphs remain unchanged. Both leading
symbols are distinct; scalar M/N/K/G/C retain their original axes. Counterexamples
reject permuted equal-product prefixes before capture and preserve scalar
Range(M,1,6) rejection before native JVP trace/compile. Reference certificates
compare an independent scalar-map oracle with the projected Graph result,
preserving explicit gated byte annotations.

## Exact-device proof and measurements

Tajasaurus RX 9070 XT / gfx1201, GPU-28d9e7efbf2ef716.
102 owning-device cases pass: 24 nested primal, six nested FP32 scale-JVP,
and 72 existing mapped primal/JVP regressions. Changed inputs/seeds execute
warm with compiler subprocesses forbidden. Independent f64 block oracles and
scalar/inner owner isolation are checked. Compiler/runtime and actual source
hashes are in identity.json. timings.json retains GPU probe, image/Tile hashes,
native member scalars/geometry and original logical output shape.

Twenty-four primal timing rows cover (2,3,7,19,256) and (2,2,200,129,1536),
three matching policies, FP8/MXFP8, and KN/NK storage.

| Separate timing domain | Median range across rows |
| --- | ---: |
| Public nested JIT call wall | 0.796–1.851 ms |
| Prepared update/invoke/readback wall | 0.354–1.215 ms |
| Native HIP sequence event window | 0.0103–0.5352 ms |

No comparative speedup, isolated-instruction claim or selector promotion.
Events include dispatch/enqueue gaps. These primal measurements do not claim
nested AD timing, MXFP4/Radiance closure or model-quality improvement.

## Shared and sibling validation

415 shared tests pass with 16 gated skips, including native JVP, dtype/op/
diagnostic/pass gates. 11 existing public SM120 vmap cases pass on the RTX 5070
with 46 unrelated cases deselected; no nested NVIDIA capability is promoted.
The initial negative-exception fixture and missing test-path logs are retained
separately from success receipts. No C++ pass, native image generator, operation,
dtype or ABI schema is introduced by this frontend projection change.

From Tajasaurus scratch:

    source .build-gfx1201-current/validation-env.sh
    export PYTHONPATH=$PWD/python:$PWD TESSERA_ROCM_CHIP=gfx1201 TESSERA_GFX1201_DEVICE_PROOF=1
    .venv-movement-capture/bin/python -m pytest -q tests/device/rocm/test_nested_typed_scaled_vmap.py tests/device/rocm/test_public_typed_scaled_vmap.py
    .venv-movement-capture/bin/python benchmarks/rocm/benchmark_nested_typed_scaled_vmap.py --output timings.json

Remaining: mixed inner/outer broadcast policies, dynamic/nonleading/deeper
maps, linear-transpose/storage AD, sibling physical routes, generic full-unit
closure and curated PR delivery. Generic batching/transpose states stay open.

Final delivery: twelve audit/recorder-inventory checks and compiler-plan
ownership/links pass. All 32 derived views are regenerated. The strengthened
production-boundary-tests.txt lane passes 30 cases with both compiler
subprocesses and CPU arithmetic oracle forbidden during warm execution.
The prior graph refresh completed; a final AST refresh follows the strengthened
device-test source edit.
