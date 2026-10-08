# Native static two-axis scaled batches on gfx1201

Owner: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Sync: ROCM-MULTIDIMENSIONAL-SCALED-BATCH-2026-10-07.
Source remains in the unpublished aggregate.

## Native contract

Typed FP8/MXFP8 static rank-four matrices and scales now preserve two logical
leading dimensions through ordinary Python JIT Graph capture, native Schedule,
Tile, ROCm Target and LLVM/HSACO. Shared-RHS rows flatten batch dimensions and
M natively; independent-RHS and shared-LHS use the product of batch dimensions
as grid-z while retaining per-plane M. Shared storage is not replicated.
Runtime program bindings retain original rank-four shapes and checked lifetimes.
There is one native primal step, no Python batch arithmetic or launch loop.

Both Graph and runtime require equal leading dimension tuples, not merely equal
products. The native member projector also preserves matrix row equality before
flattening. Frontend and native verifier counterexamples reject (2,3) versus
(3,2). An existing scalar test caught a zero-batch admission regression; the
repair requires a nonempty leading prefix for an explicit batching policy.
Generic batching and linear-transpose coverage states remain partial/planned.

## Exact-device numerical and timing evidence

RX 9070 XT / gfx1201 on Tajasaurus, GPU-28d9e7efbf2ef716.
Final matching compiler/runtime/source hashes are in identity-final.json;
timings-final.json retains the live GPU probe and per-member image identities.
102 final device tests pass: 24 new rank-four primal cases, six rank-four
FP32 scale-JVP cases, and 72 existing public primal/mapped-JVP regressions.
Numerics use independent f64 block products, distinct batch scales, KN/NK
storage, changed scales/seeds, and compiler-free warm reuse. The primal/JVP
route carries checked native allocation/lifetime/completion ownership.

Twenty-four correctness-gated timing rows cover logical shapes
(2,3,7,19,256) and (2,2,200,129,1536), three batch policies, FP8/MXFP8 and
KN/NK storage. Every primal row has one native step and a rank-four output.

| Timing domain | Median range across rows |
| --- | ---: |
| Public JIT call wall time | 0.788–2.005 ms |
| Prepared update/invoke/readback wall time | 0.366–1.175 ms |
| Native HIP sequence event window | 0.0103–0.5353 ms |

These domains are separate; event samples include dispatch/enqueue gaps and
are not isolated instruction timing. No comparative speedup or selector
promotion follows. MXFP4, Radiance attribution and NVFP4 model-quality remain
separate required evaluation work.

## Shared and sibling gates

381 final matching-compiler frontend/contract/operator/diagnostic/pass tests
pass; 48 dtype/target/native-program ABI tests pass. Final native fixture lane:
649 pass, 66 unsupported. The RTX 5070 / SM120 native shared-Schedule NVFP4
regression passes nine exact-device cases. This is parity evidence for existing
rank-two/rank-three SM120 routes, not new rank-four NVIDIA execution.
Initial missing-test-path, admission-regression and negative-exception fixture
logs remain separate from final success receipts.

## Reproduce

From the owning gfx1201 scratch checkout:

    source .build-gfx1201-current/validation-env.sh
    export PYTHONPATH=$PWD/python:$PWD TESSERA_ROCM_CHIP=gfx1201 TESSERA_GFX1201_DEVICE_PROOF=1
    .venv-movement-capture/bin/python -m pytest -q tests/device/rocm/test_multidimensional_scaled_batch.py tests/device/rocm/test_public_typed_scaled_vmap.py
    .venv-movement-capture/bin/python benchmarks/rocm/benchmark_multidimensional_scaled_batch.py --output timings-final.json

Remaining: nested public vmap projection, dynamic/nonleading axes, general
linear-transpose/storage AD, sibling physical parity, generic full-unit closure
and curated PR delivery. This executes explicitly authored two-axis batch
intent; it does not claim nested map transformation closure.

Final delivery checks: twelve audit/recorder-inventory tests pass; compiler-plan
ownership/links pass. generic-closure-open.txt retains the current two failures
and four passes: scaled_matmul batching and linear-transpose generic closure
are still open. nvidia-device.txt/nvidia-compiler-source.txt bind the sibling
regression to its actual RTX 5070 and matching compiler.
