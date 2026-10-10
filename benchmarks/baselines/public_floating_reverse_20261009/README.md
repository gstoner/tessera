# Public continuous scaled-product reverse (2026-10-09)

Owner: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Synchronization key: PUBLIC-FLOATING-REVERSE-20261009.
Depends on the native f32 adjoint foundation in PR907.

## Route and scope

Python/textual source -> typed semantic Graph -> native MLIR linear transpose
and checked SSA/cotangent/lifetime export -> Schedule -> Tile/GPU reduction ->
ROCm Target -> LLVM/HSACO -> checked prepared native ABI -> copied gradients.

The public `from_text(..., autodiff="reverse")` owner executes
`native_backward` on gfx1201. f32 A/B and f32 scales use exact_per_block
fp32 accumulation and explicit block [4,4] scale layout. Logical M=2, N=5, K=9
exercises both ragged scale boundaries. The numerical cases cover both matrix
transposes, every single gradient, reordered complete gradients, direct named
shared/independent batches, independently broadcast matrix/scale prefixes,
changed nonuniform seeds, retained outputs, and warm package reuse with all
compiler subprocess calls forbidden.

The existing eager semantic reference accepts matching f32 matrices for
frontend differential certification; native MLIR computes every returned
gradient. It is not a Python arithmetic backend or Python Tile constructor.

The capability model has a separate `graph_only_dtypes` field. These dtype
queries return `artifact_only`, allowing Graph representation without
advertising a ready primal kernel. Existing declared physical dtypes remain
unchanged. gfx1201 now appears explicitly in the dtype-flow report; the
physical manifest still records no ordinary f32 scaled-product primal kernel.

This proves direct public source/Graph reverse, not general f32 vmap wrappers,
dynamic shapes, other storage/layouts, higher derivatives, sibling backend
execution or generic AD closure. Ordinary f32 primal packaging remains open.

## Evidence

- 56 public gfx1201 numerical cases pass (24 unbatched, 32 direct batched).
- 26 public mapped FP8 inverse-cotangent regressions pass.
- Final focused public frontend/capability, dtype-flow, native AD, registry and
  lifecycle gates: 645 passed. Ruff passes; mypy ratchet is zero.
- Required CI unit marker lane: 19,644 passed, 9,559 skipped and two
  failures in the generic scaled_matmul batching/transpose zero-open guards.
  Their states/assertions are unchanged; this dependent draft is not
  merge-ready. The subsequent parent sync changes only documentation and
  the excluded native compiler-route bias-JVP fixture, checked separately.
- Benchmark delivery/citation/generated-doc registry gates: 72 passed.
- Mixed matrix storage is rejected; semantic f32 admission does not widen the
  FP8/packed primal selector.
- Device packet: `gfx1201.json`.
- Recorder: `benchmarks/rocm/record_public_floating_reverse.py`.

Every packet case passes an independent float64 gradient oracle before and
after each timing domain. The observed architecture and rocminfo inventory,
compiler/provider/source hashes, native Graph/package hashes and physical
attestation are included. No speedup claim is made.

| Case | Warm public reverse (ms) | Prepared native program (ms) | Max absolute error |
| --- | ---: | ---: | ---: |
| Unbatched | 1.11631 | 0.01148 | 2.26e-08 |
| Unbatched A/B transposed | 1.12286 | 0.01364 | 2.26e-08 |
| Shared LHS, A transposed | 1.26939 | 0.02131 | 5.85e-08 |
| Broadcast A/B transposed | 1.31945 | 0.02020 | 4.68e-08 |

Public warm timings include concrete frontend work, a newly prepared native
owner, execution and copied gradients; compilation is forbidden. Prepared
native program events include host dispatch gaps between kernels but exclude
input update and output copies. Captured member graph windows are a third
separate device domain, excluding graph creation/instantiation/copies.
Decoration takes about 102–115 ms; first public reverse takes about 176–251 ms
in this packet. These are cold host costs, not kernel times. The remaining
host path should be profiled before choosing an owner/cache optimization.

## Reproduction

On the owning gfx1201 machine with the matching compiler and full native
program/movement provider:

```bash
source .build-owning/validation-env.sh
export TESSERA_OPT="$PWD/.build-output-axes/tools/tessera-opt/tessera-opt"
export TESSERA_ROCM_OPT="$PWD/.build-output-axes/src/compiler/codegen/Tessera_ROCM_Backend/tools/tessera-rocm-opt"
export TESSERA_ROCM_NATIVE_MOVEMENT_LIB="$PWD/.transfer/libtessera_rocm_mapped_output_runtime.so"
export PYTHONPATH="$PWD/python:$PWD"
python benchmarks/rocm/record_public_floating_reverse.py --repetitions 2048 --output gfx1201.json
```
