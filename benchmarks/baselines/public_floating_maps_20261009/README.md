# Public continuous mapped reverse (2026-10-09)

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / LAYOUT-ALG-1.
Synchronization key FLOATING-SCALED-MAPS-20261009. Depends on PR908.

## Compiler route

Public Python/textual reverse JIT -> typed semantic Graph with native batch
intent and output permutation -> verified native linear transpose and
SSA/cotangent/lifetime export -> Schedule -> Tile/GPU reductions and movement
-> ROCm Target -> LLVM/HSACO -> checked native ownership -> copied gradients.

The public vmap dispatcher now preserves the continuous reverse JIT owner.
Admission recognizes matching f32 matrices/scales separately from the existing
FP8/MXFP8 primal and scale-JVP routes. Traced-Graph revalidation uses the same
reverse contract. No f32 primal selector, Python numerical backend, Python
Tile constructor, runtime ABI or physical schedule is added.

Static M=2, N=5, K=9 and block [4,4] exercise ragged K/N bounds.
All 15 nonempty operand mapping masks, A/B logical transposes, one/two map
levels, leading/nonleading result axes and reordered complete gradient
requests are tested. Nonleading input axes and different policies at nested
levels restore gradients to their original matrix/scale layout and reduce
shared operands over the missing map axes. Native inverse output seeds are
private program buffers; previous returned gradients remain unchanged.

Frontend eager plane loops remain independent numerical certification only.
The executed gradients are native MLIR products. Legacy FP8 primal and
scale-JVP selectors retain their previous admission; no coverage state is
promoted and encoded storage still has no implicit straight-through rule.

## Validation

- Initial mapped owner/certificate lane: 241 passed; mixed axes and policy
  conflicts: 13 passed.
- Host f32/FP8 mapped frontend, native adjoint and result-axis integration:
  625 passed. The subsequent reverse tracer guard repair is exercised by
  owning-device execution.
- Operator/dtype/diagnostic/pass/capability drift gates: 478 passed.
- Owning RX 9070 XT/gfx1201: 598 passed, comprising 496 new mapped f32
  numerical/ownership cases and 102 existing public f32/FP8 regressions.
  Changed nonuniform seeds reuse packages with compiler calls forbidden;
  prior outputs remain unchanged.
- The first 16-device mixed-axis run exposed the leftover scale-JVP tracer
  guard; replacing its reverse-only check fixed all 16 before the full sweep.
- Ruff passes; mypy ratchet is zero.
- Final mapped semantic admission/role-validation checks: 257 passed.
- The later scalar role-bound filter changes admission only; the final
  benchmark reruns four public/native cases with its recorded source hashes.
- Final packet identities match the publication source. Delivery/citation/
  generated-doc registry/audit gates: 83 passed; AST Graphify completed.
- Initial exact CI unit marker lane: 3 failed, 19,900 passed, 9,559 skipped.
  The extra recorder census failure came from starting before this README was
  tracked; its final delivery gate passes. The committed snapshot is rerunning.
  Generic batching/transpose zero-open guards remain unchanged and open.

## Benchmark and source identities

Recorder: benchmarks/rocm/record_public_floating_maps.py.
Packet: gfx1201.json.
Four cases independently compare gradients before and after every timing
domain. Live architecture/rocminfo, compiler/provider/source SHA-256, actual
Graph/package identities and native execution attestations are recorded.

Completed warm public native_backward calls include frontend work, native
preparation, execution and copied gradients; compiler subprocess calls are
forbidden. Prepared native program events include host dispatch between
members, excluding input update/output copies. Captured member graph windows
exclude graph creation/instantiation and copies. Setup includes fixture/oracle
generation and is not a compiler-only measurement. No speedup claim is made.


| Case | Warm public reverse (ms) | Native program (ms) | Maximum error |
| --- | ---: | ---: | ---: |
| shared_rhs_leading | 1.78296 | 0.01535 | 3.44e-08 |
| independent_matrix_scale_nonleading | 1.71429 | 0.03180 | 4.41e-08 |
| all_mapped_transposed_nonleading | 1.68842 | 0.01813 | 2.21e-08 |
| mixed_nested_input_result_axes | 2.00045 | 0.02339 | 3.3e-08 |

## Reproduction

On the owning gfx1201 machine with the recorded matching compiler/provider:

```bash
source .build-owning/validation-env.sh
export TESSERA_OPT="$PWD/.build-output-axes/tools/tessera-opt/tessera-opt"
export TESSERA_ROCM_OPT="$PWD/.build-output-axes/src/compiler/codegen/Tessera_ROCM_Backend/tools/tessera-rocm-opt"
export TESSERA_ROCM_NATIVE_MOVEMENT_LIB="$PWD/.transfer/libtessera_rocm_mapped_output_runtime.so"
export PYTHONPATH="$PWD/python:$PWD"
export TESSERA_GFX1201_DEVICE_PROOF=1
python -m pytest -q tests/device/rocm/test_public_floating_scaled_maps.py
python benchmarks/rocm/record_public_floating_maps.py --output gfx1201.json
```

General dynamic shapes, storage/layout widening, ordinary f32 primal/JVP
packages, higher derivatives, generic scaled_matmul batching/transpose
closure and sibling consumers remain open. This is one frontend/AD
integration slice within the larger five-slice objective.
