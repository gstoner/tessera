# Native continuous scaled-product adjoints (2026-10-09)

Owner: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Synchronization key: FLOATING-SCALED-ADJOINT-20261009.

The textual typed Graph flows through native MLIR reverse recipes, verified
SSA/cotangent/lifetime export, Schedule, sealed Tile/GPU reduction, ROCm Target,
LLVM/HSACO and the checked prepared-program ABI. Python marshals operands and
packages compiler artifacts; it does not calculate backend gradients.

## Envelope and validation

Both matrices and both scales are f32; output is f32, accumulation is fp32,
scales use block layout [4, 4] with fp32 values and exact_per_block policy.
The exercised logical matrix is M=2, N=5, K=9 (ragged N and K blocks).
All four gradient roles, single/reordered selections, both matrix transposes,
shared RHS, shared LHS, independent batches and broadcast prefixes are covered.
Matrix and scale prefixes may broadcast independently. The native recipe sums
missing/singleton batch axes into the original operand shape.

- Native host export/registry/audit gates: 463 passed.
- Owning gfx1201 continuous f32 execution: 80 passed (48 unbatched, 32 batched).
- Composed FP8 reverse regression: 8 passed.
- Public mapped FP8 inverse-cotangent regression: 26 passed.
- Existing public NVIDIA attention-JVP regression: 4 passed on RTX 5070.
  This is sibling regression evidence, not four-f32 NVIDIA execution.
- Ruff passes; mypy ratchet remains zero.
- Exact required CI unit marker lane reproduced in host WSL: 19,619 passed,
  9,559 skipped, 2 failed in 315.18 s. Both failures are the existing generic
  scaled_matmul batching/transpose closure assertions.
- The generic batching/transpose zero-open assertions still fail for
  scaled_matmul (2 failures, 4 passes). Their partial/planned coverage states
  remain unchanged. This slice does not claim generic AD closure.

Every device gradient is compared with an independent float64 block-product
oracle; tests update every input and check old copied outputs remain stable.
Batched oracle reductions iterate logical planes independently of the compiler
indexing recipe. Encoded matrix/scale storage receives no implicit STE.
Dynamic shapes, wider storage/layouts, public Python JIT integration for this
continuous domain, and Apple/x86/SM120 physical consumers remain follow-ups.

## Expanded native validation

A separate expanded native-tool run included hardware/performance/compiler
rows outside the CI unit marker lane: 609 failed, 22,576 passed, 7,973 skipped,
38 errors, 874 deselected. It used a GPU-focused compiler configuration with
x86/EBM/Clifford disabled and lacked ROCm serializer linker layout. This result
is not green and is not reported as the required CI lane.

Restoring those build components and matched LLVM/CUDA tools yielded 628
passes and 514 skips in a representative rerun; 21 failures/4 errors remained.
The remaining selected cases were traced to the LLVM23 llvm/bin/ld.lld
serialization path and absent bias-JVP benchmark JSON inputs. The tests now
use self-contained typed Graph fixtures instead of optional packets, and a
focused rerun passes all 63 checks. The earlier recorder naming failure was
from a pre-staging README; the tracked naming gate now passes. These focused
repairs do not assert that every expanded-run case has been rerun.

## Timing evidence

The packet records the observed gfx1201 architecture, rocminfo inventory,
compiler/provider/source hashes, four exact Graph hashes and five samples per
domain. The provider and compiler match the sources in this branch.
Each case passes numerical comparison before and after timing.

| Case | Native program median (ms) | Checked update/invoke/read median (ms) | Maximum absolute error |
| --- | ---: | ---: | ---: |
| Unbatched | 0.01144 | 0.77628 | 2.27e-8 |
| Unbatched A/B transposed, inverse output seed | 0.01713 | 0.77897 | 2.27e-8 |
| Shared LHS, A transposed | 0.02200 | 0.79615 | 5.85e-8 |
| Broadcast matrices/scales, A/B transposed | 0.02058 | 0.80273 | 4.69e-8 |

Captured member times are grouped pure-SSA device graph windows including
dispatch, excluding capture/instantiation and transfers. They are not ordinary
host launch times. Ordinary native program samples use a 128-repeat event
window and include host dispatch gaps between kernels. Checked host samples
include prepared ABI input update, native execution and copied outputs, and
exclude frontend/compile cost. The packet also records cold package compile
cost. No speedup or broad performance closure is claimed.

## Reproduction

On the owning gfx1201 host with matching compiler binaries and the full native
program/movement provider:

```bash
source .build-owning/validation-env.sh
export TESSERA_OPT="$PWD/.build-output-axes/tools/tessera-opt/tessera-opt"
export TESSERA_ROCM_OPT="$PWD/.build-output-axes/src/compiler/codegen/Tessera_ROCM_Backend/tools/tessera-rocm-opt"
export TESSERA_ROCM_NATIVE_MOVEMENT_LIB="$PWD/.transfer/libtessera_rocm_mapped_output_runtime.so"
export PYTHONPATH="$PWD/python:$PWD"
python benchmarks/rocm/record_floating_scaled_adjoint.py --repetitions 2048 --output gfx1201.json
```

The recorder rejects any architecture other than the live gfx1201.
