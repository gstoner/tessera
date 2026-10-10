# Continuous native scaled product and JVP — gfx1201

Owner: E2E-REAL-6 / AD-RESIDUAL-EVAL-1 / FRONTEND-IR-MEDIUM-1.
Synchronization: CONTINUOUS-SCALED-PRODUCT-20261009; parent PR911.

[gfx1201.json](gfx1201.json) records 16 correctness-gated timing cases on
Tajasaurus, AMD Radeon RX 9070 XT / gfx1201, UUID
GPU-28d9e7efbf2ef716. The matching full driver uses LLVM/MLIR 23.1.1,
optimized with assertions. Compiler, provider, recorder and source hashes
are retained in the packet.

## Route and proof

The original typed Graph scaled-matmul is the semantic program witness.
Native export expands its isolated continuous f32 member into tensor.generate,
SCF group/dot reductions and strict f32 arithmetic. Each original K group
starts from zero, applies its original scales and joins the outer sum.
Native Schedule/Tile, GPU/ROCm/LLVM lowering, HSACO and checked HIP lifetime
execute the result. Python marshals arguments and verifies an immutable
compiler-owned ABI manifest.

44 native image/ABI tests pass: 40 primal/JVP cases across all four matrix
orientations and the unbatched/shared-RHS/shared-LHS/independent/broadcast
policies, plus four existing FP8/MXFP8 package regressions. 426 native
adjoint/map/auto regressions and 307 focused drift gates pass.
gfx1201 proves 40 independent float64 numerical comparisons with changed
inputs/seeds and retained outputs. The named suffix is M=2, N=5, K=9,
scale blocks [4,4], with a ragged final K and column group; batches use [2,3].
All four continuous operand JVP terms execute from native TangentInterface.
48 existing gfx1201 FP8/MXFP8/NVFP4/MXFP4 device regressions also pass.

## Separate timing domains

Seven rounds alternate original and changed inputs. Every native-event and
warm-call round passes the independent oracle before or after timing.

| Kind | Ordinary native program event median range | Warm prepared update/invoke/read median range |
| --- | --- | --- |
| Primal | 0.00221–0.00448 ms | 0.336–0.589 ms |
| Paired JVP | 0.02019–0.03096 ms | 0.513–0.741 ms |

Ordinary HIP events include host dispatch gaps between native members and
exclude compilation, upload and readback. Warm calls include update and
checked readback, exclude compilation/preparation, and are not public
frontend latency. Compilation is recorded separately. Maximum absolute
error across these timing cases is below 6.8e-8.

These are characterization baselines, not speedups or default promotions.
No gfx1151, SM120, Metal or x86 physical proof is inferred.

## Reproduce

On the owning gfx1201 host, select the delivered matching compiler and
the checked native program runtime library, set PYTHONPATH to the repository,
then run:

    python -m pytest tests/device/rocm/test_floating_scaled_product.py -q
    python benchmarks/rocm/record_floating_scaled_product.py --output packet.json

The device test requires TESSERA_GFX1201_DEVICE_PROOF=1; the recorder queries
the live architecture and rejects a mismatch.

Public ordinary f32 @jit primal/JVP admission, wider dimensions/dynamic
layouts and generic primitive coverage closure remain open. This package
proof does not close those frontend obligations or the overall five-slice goal.
