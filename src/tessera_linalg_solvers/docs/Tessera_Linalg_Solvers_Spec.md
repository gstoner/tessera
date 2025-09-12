<!-- MERGE-START: Tessera_Linalg_Solvers_Spec -->
# Tessera Linear Algebra Solvers — Mixed Precision (FP4/FP8/FP16/BF16/FP32) — v0.1

> Scope: dense, batched dense, and block-sparse patterns (preview) with solver APIs and IR for LU/Cholesky/QR/LS/GMRES + iterative refinement.

## Goals
- First-class **solver ops** in Tessera IR with clear semantics.
- **Mixed-precision** compute policies: compute type vs accumulation type vs residual type.
- **Iterative refinement** (IR) and **flexible preconditioning** hooks.
- Autotune-friendly **tile shapes** and **backend-agnostic** lowering to MMA/MFMA.
- **Batched** and **strided-batched** forms as citizens; small/medium matrices are common in inference.

## Precision model
We distinguish three precisions per op:
- `compute_type` (Fp4/Fp8/Fp16/BF16/Fp32)
- `accum_type` (Fp16/BF16/Fp32/Fp64 where supported)
- `residual_type` (Fp32/Fp64)

We support encodings:
- **FP4**: `nvfp4` packed nibbles with per-tile scale (2^k) metadata.
- **FP8**: `e4m3`, `e5m2` with per-channel or per-tile scales.
- **FP16/BF16/FP32**: standard IEEE.

Promotion/demotion ops are explicit in IR: `tessera.quantize`, `tessera.dequantize`, `tessera.rescale`.
Accumulate type is carried as an attribute on matmul/solve ops: `accum_type = #tessera.type<"f32">`.

## Core ops (Solver dialect)
```
tessera.solver.getrf   // LU w/ partial pivoting (in-place or out)
tessera.solver.getrs   // triangular solves using LU factors + pivots
tessera.solver.potrf   // Cholesky (LL^T / U^TU)
tessera.solver.potrs   // solve with Cholesky
tessera.solver.geqrf   // QR (Householder, blocked)
tessera.solver.ormqr   // apply Q
tessera.solver.gels    // least squares via QR
tessera.solver.trsm    // triangular solve (general)
tessera.solver.gmres   // flexible GMRES kernel (blocked Arnoldi) with callbacks
tessera.solver.ir_step // iterative refinement step (A, x_k, b) -> (x_{k+1}, r)
```
All ops accept attributes:
- `policy`: `#tessera.precision<compute=fp8:e4m3, accum=fp32, residual=fp32, scale=per_tile>`
- `tile`: preferred (m,n,k) tile for kernels; can be undefined (autotuner fills).
- `layout`: row-major/col-major/tensorcore-friendly swizzles.
- `batch`: optional batch dims and strides.

## Mixed-precision patterns
- **LU/Cholesky factorization** in FP8 compute + FP32 accum; **residual compute** in FP32.
- **TRSM** in FP8/Fp16 compute + FP32 accum.
- **Iterative refinement**: compute Δx in low precision, update in FP32, stop by ‖r‖/‖b‖ or cap iters.
- **GMRES**: matvec in FP8+FP32acc, orthogonalization in FP32, optional re-orthogonalize in FP32.

## Passes (pipeline sketch)
1. `-tessera-solver-legalize` – materialize solver ops from linalg decompositions.
2. `-tessera-mixed-precision-schedule` – insert (de)quantize, set accum types, choose encode (e4m3/e5m2/nvfp4).
3. `-tessera-blocked-algo` – tile to panel/block algorithms (GETRF, POTRF, GEQRF).
4. `-tessera-solver-to-tile` – rewrite to Tessera Tile IR kernels (GEMM, TRSM, SYRK, HERK).
5. `-tessera-vector-to-mma` – lower to MMA/MFMA/AMX using device tables (Hopper/Blackwell, MI300, AMX).
6. `-tessera-iterative-refinement` – wrap solve with IR loop (max iters, tolerance).
7. `-tessera-solver-verify` – structural + numeric guards (SPD for potrf, pivot tolerances).

## Device mappings (initial)
- **NVIDIA (SM90+)**: WGMMA 16x16x16 (fp8/fp16/bf16→fp32), scale-fuse via TMA if available; TRSM via block inversion + GEMM updates.
- **AMD (MI300)**: MFMA variants; LDS-tiling with async ds_read/write; vector→MFMA step wired.
- **Intel AMX**: BF16/INT8 tiles with FP32 accum (preview path).

## Batched support
Ops accept a leading batch dimension; lowering chooses block-panel batching and persistent-thread kernels. Strided-batch metadata supplied via attributes.

## Testing
- Deterministic seeds; residual-norm FileCheck via `// CHECK: residual_norm <= 1e-3` style comment hooks.
- Compare against FP32 solver as oracle; ensure convergence for IR/GMRES on well-conditioned systems.

## Example (MLIR sketch)
```mlir
// A x = b (SPD), FP8 compute with FP32 accum + iterative refinement
%Af8  = tessera.quantize %A : tensor<?x?xf32> to tensor<?x?x!tessera.fp8.e4m3> {scale = #tessera.scale<per_tile>}
%bf8  = tessera.quantize %b : tensor<?xf32> to tensor<?x!tessera.fp8.e4m3> {scale = #tessera.scale<per_tile>}
%L    = tessera.solver.potrf %Af8 : tensor<?x?x!tessera.fp8.e4m3> {policy = #tessera.precision<compute=fp8:e4m3,accum=f32,residual=f32>}
%x0   = tessera.solver.potrs %L, %bf8 : ...
%x, %r = tessera.solver.ir_step %A, %x0, %b : (tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {max_iters = 5, tol = 1e-3}
return %x : tensor<?xf32>
```

## Roadmap
- v0.2: pivot blocking for LU; flexible preconditioners; sparse CSR/BSR pilot.
- v0.3: dynamic precision switching on divergence; batched QR base and thin-SVD (least squares).
- v1.0: full conformance and perf gates on H100/B200 and MI300A/X; CI with roofline & residual dashboards.

<!-- MERGE-END: Tessera_Linalg_Solvers_Spec -->
