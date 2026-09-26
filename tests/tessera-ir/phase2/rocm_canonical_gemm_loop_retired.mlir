// ROCm GEMM has no canonical M/N/K scf.for entry. That entry -- a
// tessera.matmul / tile.mma carrying tessera.canonical_k_step, produced by
// tessera-tiling (and carried to tile.mma by tessera-tile-ir-lowering) --
// was Lane B's Graph->Tile route, which skipped Schedule IR. Its matcher, its
// one-wave LDS comparison body and the `canonical_mnk_scf_for` source stamp in
// generate-wmma-gemm-kernel were deleted 2026-09-26
// (docs/audit/backend/rocm/ROCM_LANE_MAP.md, "Lane B is retired"); ROCm GEMM
// enters at Tile level as tile.matmul_kernel from the scheduled route.
//
// Negative fixture (Decision #21): a marked step that still reaches the
// generator is refused by name, not passed through unlowered and not
// answered with some other kernel. Both marker carriers are covered: the
// Graph-level tessera.matmul straight out of tessera-tiling, and the tile.mma
// tessera-tile-ir-lowering rewrites it into. Converted from
// canonical_lds_arch_refused.mlir, whose subject (the deleted LDS comparison
// body's gfx12 refusal) went with the body.
//
// The tessera.canonical_k_step marker itself is NOT retired: Apple's
// tessera-apple-canonical-gemm consumes it (phase8/apple_canonical_gemm.mlir).

// REQUIRES: tessera-rocm-backend
// RUN: not tessera-opt %s --tessera-tiling --generate-wmma-gemm-kernel 2>&1 | FileCheck %s
// RUN: not tessera-opt %s --tessera-tiling --tessera-tile-ir-lowering --rocm-wave-lds-pipeline --rocm-wave-lds-legality --generate-wmma-gemm-kernel 2>&1 | FileCheck %s

module attributes {tessera.arch = "gfx1151"} {
  func.func @canonical_gemm(%a: tensor<31x23xf16>, %b: tensor<23x47xf16>) -> tensor<31x47xf32> {
    %0 = "tessera.matmul"(%a, %b) : (tensor<31x23xf16>, tensor<23x47xf16>) -> tensor<31x47xf32>
    return %0 : tensor<31x47xf32>
  }
}

// CHECK: error: ROCM_CANONICAL_GEMM_LOOP_RETIRED: ROCm has no canonical M/N/K scf.for GEMM entry
// CHECK-NOT: gpu.func
