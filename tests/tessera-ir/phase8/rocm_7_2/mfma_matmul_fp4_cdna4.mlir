// RUN: tessera-opt --tessera-lower-to-rocm --rocm-target=gfx950 %s | FileCheck %s
// REQUIRES: tessera_opt_built
//
// Sprint H-4 — CDNA 4 (gfx950) MFMA FP4 matmul.  CDNA 4 adds FP4 lanes
// at (M=32, N=32, K=32, K_blocks=1).  fp4_e2m1 inputs (packed two-per-byte),
// fp32 accumulator.

module attributes {tessera.target = "rocm_gfx950"} {
  func.func @mfma_matmul_fp4_cdna4(
      %A : memref<32x16xi8, 3>,    // fp4_e2m1 packed 2/byte → 16 bytes per row
      %B : memref<16x32xi8, 3>,
      %C : memref<32x32xf32, 3>) {
    "tessera_rocm.mfma"(%A, %B, %C) {
      mfma_shape = array<i64: 32, 32, 32, 1>,
      elem_dtype = "fp4_e2m1",
      acc_dtype = "fp32",
      hipcc_arch = "gfx950"
    } : (memref<32x16xi8, 3>,
         memref<16x32xi8, 3>,
         memref<32x32xf32, 3>) -> ()
    return
  }
}

// CHECK: tessera_rocm.mfma
// CHECK-SAME: mfma_shape = array<i64: 32, 32, 32, 1>
// CHECK-SAME: elem_dtype = "fp4_e2m1"
// CHECK-SAME: hipcc_arch = "gfx950"
//
// CDNA 4 FP4 MFMA intrinsic:
// CHECK-DAG: llvm.amdgcn.mfma.f32.32x32x32f4f4
