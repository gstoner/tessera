// RUN: tessera-opt --tessera-lower-to-rocm --rocm-target=gfx942 %s | FileCheck %s
// REQUIRES: tessera_opt_built
//
// Sprint H-4 — CDNA 3 MFMA FP8 matmul.  K is doubled vs bf16:
// (M=32, N=32, K=16, K_blocks=1).  E4M3 inputs, fp32 accumulator.

module attributes {tessera.target = "rocm_gfx942"} {
  func.func @mfma_matmul_fp8(
      %A : memref<32x16xi8, 3>,   // fp8_e4m3 packed
      %B : memref<16x32xi8, 3>,
      %C : memref<32x32xf32, 3>) {
    "tessera_rocm.mfma"(%A, %B, %C) {
      mfma_shape = array<i64: 32, 32, 16, 1>,
      elem_dtype = "fp8_e4m3",
      acc_dtype = "fp32",
      hipcc_arch = "gfx942"
    } : (memref<32x16xi8, 3>,
         memref<16x32xi8, 3>,
         memref<32x32xf32, 3>) -> ()
    return
  }
}

// CHECK: tessera_rocm.mfma
// CHECK-SAME: mfma_shape = array<i64: 32, 32, 16, 1>
// CHECK-SAME: elem_dtype = "fp8_e4m3"
//
// CDNA 3 FP8 MFMA intrinsic:
// CHECK-DAG: llvm.amdgcn.mfma.f32.32x32x16f8f8
