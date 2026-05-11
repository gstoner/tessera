// RUN: tessera-opt --tessera-lower-to-rocm --rocm-target=gfx1100 %s | FileCheck %s
// REQUIRES: tessera_opt_built
//
// Sprint H-4 — RDNA 3 (gfx1100) WMMA matmul.  RDNA has no MFMA; the
// equivalent intrinsic is WMMA at (16, 16, 16).  Wavefront width is
// 32 lanes (vs CDNA's 64).

module attributes {tessera.target = "rocm_gfx1100"} {
  func.func @wmma_matmul_rdna3(
      %A : memref<16x16xbf16, 3>,
      %B : memref<16x16xbf16, 3>,
      %C : memref<16x16xf32, 3>) {
    "tessera_rocm.wmma"(%A, %B, %C) {
      wmma_shape = array<i64: 16, 16, 16>,
      acc_dtype = "fp32",
      hipcc_arch = "gfx1100"
    } : (memref<16x16xbf16, 3>,
         memref<16x16xbf16, 3>,
         memref<16x16xf32, 3>) -> ()
    return
  }
}

// CHECK: tessera_rocm.wmma
// CHECK-SAME: wmma_shape = array<i64: 16, 16, 16>
// CHECK-SAME: hipcc_arch = "gfx1100"
//
// RDNA 3 WMMA bf16 intrinsic:
// CHECK-DAG: llvm.amdgcn.wmma.f32.16x16x16
