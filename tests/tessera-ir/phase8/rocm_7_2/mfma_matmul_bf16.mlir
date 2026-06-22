// RUN: tessera-opt --tessera-lower-to-rocm --rocm-target=gfx942 %s | FileCheck %s
// REQUIRES: tessera_opt_built
//
// Sprint H-4 (2026-05-11) — AMD CDNA 3 MFMA matmul (bf16 storage, fp32
// accumulator).  Validates the canonical (M=32, N=32, K=8, K_blocks=1)
// shape emitted by Tessera's MFMA lowering pass under ROCm 7.2.4 + HIP
// 7.2.4 for gfx942 (MI300X).

module attributes {tessera.target = "rocm_gfx942"} {
  func.func @mfma_matmul_bf16(
      %A : memref<32x8xbf16, 3>,    // LDS = address space 3
      %B : memref<8x32xbf16, 3>,
      %C : memref<32x32xf32, 3>) {
    "tessera_rocm.mfma"(%A, %B, %C) {
      mfma_shape = array<i64: 32, 32, 8, 1>,
      acc_dtype = "fp32",
      hipcc_arch = "gfx942"
    } : (memref<32x8xbf16, 3>,
         memref<8x32xbf16, 3>,
         memref<32x32xf32, 3>) -> ()
    return
  }
}

// CHECK: tessera_rocm.mfma
// CHECK-SAME: mfma_shape = array<i64: 32, 32, 8, 1>
// CHECK-SAME: hipcc_arch = "gfx942"
//
// AMDGCN intrinsic emission contract — bf16 32x32x8 MFMA:
// CHECK-DAG: llvm.amdgcn.mfma.f32.32x32x8bf16.1k
