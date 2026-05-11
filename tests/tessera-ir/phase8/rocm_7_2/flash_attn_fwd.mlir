// RUN: tessera-opt --tessera-lower-to-rocm --rocm-target=gfx942 %s | FileCheck %s
// REQUIRES: tessera_opt_built
//
// Sprint H-4 — Flash-Attention forward on CDNA 3 / MI300X.  Smaller
// MFMA tile (16x16x16) for the narrow-N score matrix.  rocm-FA2 idiom:
// online softmax + LDS-staged double-buffer for KV streaming.

module attributes {tessera.target = "rocm_gfx942"} {
  func.func @flash_attn_fwd_rocm(
      %Q : memref<1x32x1024x128xbf16, 1>,
      %K : memref<1x32x1024x128xbf16, 1>,
      %V : memref<1x32x1024x128xbf16, 1>,
      %O : memref<1x32x1024x128xbf16, 1>) {
    "tessera_rocm.flash_attn_fwd"(%Q, %K, %V, %O) {
      tile_q = 64 : i64,
      tile_kv = 64 : i64,
      head_dim = 128 : i64,
      mfma_shape = array<i64: 16, 16, 16, 1>,
      pipeline_stages = 2 : i64,
      acc_dtype = "fp32",
      hipcc_arch = "gfx942"
    } : (memref<1x32x1024x128xbf16, 1>,
         memref<1x32x1024x128xbf16, 1>,
         memref<1x32x1024x128xbf16, 1>,
         memref<1x32x1024x128xbf16, 1>) -> ()
    return
  }
}

// CHECK: tessera_rocm.flash_attn_fwd
// CHECK-SAME: mfma_shape = array<i64: 16, 16, 16, 1>
// CHECK-SAME: pipeline_stages = 2
//
// Two MFMA passes (score QK^T + value P@V):
// CHECK-COUNT-2: llvm.amdgcn.mfma.f32.16x16x16bf16.1k
//
// LDS async copy for the KV staging:
// CHECK-DAG: llvm.amdgcn.global.load.lds
// CHECK-DAG: llvm.amdgcn.s.barrier
