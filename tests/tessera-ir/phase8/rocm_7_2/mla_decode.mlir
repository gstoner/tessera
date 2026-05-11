// RUN: tessera-opt --tessera-lower-to-rocm --rocm-target=gfx942 %s | FileCheck %s
// REQUIRES: tessera_opt_built
//
// Sprint H-4 — DeepSeek MLA decode on MI300X.  Latent KV expansion via
// MFMA (16, 16, 16, 1); KV-cache-memory-bound, lower MFU than FA fwd.

module attributes {tessera.target = "rocm_gfx942"} {
  func.func @mla_decode_rocm(
      %Q : memref<1x32x1x128xbf16, 1>,
      %C_kv : memref<1x32x4096x32xbf16, 1>,   // latent rank=32
      %W_kv_expand : memref<32x128xbf16, 1>,
      %O : memref<1x32x1x128xbf16, 1>) {
    "tessera_rocm.mla_decode"(%Q, %C_kv, %W_kv_expand, %O) {
      tile_q = 64 : i64,
      tile_kv = 64 : i64,
      head_dim = 128 : i64,
      latent_rank = 32 : i64,
      mfma_shape = array<i64: 16, 16, 16, 1>,
      acc_dtype = "fp32",
      hipcc_arch = "gfx942"
    } : (memref<1x32x1x128xbf16, 1>,
         memref<1x32x4096x32xbf16, 1>,
         memref<32x128xbf16, 1>,
         memref<1x32x1x128xbf16, 1>) -> ()
    return
  }
}

// CHECK: tessera_rocm.mla_decode
// CHECK-SAME: latent_rank = 32
// CHECK-SAME: mfma_shape = array<i64: 16, 16, 16, 1>
//
// MFMA for the latent expansion + score pass:
// CHECK-DAG: llvm.amdgcn.mfma.f32.16x16x16bf16.1k
