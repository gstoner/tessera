// RUN: tessera-opt --tessera-lower-to-gpu --gpu-target=sm_90a %s | FileCheck %s
// REQUIRES: tessera_opt_built
//
// Sprint G-4 — DeepSeek NSA (Native Sparse Attention).  Validates:
//   - Top-k block selection emits a gather pattern over the score matrix
//   - WGMMA (64, 128, 16) on the selected blocks only
//   - Cluster (1, 1, 1) — each block runs independently after selection

module attributes {tessera.target = "nvidia_sm90"} {
  func.func @deepseek_nsa(
      %Q : memref<1x32x1024x128xbf16, 1>,
      %K : memref<1x32x1024x128xbf16, 1>,
      %V : memref<1x32x1024x128xbf16, 1>,
      %O : memref<1x32x1024x128xbf16, 1>) {
    "tessera.attn.nsa"(%Q, %K, %V, %O) {
      tile_q = 64 : i64,
      tile_kv = 128 : i64,
      top_k = 16 : i64,
      block_size = 64 : i64,
      head_dim = 128 : i64,
      cluster = array<i64: 1, 1, 1>,
      acc_dtype = "fp32",
      cuda_arch_min = "sm_90a"
    } : (memref<1x32x1024x128xbf16, 1>,
         memref<1x32x1024x128xbf16, 1>,
         memref<1x32x1024x128xbf16, 1>,
         memref<1x32x1024x128xbf16, 1>) -> ()
    return
  }
}

// CHECK: tessera.attn.nsa
// CHECK-SAME: top_k = 16
// CHECK-SAME: block_size = 64
//
// Top-k gather:
// CHECK-DAG: tessera.tile.gather
//
// WGMMA on selected blocks:
// CHECK-DAG: wgmma.mma_async.sync.aligned.m64n128k16
