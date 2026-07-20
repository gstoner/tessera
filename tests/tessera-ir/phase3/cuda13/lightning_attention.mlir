// RUN: tessera-opt --tessera-lower-to-gpu --gpu-target=sm_90a %s | FileCheck %s
// REQUIRES: tessera_opt_built
//
// Sprint G-4 — MiniMax Lightning attention.  Linear-attention with a
// delta-rule state update.  The recurrence is serial; each step uses a
// small WGMMA (32, 32, 16) instead of the larger FA tile.

module attributes {tessera.target = "nvidia_sm90"} {
  func.func @lightning_attention(
      %Q : memref<1x32x1024x64xbf16, 1>,
      %K : memref<1x32x1024x64xbf16, 1>,
      %V : memref<1x32x1024x64xbf16, 1>,
      %S0 : memref<1x32x64x64xbf16, 1>,    // initial state
      %O : memref<1x32x1024x64xbf16, 1>) {
    "tessera_attn.lightning"(%Q, %K, %V, %S0, %O) {
      tile_q = 32 : i64,
      tile_kv = 32 : i64,
      tile_k = 16 : i64,
      head_dim = 64 : i64,
      cluster = array<i64: 1, 1, 1>,
      acc_dtype = "fp32",
      cuda_arch_min = "sm_90a"
    } : (memref<1x32x1024x64xbf16, 1>,
         memref<1x32x1024x64xbf16, 1>,
         memref<1x32x1024x64xbf16, 1>,
         memref<1x32x64x64xbf16, 1>,
         memref<1x32x1024x64xbf16, 1>) -> ()
    return
  }
}

// CHECK: tessera_attn.lightning
// CHECK-SAME: tile_q = 32
// CHECK-SAME: tile_kv = 32
//
// Small WGMMA shape for the recurrence step:
// CHECK-DAG: wgmma.mma_async.sync.aligned.m32n32k16
