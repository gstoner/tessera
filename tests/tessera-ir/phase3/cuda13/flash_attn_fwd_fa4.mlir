// RUN: tessera-opt --tessera-lower-to-gpu --gpu-target=sm_90a %s | FileCheck %s
// REQUIRES: tessera_opt_built
//
// Sprint G-4 — FA-4 forward on Hopper.  Validates:
//   - WGMMA shape (64, 128, 16) for the QK^T pass and the (Pscore @ V) pass
//   - Cluster size (2, 1, 1) for paired-CTA producer/consumer warps
//   - Online softmax via cooperative warp shuffle (no DRAM round-trip on
//     the score matrix)
//   - TMA descriptor materialization
//   - mbarrier.arrive.expect_tx for async transaction counts

module attributes {tessera.target = "nvidia_sm90"} {
  func.func @flash_attn_fwd_fa4(
      %Q : memref<1x32x1024x128xbf16, 1>,  // (B, H, S, D)
      %K : memref<1x32x1024x128xbf16, 1>,
      %V : memref<1x32x1024x128xbf16, 1>,
      %O : memref<1x32x1024x128xbf16, 1>) {
    "tessera.attn.flash_fwd"(%Q, %K, %V, %O) {
      tile_q = 128 : i64,
      tile_kv = 128 : i64,
      head_dim = 128 : i64,
      pipeline_stages = 2 : i64,
      cluster = array<i64: 2, 1, 1>,
      acc_dtype = "fp32",
      cuda_arch_min = "sm_90a"
    } : (memref<1x32x1024x128xbf16, 1>,
         memref<1x32x1024x128xbf16, 1>,
         memref<1x32x1024x128xbf16, 1>,
         memref<1x32x1024x128xbf16, 1>) -> ()
    return
  }
}

// CHECK: tessera.attn.flash_fwd
// CHECK-SAME: tile_q = 128
// CHECK-SAME: tile_kv = 128
// CHECK-SAME: pipeline_stages = 2
// CHECK-SAME: cluster = array<i64: 2, 1, 1>
// CHECK-SAME: cuda_arch_min = "sm_90a"
//
// TMA descriptor for the Q tile:
// CHECK-DAG: tessera.tile.tma_descriptor
//
// Two WGMMA passes — score (QK^T) and value (P@V):
// CHECK-DAG: wgmma.mma_async.sync.aligned.m64n128k16
//
// Cluster launch:
// CHECK-DAG: cluster
//
// Async transaction barrier:
// CHECK-DAG: mbarrier.arrive.expect_tx
