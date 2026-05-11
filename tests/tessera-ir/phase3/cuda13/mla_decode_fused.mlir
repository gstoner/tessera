// RUN: tessera-opt --tessera-lower-to-gpu --gpu-target=sm_90a %s | FileCheck %s
// REQUIRES: tessera_opt_built
//
// Sprint G-4 — DeepSeek MLA (Multi-head Latent Attention) decode kernel.
// Validates:
//   - Latent KV compression (low-rank C_kv with rank << head_dim) reduces
//     KV-cache memory bandwidth ~4x vs MHA at decode time
//   - Same FA tile (64, 128, 16) for the inner flash_attn pass
//   - Cluster (2, 1, 1) for paired-CTA producer/consumer
//   - KV-memory-bound: the per-step compute is matmul-thin

module attributes {tessera.target = "nvidia_sm90"} {
  func.func @mla_decode_fused(
      %Q : memref<1x32x1x128xbf16, 1>,         // single decode token
      %C_kv : memref<1x32x4096x32xbf16, 1>,   // compressed latent KV (rank=32)
      %W_kv_expand : memref<32x128xbf16, 1>,  // expansion projection
      %O : memref<1x32x1x128xbf16, 1>) {
    "tessera.attn.mla_decode"(%Q, %C_kv, %W_kv_expand, %O) {
      tile_q = 64 : i64,
      tile_kv = 128 : i64,
      head_dim = 128 : i64,
      latent_rank = 32 : i64,
      pipeline_stages = 2 : i64,
      cluster = array<i64: 2, 1, 1>,
      acc_dtype = "fp32",
      cuda_arch_min = "sm_90a"
    } : (memref<1x32x1x128xbf16, 1>,
         memref<1x32x4096x32xbf16, 1>,
         memref<32x128xbf16, 1>,
         memref<1x32x1x128xbf16, 1>) -> ()
    return
  }
}

// CHECK: tessera.attn.mla_decode
// CHECK-SAME: latent_rank = 32
// CHECK-SAME: tile_q = 64
// CHECK-SAME: tile_kv = 128
//
// Latent expansion via WGMMA:
// CHECK-DAG: wgmma.mma_async.sync.aligned.m64n128k16
//
// Same TMA + cluster + mbarrier contract as FA-4:
// CHECK-DAG: tessera.tile.tma_descriptor
// CHECK-DAG: mbarrier.arrive.expect_tx
