// RUN: tessera-opt --tessera-lower-to-gpu --gpu-target=sm_90a %s | FileCheck %s
// REQUIRES: tessera_opt_built
//
// MSA Phase 3 CUDA contract fixture. This does not claim a native kernel yet;
// it locks the intended KV-outer sparse-attention target shape and the selected
// block worklist layout consumed by a future lowering.

module attributes {tessera.target = "nvidia_sm90"} {
  func.func @msa_kv_outer_prefill(
      %Q : memref<1x8x1024x128xbf16, 1>,
      %K : memref<1x2x1024x128xbf16, 1>,
      %V : memref<1x2x1024x128xbf16, 1>,
      %block_ids : memref<1x2x1024x8xi64, 1>,
      %O : memref<1x8x1024x128xbf16, 1>) {
    "tessera_attn.msa_kv_outer_sparse"(%Q, %K, %V, %block_ids, %O) {
      block_size = 64 : i64,
      top_k = 8 : i64,
      gqa_group_size = 4 : i64,
      tile_q = 64 : i64,
      tile_kv = 128 : i64,
      head_dim = 128 : i64,
      mode = "prefill",
      acc_dtype = "fp32",
      dense_equivalence_oracle = false,
      cuda_arch_min = "sm_90a"
    } : (memref<1x8x1024x128xbf16, 1>,
         memref<1x2x1024x128xbf16, 1>,
         memref<1x2x1024x128xbf16, 1>,
         memref<1x2x1024x8xi64, 1>,
         memref<1x8x1024x128xbf16, 1>) -> ()
    return
  }

  func.func @msa_kv_outer_decode_dense_oracle(
      %Q : memref<1x8x1x128xbf16, 1>,
      %K : memref<1x2x1024x128xbf16, 1>,
      %V : memref<1x2x1024x128xbf16, 1>,
      %block_ids : memref<1x2x1x16xi64, 1>,
      %O : memref<1x8x1x128xbf16, 1>) {
    "tessera_attn.msa_kv_outer_sparse"(%Q, %K, %V, %block_ids, %O) {
      block_size = 64 : i64,
      top_k = 16 : i64,
      gqa_group_size = 4 : i64,
      tile_q = 1 : i64,
      tile_kv = 128 : i64,
      head_dim = 128 : i64,
      mode = "decode",
      acc_dtype = "fp32",
      dense_equivalence_oracle = true,
      cuda_arch_min = "sm_90a"
    } : (memref<1x8x1x128xbf16, 1>,
         memref<1x2x1024x128xbf16, 1>,
         memref<1x2x1024x128xbf16, 1>,
         memref<1x2x1x16xi64, 1>,
         memref<1x8x1x128xbf16, 1>) -> ()
    return
  }
}

// CHECK-LABEL: func.func @msa_kv_outer_prefill
// CHECK: tessera_attn.msa_kv_outer_sparse
// CHECK-SAME: block_size = 64
// CHECK-SAME: gqa_group_size = 4
// CHECK-SAME: mode = "prefill"
// CHECK-SAME: top_k = 8

// CHECK-LABEL: func.func @msa_kv_outer_decode_dense_oracle
// CHECK: tessera_attn.msa_kv_outer_sparse
// CHECK-SAME: dense_equivalence_oracle = true
// CHECK-SAME: mode = "decode"
// CHECK-SAME: top_k = 16
