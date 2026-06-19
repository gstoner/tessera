// RUN: tessera-opt %s --allow-unregistered-dialect | FileCheck %s
//
// Full-config MiniMax-M3 decoder-block core as a Tessera IR artifact at
// PRODUCTION dimensions (H=6144, 64 Q heads / 4 KV heads, head_dim=128,
// 128 experts, F=3072): GQA + MiniMax Sparse Attention (MSA) + a BF16
// grouped-SwiGLU MoE expert FFN. This fixture proves the compiler-visible
// `tessera.msa_sparse_attention` / `moe_swiglu_block` ops verify at scale.
// It is an artifact contract only: MSA runtime decode, fused backend lowering,
// tokenizer/processor import, and vision/video execution remain future work.

// CHECK-LABEL: func.func @minimax_m3_block
// CHECK: tessera.msa_sparse_attention
// CHECK-SAME: block_size = 128
// CHECK-SAME: top_k = 16
// CHECK: tessera.moe_swiglu_block
// CHECK-SAME: storage = "bf16"
func.func @minimax_m3_block(
    %q: tensor<1x64x2048x128xf32>, %k: tensor<1x4x2048x128xf32>,
    %v: tensor<1x4x2048x128xf32>,
    %x: tensor<128x6144xf32>,
    %wg: tensor<128x6144x3072xf32>, %wu: tensor<128x6144x3072xf32>,
    %wd: tensor<128x3072x6144xf32>, %gs: tensor<128xi64>,
    %xs: tensor<128x1xf32>, %wgs: tensor<128x1xf32>, %wus: tensor<128x1xf32>,
    %wds: tensor<128x1xf32>) -> tensor<128x6144xf32> {
  // MSA over a production GQA shape. Sk=2048 and block_size=128 give exactly
  // 16 KV blocks, matching MiniMax-M3's released top-k block count.
  %attn = tessera.msa_sparse_attention %q, %k, %v {
            block_size = 128 : i64, top_k = 16 : i64,
            force_local_block = true, causal = true
          } : (tensor<1x64x2048x128xf32>, tensor<1x4x2048x128xf32>,
               tensor<1x4x2048x128xf32>) -> tensor<1x64x2048x128xf32>
  // BF16 grouped-SwiGLU MoE expert FFN (128 experts).
  %moe = tessera.moe_swiglu_block %x, %wg, %wu, %wd, %gs
           scales(%xs, %wgs, %wus, %wds) {
           grouped_kind = "contiguous", grouped_alignment = 128 : i64,
           scale_layout = {granularity = "block", block = [1, 128],
                           packing = "none", vector_size = 128 : i64,
                           alignment = 128 : i64, transposed = false},
           numeric_policy = {storage = "bf16", accum = "fp32"}
         } : (tensor<128x6144xf32>, tensor<128x6144x3072xf32>,
              tensor<128x6144x3072xf32>, tensor<128x3072x6144xf32>, tensor<128xi64>,
              tensor<128x1xf32>, tensor<128x1xf32>, tensor<128x1xf32>,
              tensor<128x1xf32>) -> tensor<128x6144xf32>
  return %moe : tensor<128x6144xf32>
}
