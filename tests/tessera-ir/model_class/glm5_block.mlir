// RUN: tessera-opt %s --allow-unregistered-dialect | FileCheck %s
//
// Full-config GLM-5 (placeholder dims) decoder-block core as a Tessera IR
// artifact: GQA attention + DeepSeek Sparse Attention (DSA) + an FP8 block-scaled
// MoE expert FFN at H=5120, 160 experts, F=1536. The registered
// `tessera.deepseek_sparse_attention` / `moe_swiglu_block` ops verify at scale.
// GLM-5 dimensions are unconfirmed (see python/tessera/models/glm5.py).

// CHECK-LABEL: func.func @glm5_block
// CHECK: tessera.deepseek_sparse_attention
// CHECK-SAME: block_size = 64
// CHECK: tessera.moe_swiglu_block
// CHECK-SAME: storage = "fp8_e4m3"
func.func @glm5_block(
    %q: tensor<1x96x128x128xf32>, %k: tensor<1x8x128x128xf32>,
    %v: tensor<1x8x128x128xf32>,
    %x: tensor<128x5120xf32>,
    %wg: tensor<160x5120x1536xf32>, %wu: tensor<160x5120x1536xf32>,
    %wd: tensor<160x1536x5120xf32>, %gs: tensor<160xi64>,
    %xs: tensor<128x1xf32>, %wgs: tensor<160x1xf32>, %wus: tensor<160x1xf32>,
    %wds: tensor<160x1xf32>) -> tensor<128x5120xf32> {
  // GQA (96 query / 8 KV heads) DeepSeek Sparse Attention.
  %attn = tessera.deepseek_sparse_attention %q, %k, %v {
            window_size = 1024 : i64, block_size = 64 : i64, top_k = 32 : i64,
            causal = true
          } : (tensor<1x96x128x128xf32>, tensor<1x8x128x128xf32>,
               tensor<1x8x128x128xf32>) -> tensor<1x96x128x128xf32>
  // FP8 block-scaled grouped-SwiGLU MoE expert FFN (160 experts).
  %moe = tessera.moe_swiglu_block %x, %wg, %wu, %wd, %gs
           scales(%xs, %wgs, %wus, %wds) {
           grouped_kind = "contiguous", grouped_alignment = 128 : i64,
           scale_layout = {granularity = "block", block = [1, 128],
                           packing = "ue8m0", vector_size = 128 : i64,
                           alignment = 128 : i64, transposed = false},
           numeric_policy = {storage = "fp8_e4m3", accum = "fp32"}
         } : (tensor<128x5120xf32>, tensor<160x5120x1536xf32>,
              tensor<160x5120x1536xf32>, tensor<160x1536x5120xf32>, tensor<160xi64>,
              tensor<128x1xf32>, tensor<160x1xf32>, tensor<160x1xf32>,
              tensor<160x1xf32>) -> tensor<128x5120xf32>
  return %moe : tensor<128x5120xf32>
}
