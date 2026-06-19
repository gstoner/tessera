// RUN: tessera-opt %s --allow-unregistered-dialect | FileCheck %s
//
// Full-config GLM-5.2 decoder-block core as a Tessera IR artifact at PRODUCTION
// dimensions (H=6144, 64 heads, 256 experts, F=2048): MLA latent attention +
// DeepSeek Sparse Attention (DSA + IndexShare in the model contract) + a BF16
// grouped-SwiGLU MoE expert FFN. The registered `latent_kv_compress`,
// `tessera.deepseek_sparse_attention`, and `moe_swiglu_block` ops verify at
// scale.

// CHECK-LABEL: func.func @glm5_block
// CHECK: tessera.latent_kv_compress
// CHECK: tessera.deepseek_sparse_attention
// CHECK-SAME: block_size = 64
// CHECK: tessera.moe_swiglu_block
// CHECK-SAME: storage = "bf16"
func.func @glm5_block(
    %x: tensor<128x6144xf32>,
    %w_dkv: tensor<6144x512xf32>,
    %q: tensor<1x64x128x256xf32>, %k: tensor<1x64x128x256xf32>,
    %v: tensor<1x64x128x256xf32>,
    %wg: tensor<256x6144x2048xf32>, %wu: tensor<256x6144x2048xf32>,
    %wd: tensor<256x2048x6144xf32>, %gs: tensor<256xi64>,
    %xs: tensor<128x1xf32>, %wgs: tensor<256x1xf32>, %wus: tensor<256x1xf32>,
    %wds: tensor<256x1xf32>) -> tensor<128x6144xf32> {
  // MLA latent compression (kv_lora_rank=512).
  %c = tessera.latent_kv_compress %x, %w_dkv
       : (tensor<128x6144xf32>, tensor<6144x512xf32>) -> tensor<128x512xf32>
  // GLM-5.2 DSA over the expanded MLA Q/K/V tensors.
  %attn = tessera.deepseek_sparse_attention %q, %k, %v {
            window_size = 1024 : i64, block_size = 64 : i64, top_k = 32 : i64,
            causal = true
          } : (tensor<1x64x128x256xf32>, tensor<1x64x128x256xf32>,
               tensor<1x64x128x256xf32>) -> tensor<1x64x128x256xf32>
  // BF16 grouped-SwiGLU MoE expert FFN (256 experts).
  %moe = tessera.moe_swiglu_block %x, %wg, %wu, %wd, %gs
           scales(%xs, %wgs, %wus, %wds) {
           grouped_kind = "contiguous", grouped_alignment = 128 : i64,
           scale_layout = {granularity = "block", block = [1, 128],
                           packing = "none", vector_size = 128 : i64,
                           alignment = 128 : i64, transposed = false},
           numeric_policy = {storage = "bf16", accum = "fp32"}
         } : (tensor<128x6144xf32>, tensor<256x6144x2048xf32>,
              tensor<256x6144x2048xf32>, tensor<256x2048x6144xf32>, tensor<256xi64>,
              tensor<128x1xf32>, tensor<256x1xf32>, tensor<256x1xf32>,
              tensor<256x1xf32>) -> tensor<128x6144xf32>
  return %moe : tensor<128x6144xf32>
}
