// RUN: tessera-opt %s --allow-unregistered-dialect | FileCheck %s
//
// Full-config DeepSeek-V3.2 decoder-block core as a Tessera IR artifact at
// PRODUCTION dimensions (H=7168, 128 heads, 256 experts, F=2048): MLA latent
// attention + DeepSeek Sparse Attention (DSA) + an FP8 block-scaled MoE expert
// FFN + the fused dequant-grouped-GEMM op. All four registered ops
// (`latent_kv_compress`, `deepseek_sparse_attention`, `moe_swiglu_block`,
// `dequant_grouped_gemm`) are verified at full scale, proving the artifact is a
// stable, valid IR module at full config (execution stays hardware-gated — Phase
// G).

// CHECK-LABEL: func.func @deepseek_v32_block
// CHECK: tessera.latent_kv_compress
// CHECK: tessera.deepseek_sparse_attention
// CHECK-SAME: top_k = 32
// CHECK: tessera.moe_swiglu_block
// CHECK-SAME: storage = "fp8_e4m3"
// CHECK-SAME: granularity = "block"
// CHECK: tessera.dequant_grouped_gemm
// CHECK-SAME: weight_dtype = "fp8_e4m3"
func.func @deepseek_v32_block(
    %x: tensor<128x7168xf32>,
    %w_dkv: tensor<7168x512xf32>,
    %q: tensor<1x128x128x128xf32>, %k: tensor<1x128x128x128xf32>,
    %v: tensor<1x128x128x128xf32>,
    %wg: tensor<256x7168x2048xf32>, %wu: tensor<256x7168x2048xf32>,
    %wd: tensor<256x2048x7168xf32>, %gs: tensor<256xi64>,
    %xs: tensor<128x1xf32>, %wgs: tensor<256x1xf32>, %wus: tensor<256x1xf32>,
    %wds: tensor<256x1xf32>) -> tensor<128x7168xf32> {
  // MLA latent compression (kv_lora_rank=512) — the cacheable latent.
  %c = tessera.latent_kv_compress %x, %w_dkv
       : (tensor<128x7168xf32>, tensor<7168x512xf32>) -> tensor<128x512xf32>
  // DeepSeek Sparse Attention over the 128-head Q/K/V (block-sparse, top-k=32).
  %attn = tessera.deepseek_sparse_attention %q, %k, %v {
            window_size = 1024 : i64, block_size = 64 : i64, top_k = 32 : i64,
            causal = true
          } : (tensor<1x128x128x128xf32>, tensor<1x128x128x128xf32>,
               tensor<1x128x128x128xf32>) -> tensor<1x128x128x128xf32>
  // FP8 block-scaled grouped-SwiGLU MoE expert FFN (256 experts).
  %moe = tessera.moe_swiglu_block %x, %wg, %wu, %wd, %gs
           scales(%xs, %wgs, %wus, %wds) {
           grouped_kind = "contiguous", grouped_alignment = 128 : i64,
           scale_layout = {granularity = "block", block = [1, 128],
                           packing = "ue8m0", vector_size = 128 : i64,
                           alignment = 128 : i64, transposed = false},
           numeric_policy = {storage = "fp8_e4m3", accum = "fp32"}
         } : (tensor<128x7168xf32>, tensor<256x7168x2048xf32>,
              tensor<256x7168x2048xf32>, tensor<256x2048x7168xf32>, tensor<256xi64>,
              tensor<128x1xf32>, tensor<256x1xf32>, tensor<256x1xf32>,
              tensor<256x1xf32>) -> tensor<128x7168xf32>
  // Fused dequantize-into-grouped-GEMM (packed FP8 expert weights + per-group
  // scales) — the M5.1 first-class IR op.
  %dq = tessera.dequant_grouped_gemm %x, %wg, %gs scale(%wgs) {
          grouped_kind = "contiguous", weight_dtype = "fp8_e4m3",
          quant_group_size = 128 : i64,
          scale_layout = {granularity = "block", block = [1, 128],
                          packing = "ue8m0", vector_size = 128 : i64,
                          alignment = 128 : i64, transposed = false},
          numeric_policy = {storage = "fp8_e4m3", accum = "fp32"}
        } : (tensor<128x7168xf32>, tensor<256x7168x2048xf32>, tensor<256xi64>,
             tensor<256x1xf32>) -> tensor<128x2048xf32>
  return %moe : tensor<128x7168xf32>
}
