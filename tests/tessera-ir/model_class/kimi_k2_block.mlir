// RUN: tessera-opt %s --allow-unregistered-dialect | FileCheck %s
//
// Full-config Kimi-K2 decoder-block core as a Tessera IR artifact at PRODUCTION
// dimensions (H=7168, 64 heads, 384 experts, F=2048): MLA latent attention + a
// native INT4 group-scaled MoE expert FFN. Kimi has no DSA (dense MLA). The INT4
// weight contract rides on `numeric_policy.storage = "int4"` + a per-group
// (block, packing="none") `scale_layout` on the registered
// `tessera.moe_swiglu_block` (the M1/M2 quant path).

// CHECK-LABEL: func.func @kimi_k2_block
// CHECK: tessera.latent_kv_compress
// CHECK: tessera.moe_swiglu_block
// CHECK-SAME: storage = "int4"
// CHECK-SAME: packing = "none"
func.func @kimi_k2_block(
    %x: tensor<128x7168xf32>,
    %w_dkv: tensor<7168x512xf32>,
    %wg: tensor<384x7168x2048xf32>, %wu: tensor<384x7168x2048xf32>,
    %wd: tensor<384x2048x7168xf32>, %gs: tensor<384xi64>,
    %xs: tensor<128x1xf32>, %wgs: tensor<384x1xf32>, %wus: tensor<384x1xf32>,
    %wds: tensor<384x1xf32>) -> tensor<128x7168xf32> {
  // MLA latent compression (kv_lora_rank=512).
  %c = tessera.latent_kv_compress %x, %w_dkv
       : (tensor<128x7168xf32>, tensor<7168x512xf32>) -> tensor<128x512xf32>
  // Native INT4 per-group (128) MoE expert FFN (384 experts).
  %moe = tessera.moe_swiglu_block %x, %wg, %wu, %wd, %gs
           scales(%xs, %wgs, %wus, %wds) {
           grouped_kind = "contiguous", grouped_alignment = 128 : i64,
           scale_layout = {granularity = "block", block = [128, 1],
                           packing = "none", vector_size = 128 : i64,
                           alignment = 128 : i64, transposed = false},
           numeric_policy = {storage = "int4", accum = "fp32"}
         } : (tensor<128x7168xf32>, tensor<384x7168x2048xf32>,
              tensor<384x7168x2048xf32>, tensor<384x2048x7168xf32>, tensor<384xi64>,
              tensor<128x1xf32>, tensor<384x1xf32>, tensor<384x1xf32>,
              tensor<384x1xf32>) -> tensor<128x7168xf32>
  return %moe : tensor<128x7168xf32>
}
