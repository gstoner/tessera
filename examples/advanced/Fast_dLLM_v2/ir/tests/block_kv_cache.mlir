// Current compiler smoke for the Fast dLLM block-cache tensor skeleton.
// RUN: tessera-opt %s | FileCheck %s

module attributes {tessera.ir.version = "1.0", tessera.example = "Fast_dLLM_v2.block_cache"} {
  func.func @block_cache_pack_read_contract(
      %cache_rows: tensor<32x64xf32>,
      %pack_w: tensor<64x64xf32>,
      %read_w: tensor<64x64xf32>) -> tensor<32x64xf32>
      attributes {
        tessera.kv.block_tokens = 8 : i64,
        tessera.kv.approx = "int8_rowscale_with_fp16_boundary_stripes"
      } {
    %packed = "tessera.matmul"(%cache_rows, %pack_w) : (tensor<32x64xf32>, tensor<64x64xf32>) -> tensor<32x64xf32>
    %read = "tessera.matmul"(%packed, %read_w) : (tensor<32x64xf32>, tensor<64x64xf32>) -> tensor<32x64xf32>
    %norm = "tessera.rmsnorm_safe"(%read) {eps = 1.0e-05 : f64} : (tensor<32x64xf32>) -> tensor<32x64xf32>
    return %norm : tensor<32x64xf32>
  }
}

// CHECK: func.func @block_cache_pack_read_contract
// CHECK: tessera.kv.block_tokens = 8
// CHECK: "tessera.matmul"
// CHECK: "tessera.rmsnorm_safe"
