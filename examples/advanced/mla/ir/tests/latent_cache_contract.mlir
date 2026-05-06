// Current compiler smoke for MLA latent cache compression metadata.
// RUN: tessera-opt %s | FileCheck %s

module attributes {tessera.ir.version = "1.0", tessera.example = "mla.latent_cache"} {
  func.func @latent_cache_pack_read_contract(
      %hidden: tensor<16x64xf32>,
      %kv_down: tensor<64x16xf32>,
      %read_w: tensor<16x64xf32>) -> tensor<16x64xf32>
      attributes {
        tessera.mla.latent_dim = 16 : i64,
        tessera.mla.cache = "compressed_latent_rows"
      } {
    %latent_raw = "tessera.matmul"(%hidden, %kv_down) : (tensor<16x64xf32>, tensor<64x16xf32>) -> tensor<16x16xf32>
    %latent = "tessera.rmsnorm_safe"(%latent_raw) {eps = 1.0e-05 : f64} : (tensor<16x16xf32>) -> tensor<16x16xf32>
    %read = "tessera.matmul"(%latent, %read_w) : (tensor<16x16xf32>, tensor<16x64xf32>) -> tensor<16x64xf32>
    return %read : tensor<16x64xf32>
  }
}

// CHECK: func.func @latent_cache_pack_read_contract
// CHECK: tessera.mla.latent_dim = 16
// CHECK: "tessera.matmul"
// CHECK: "tessera.rmsnorm_safe"
