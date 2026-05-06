// Tiny FlashMLA compiler graph for the current Tessera dialect.
//
// The full design uses paged latent KV cache blocks, decoupled RoPE, online
// softmax, and weight-absorbed K/V projections.  This fixture keeps the
// straight-line tensor core that the current compiler can parse and lower
// today.

module attributes {tessera.ir.version = "1.0", tessera.example = "mla"} {
  func.func @flash_mla_tiny_prefill(
      %x: tensor<16x64xf32>,
      %q_down: tensor<64x16xf32>,
      %q_up: tensor<16x64xf32>,
      %kv_down: tensor<64x16xf32>,
      %k_absorb: tensor<16x64xf32>,
      %score_w: tensor<64x16xf32>,
      %v_absorb: tensor<16x64xf32>,
      %out_w: tensor<64x64xf32>) -> tensor<16x64xf32>
      attributes {
        tessera.example = "mla",
        tessera.mla.latent_dim = 16 : i64,
        tessera.mla.num_q_heads = 4 : i64,
        tessera.mla.num_kv_heads = 2 : i64
      } {
    %q_latent = "tessera.matmul"(%x, %q_down) : (tensor<16x64xf32>, tensor<64x16xf32>) -> tensor<16x16xf32>
    %q_full = "tessera.matmul"(%q_latent, %q_up) : (tensor<16x16xf32>, tensor<16x64xf32>) -> tensor<16x64xf32>
    %kv_latent_raw = "tessera.matmul"(%x, %kv_down) : (tensor<16x64xf32>, tensor<64x16xf32>) -> tensor<16x16xf32>
    %kv_latent = "tessera.rmsnorm_safe"(%kv_latent_raw) {eps = 1.0e-05 : f64} : (tensor<16x16xf32>) -> tensor<16x16xf32>
    %k_full = "tessera.matmul"(%kv_latent, %k_absorb) : (tensor<16x16xf32>, tensor<16x64xf32>) -> tensor<16x64xf32>
    %score_input = "tessera.relu"(%q_full) : (tensor<16x64xf32>) -> tensor<16x64xf32>
    %scores = "tessera.matmul"(%score_input, %score_w) : (tensor<16x64xf32>, tensor<64x16xf32>) -> tensor<16x16xf32>
    %probs = "tessera.softmax"(%scores) {axis = -1 : i64} : (tensor<16x16xf32>) -> tensor<16x16xf32>
    %v_full = "tessera.matmul"(%kv_latent, %v_absorb) : (tensor<16x16xf32>, tensor<16x64xf32>) -> tensor<16x64xf32>
    %context = "tessera.matmul"(%probs, %v_full) : (tensor<16x16xf32>, tensor<16x64xf32>) -> tensor<16x64xf32>
    %out = "tessera.matmul"(%context, %out_w) : (tensor<16x64xf32>, tensor<64x64xf32>) -> tensor<16x64xf32>
    %norm_out = "tessera.rmsnorm_safe"(%out) {eps = 1.0e-05 : f64} : (tensor<16x64xf32>) -> tensor<16x64xf32>
    return %norm_out : tensor<16x64xf32>
  }
}
