// Tiny Nemotron Nano M/*/- compiler graph for the current Tessera dialect.
//
// The full model includes embeddings, Mamba2 state updates, GQA, residuals, and
// LM head projection.  This fixture keeps only the straight-line tensor core
// skeleton that the current compiler can verify and lower today:
//   Mamba stub: matmul -> relu -> matmul
//   Attention stub: matmul -> softmax -> matmul
//   MLP stub: matmul -> relu -> matmul -> rmsnorm_safe

module attributes {tessera.ir.version = "1.0", tessera.example = "Nemotron_Nano_12B_v2"} {
  func.func @nemotron_nano_tiny_m_star_dash(
      %x: tensor<32x64xf32>,
      %m_in: tensor<64x128xf32>,
      %m_out: tensor<128x64xf32>,
      %attn_q: tensor<64x64xf32>,
      %attn_o: tensor<64x64xf32>,
      %mlp_in: tensor<64x128xf32>,
      %mlp_out: tensor<128x64xf32>) -> tensor<32x64xf32>
      attributes {tessera.example = "Nemotron_Nano_12B_v2", tessera.hybrid_pattern = "M*-"} {
    %m_proj = "tessera.matmul"(%x, %m_in) : (tensor<32x64xf32>, tensor<64x128xf32>) -> tensor<32x128xf32>
    %m_act = "tessera.relu"(%m_proj) : (tensor<32x128xf32>) -> tensor<32x128xf32>
    %m_outv = "tessera.matmul"(%m_act, %m_out) : (tensor<32x128xf32>, tensor<128x64xf32>) -> tensor<32x64xf32>
    %q_proj = "tessera.matmul"(%m_outv, %attn_q) : (tensor<32x64xf32>, tensor<64x64xf32>) -> tensor<32x64xf32>
    %attn_prob = "tessera.softmax"(%q_proj) {axis = -1 : i64} : (tensor<32x64xf32>) -> tensor<32x64xf32>
    %attn_out = "tessera.matmul"(%attn_prob, %attn_o) : (tensor<32x64xf32>, tensor<64x64xf32>) -> tensor<32x64xf32>
    %mlp_proj = "tessera.matmul"(%attn_out, %mlp_in) : (tensor<32x64xf32>, tensor<64x128xf32>) -> tensor<32x128xf32>
    %mlp_act = "tessera.relu"(%mlp_proj) : (tensor<32x128xf32>) -> tensor<32x128xf32>
    %mlp_outv = "tessera.matmul"(%mlp_act, %mlp_out) : (tensor<32x128xf32>, tensor<128x64xf32>) -> tensor<32x64xf32>
    %norm = "tessera.rmsnorm_safe"(%mlp_outv) {eps = 1.0e-05 : f64} : (tensor<32x64xf32>) -> tensor<32x64xf32>
    return %norm : tensor<32x64xf32>
  }
}
