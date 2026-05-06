// Tiny Fast dLLM v2 compiler graph for the current Tessera dialect.
//
// The full design uses diffusion decode state, approximate KV cache blocks,
// branch fork/join, confidence stats, and validated-prefix merge.  This fixture
// keeps the straight-line tensor core that the current compiler can parse and
// lower today:
//   denoise matmul -> rmsnorm -> confidence softmax -> MLP update -> rmsnorm

module attributes {tessera.ir.version = "1.0", tessera.example = "Fast_dLLM_v2"} {
  func.func @fast_dllm_v2_confidence_decode_step(
      %x: tensor<32x64xf32>,
      %denoise_w: tensor<64x64xf32>,
      %confidence_w: tensor<64x64xf32>,
      %ff_in: tensor<64x128xf32>,
      %ff_out: tensor<128x64xf32>) -> tensor<32x64xf32>
      attributes {
        tessera.example = "Fast_dLLM_v2",
        tessera.decode.branches = 4 : i64,
        tessera.kv.block_tokens = 8 : i64,
        tessera.confidence_tau = 6.200000e-01 : f64
      } {
    %denoise = "tessera.matmul"(%x, %denoise_w) : (tensor<32x64xf32>, tensor<64x64xf32>) -> tensor<32x64xf32>
    %norm0 = "tessera.rmsnorm_safe"(%denoise) {eps = 1.0e-05 : f64} : (tensor<32x64xf32>) -> tensor<32x64xf32>
    %scores = "tessera.matmul"(%norm0, %confidence_w) : (tensor<32x64xf32>, tensor<64x64xf32>) -> tensor<32x64xf32>
    %confidence = "tessera.softmax"(%scores) {axis = -1 : i64} : (tensor<32x64xf32>) -> tensor<32x64xf32>
    %ff_proj = "tessera.matmul"(%confidence, %ff_in) : (tensor<32x64xf32>, tensor<64x128xf32>) -> tensor<32x128xf32>
    %ff_act = "tessera.relu"(%ff_proj) : (tensor<32x128xf32>) -> tensor<32x128xf32>
    %updated = "tessera.matmul"(%ff_act, %ff_out) : (tensor<32x128xf32>, tensor<128x64xf32>) -> tensor<32x64xf32>
    %norm1 = "tessera.rmsnorm_safe"(%updated) {eps = 1.0e-05 : f64} : (tensor<32x64xf32>) -> tensor<32x64xf32>
    return %norm1 : tensor<32x64xf32>
  }
}
