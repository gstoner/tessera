// Current compiler artifact sample: Graph IR FlashAttention contract.
module attributes {tessera.ir.level = "graph", target = "cpu"} {
func.func @flash_attention(%Q: tensor<1x16x1024x64xf32>, %K: tensor<1x16x1024x64xf32>, %V: tensor<1x16x1024x64xf32>) -> tensor<1x16x1024x64xf32> {
  %0 = "tessera.flash_attn"(%Q, %K, %V) {
    causal = true,
    tile_q = 64,
    tile_kv = 64,
    runtime_status = "artifact_only"
  } : (tensor<1x16x1024x64xf32>, tensor<1x16x1024x64xf32>, tensor<1x16x1024x64xf32>) -> tensor<1x16x1024x64xf32>
  return %0 : tensor<1x16x1024x64xf32>
}
}
