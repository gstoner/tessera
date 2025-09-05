; FlashAttention forward (illustrative)
func.func @flash_attention(%Q: tensor<1x16x1024x64xf16>, %K: tensor<1x16x1024x64xf16>, %V: tensor<1x16x1024x64xf16>) -> tensor<1x16x1024x64xf16> {
  %0 = tessera.flash_attention %Q, %K, %V {causal = true, tile=[64,64,32]} : (...) -> tensor<1x16x1024x64xf16>
  return %0 : tensor<1x16x1024x64xf16>
}
