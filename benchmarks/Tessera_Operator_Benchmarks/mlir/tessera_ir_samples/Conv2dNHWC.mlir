; NHWC Conv2D with fused epilogue (illustrative)
func.func @conv2d_nhwc(%X: tensor<1x128x128x64xf16>, %W: tensor<3x3x64x128xf16>, %B: tensor<128xf16>) -> tensor<1x128x128x128xf16> {
  %0 = tessera.conv2d_nhwc %X, %W {stride=[1,1], pad=[1,1], tile=[64,64,32]} : (...) -> tensor<1x128x128x128xf16>
  %1 = tessera.bias_gelu %0, %B : (...) -> tensor<1x128x128x128xf16>
  return %1 : tensor<1x128x128x128xf16>
}
