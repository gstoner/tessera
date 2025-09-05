; Minimal Tessera IR (illustrative)
func.func @matmul(%A: tensor<1024x1024xf16>, %B: tensor<1024x1024xf16>) -> tensor<1024x1024xf16> {
  %0 = tessera.matmul %A, %B { tile = [128, 128, 32], accumulate = f32 } : (tensor<1024x1024xf16>, tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  return %0 : tensor<1024x1024xf16>
}
