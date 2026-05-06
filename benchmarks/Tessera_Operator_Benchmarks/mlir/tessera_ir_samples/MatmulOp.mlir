// Current compiler artifact sample: Graph IR matmul contract.
module attributes {tessera.ir.level = "graph", target = "cpu"} {
func.func @matmul(%A: tensor<1024x1024xf32>, %B: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
  %0 = "tessera.matmul"(%A, %B) {
    shape = "1024x1024x1024",
    dtype = "f32",
    layout = "row_major",
    numeric_policy = "f32_accum"
  } : (tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  return %0 : tensor<1024x1024xf32>
}
}
