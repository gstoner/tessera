// Current compiler artifact sample: Graph IR transpose/gather contract.
module attributes {tessera.ir.level = "graph", target = "cpu"} {
func.func @transpose_gather(%X: tensor<16x128x64xf32>) -> tensor<16x64x128xf32> {
  %0 = "tessera.transpose"(%X) {
    permutation = [0, 2, 1],
    runtime_status = "artifact_only"
  } : (tensor<16x128x64xf32>) -> tensor<16x64x128xf32>
  return %0 : tensor<16x64x128xf32>
}
}
