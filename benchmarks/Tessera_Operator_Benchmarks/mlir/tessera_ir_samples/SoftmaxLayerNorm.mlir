// Current compiler artifact sample: Graph IR softmax + layernorm contract.
module attributes {tessera.ir.level = "graph", target = "cpu"} {
func.func @softmax_layernorm(%X: tensor<16x128x64xf32>) -> tensor<16x128x64xf32> {
  %0 = "tessera.softmax"(%X) {
    axis = 1,
    runtime_status = "artifact_only"
  } : (tensor<16x128x64xf32>) -> tensor<16x128x64xf32>
  %1 = "tessera.layer_norm"(%0) {
    axis = 2,
    epsilon = 1.000000e-05 : f32,
    runtime_status = "artifact_only"
  } : (tensor<16x128x64xf32>) -> tensor<16x128x64xf32>
  return %1 : tensor<16x128x64xf32>
}
}
