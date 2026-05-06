// Current compiler artifact sample: Graph IR reduce contract.
module attributes {tessera.ir.level = "graph", target = "cpu"} {
func.func @reduce_sum(%X: tensor<1048576xf32>) -> tensor<f32> {
  %0 = "tessera.reduce"(%X) {
    kind = "sum",
    axis = [0],
    runtime_status = "artifact_only"
  } : (tensor<1048576xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}
}
