// Current compiler artifact sample: Graph IR elementwise contract.
module attributes {tessera.ir.level = "graph", target = "cpu"} {
func.func @elementwise(%X: tensor<1048576xf32>) -> tensor<1048576xf32> {
  %0 = "tessera.tanh"(%X) {runtime_status = "artifact_only"} :
    (tensor<1048576xf32>) -> tensor<1048576xf32>
  %1 = "tessera.mul"(%X) {scalar = 1.000000e-01 : f32, runtime_status = "artifact_only"} :
    (tensor<1048576xf32>) -> tensor<1048576xf32>
  %2 = "tessera.add"(%0, %1) {runtime_status = "artifact_only"} :
    (tensor<1048576xf32>, tensor<1048576xf32>) -> tensor<1048576xf32>
  return %2 : tensor<1048576xf32>
}
}
