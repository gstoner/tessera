// Current compiler artifact sample: Graph IR Conv2D NHWC contract.
module attributes {tessera.ir.level = "graph", target = "cpu"} {
func.func @conv2d_nhwc(%X: tensor<1x128x128x64xf32>, %W: tensor<3x3x64x128xf32>) -> tensor<1x128x128x128xf32> {
  %0 = "tessera.conv2d"(%X, %W) {
    data_format = "NHWC",
    kernel_format = "HWIO",
    stride = [1, 1],
    padding = [1, 1],
    runtime_status = "artifact_only"
  } : (tensor<1x128x128x64xf32>, tensor<3x3x64x128xf32>) -> tensor<1x128x128x128xf32>
  return %0 : tensor<1x128x128x128xf32>
}
}
