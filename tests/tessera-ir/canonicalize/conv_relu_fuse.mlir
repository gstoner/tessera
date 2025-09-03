
// RUN: tessera-opt -tessera-canonicalize %s | FileCheck %s
module {
  func.func @conv_relu(%x: tensor<1x32x32x64xf32>, %w: tensor<3x3x64x64xf32>) -> tensor<1x32x32x64xf32> {
    %c = "tessera.conv2d_nhwc"(%x, %w) {strides = [1,1], dilations = [1,1]} 
       : (tensor<1x32x32x64xf32>, tensor<3x3x64x64xf32>) -> tensor<1x32x32x64xf32>
    %r = "tessera.relu"(%c) : (tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32>
    return %r : tensor<1x32x32x64xf32>
  }
}
// CHECK: "tessera.conv2d_nhwc"
// CHECK-SAME: epilogue = "relu"
