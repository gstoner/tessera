
// RUN: not tessera-opt -tessera-verify %s 2>&1 | FileCheck %s
module {
  func.func @bad_stride(%x: tensor<1x32x32x64xf32>, %w: tensor<3x3x64x64xf32>) -> tensor<1x32x32x64xf32> {
    %c = "tessera.conv2d_nhwc"(%x, %w) {strides = [0,1], dilations = [1,1]} 
       : (tensor<1x32x32x64xf32>, tensor<3x3x64x64xf32>) -> tensor<1x32x32x64xf32>
    return %c : tensor<1x32x32x64xf32>
  }
}
// CHECK: error: [TESSERA_VFY_CONV_ATTR] strides must be positive integers
