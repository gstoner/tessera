
// RUN: tessera-tpu-opt %s -tessera-lower-conv-to-stablehlo | FileCheck %s
module {
  func.func @conv(%inp: tensor<1x32x32x64xbf16>, %w: tensor<3x3x64x128xbf16>, %b: tensor<128xbf16>) -> tensor<1x32x32x128xbf16> {
    %0 = "tessera.conv2d_bias_gelu"(%inp, %w, %b) : 
         (tensor<1x32x32x64xbf16>, tensor<3x3x64x128xbf16>, tensor<128xbf16>) -> tensor<1x32x32x128xbf16>
    return %0 : tensor<1x32x32x128xbf16>
  }
}
// CHECK: stablehlo.convolution
// CHECK: stablehlo.add
// CHECK: stablehlo.tanh
