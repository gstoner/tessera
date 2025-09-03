
// RUN: not tessera-opt -tessera-verify %s 2>&1 | FileCheck %s
module {
  func.func @bad_kind(%x: tensor<4x4xf32>, %w: tensor<4x4xf32>, %b: tensor<4xf32>) -> tensor<4x4xf32> {
    %f = "tessera.fused_epilogue"(%x, %w, %b) {epilogue = "leaky_relu", has_bias = true}
         : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4xf32>) -> tensor<4x4xf32>
    return %f : tensor<4x4xf32>
  }
}
// CHECK: error: [TESSERA_VFY_EPILOGUE_KIND] unsupported epilogue kind
