// RUN: tessera-opt --pass-pipeline='builtin.module(tessera-autodiff-forward{export-scaled-primal=true})' %s | FileCheck %s

// Regression: emitNativeScaledMatmulProgram expanded a cloned floating scaled
// product (which erases it) and then tested the freed operation with
// isNativeScaledCarrier. glibc tolerates reading freed memory, so Linux drivers
// passed; macOS returned a recycled/zeroed operation and tessera-opt died with
// EXC_BAD_ACCESS (address 0x10) inside isa<BroadcastOp, ReduceOp, ReshapeOp>.
// The export must complete and keep the scaled product as a native member.

module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  // CHECK: tessera.autodiff.scaled_program
  // CHECK-LABEL: func.func @floating_product
  func.func @floating_product(%a: tensor<2x9xf32>, %b: tensor<9x5xf32>,
                             %sa: tensor<2x3xf32>, %sb: tensor<3x2xf32>)
      -> tensor<2x5xf32> {
    %y = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      transposeA = false, transposeB = false,
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [4, 4], format = "fp32"}
    } : (tensor<2x9xf32>, tensor<9x5xf32>, tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x5xf32>
    return %y : tensor<2x5xf32>
  }
  // CHECK-LABEL: func.func private @floating_product__member_0
  // CHECK: tessera.native.scaled_product
}
