// RUN: tessera-opt %s --tessera-autodiff-paired | FileCheck %s
// RUN: tessera-opt %s --tessera-autodiff -o /dev/null
// Native scale transpose preserves shared batch reductions and ragged groups.
module {

  func.func @shared_rhs_rows(%a: tensor<2x3x7x256xf8E4M3FN>, %b: tensor<256x129xf8E4M3FN>,
                     %sa: tensor<2x3x7x2xf32>, %sb: tensor<2x2xf32>)
      -> tensor<2x3x7x129xf32>
      attributes {tessera.autodiff = "reverse", tessera.autodiff.wrt_indices = [2, 3]} {
    %y = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      batching = "shared_rhs_rows",
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [128, 128], format = "fp32"}
    } : (tensor<2x3x7x256xf8E4M3FN>, tensor<256x129xf8E4M3FN>, tensor<2x3x7x2xf32>, tensor<2x2xf32>) -> tensor<2x3x7x129xf32>
    return %y : tensor<2x3x7x129xf32>
  }

  func.func @independent_rhs(%a: tensor<2x3x7x256xf8E4M3FN>, %b: tensor<2x3x256x129xf8E4M3FN>,
                     %sa: tensor<2x3x7x2xf32>, %sb: tensor<2x3x2x2xf32>)
      -> tensor<2x3x7x129xf32>
      attributes {tessera.autodiff = "reverse", tessera.autodiff.wrt_indices = [2, 3]} {
    %y = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      batching = "independent_rhs",
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [128, 128], format = "fp32"}
    } : (tensor<2x3x7x256xf8E4M3FN>, tensor<2x3x256x129xf8E4M3FN>, tensor<2x3x7x2xf32>, tensor<2x3x2x2xf32>) -> tensor<2x3x7x129xf32>
    return %y : tensor<2x3x7x129xf32>
  }

  func.func @shared_lhs(%a: tensor<7x256xf8E4M3FN>, %b: tensor<2x3x256x129xf8E4M3FN>,
                     %sa: tensor<7x2xf32>, %sb: tensor<2x3x2x2xf32>)
      -> tensor<2x3x7x129xf32>
      attributes {tessera.autodiff = "reverse", tessera.autodiff.wrt_indices = [2, 3]} {
    %y = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      batching = "shared_lhs",
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [128, 128], format = "fp32"}
    } : (tensor<7x256xf8E4M3FN>, tensor<2x3x256x129xf8E4M3FN>, tensor<7x2xf32>, tensor<2x3x2x2xf32>) -> tensor<2x3x7x129xf32>
    return %y : tensor<2x3x7x129xf32>
  }
  func.func @transposed_ragged(%a: tensor<13x2xf8E4M3FN>,
      %b: tensor<7x13xf8E4M3FN>, %sa: tensor<2x3xf32>, %sb: tensor<3x3xf32>)
      -> tensor<2x7xf32>
      attributes {tessera.autodiff = "reverse", tessera.autodiff.wrt_indices = [2, 3]} {
    %y = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [3, 5], format = "fp32"},
      transposeA = true, transposeB = true
    } : (tensor<13x2xf8E4M3FN>, tensor<7x13xf8E4M3FN>,
         tensor<2x3xf32>, tensor<3x3xf32>) -> tensor<2x7xf32>
    return %y : tensor<2x7xf32>
  }
}
// CHECK-LABEL: func.func @shared_rhs_rows__bwd
// CHECK: tensor.generate
// CHECK: scf.for
// CHECK: arith.extf
// CHECK: arith.mulf
// CHECK: tensor.generate
// CHECK: return
// CHECK-LABEL: func.func @independent_rhs__bwd
// CHECK: tensor.generate
// CHECK: scf.for
// CHECK: arith.extf
// CHECK: arith.mulf
// CHECK: tensor.generate
// CHECK: return
// CHECK-LABEL: func.func @shared_lhs__bwd
// CHECK: tensor.generate
// CHECK: scf.for
// CHECK: arith.extf
// CHECK: arith.mulf
// CHECK: tensor.generate
// CHECK: return
// CHECK-LABEL: func.func @transposed_ragged__bwd
// CHECK: tensor.generate
// CHECK: arith.minui
// CHECK: scf.for
// CHECK: tensor.generate
// CHECK: return
