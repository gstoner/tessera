// RUN: tessera-opt %s --tessera-autodiff-forward | FileCheck %s
// Scale seeds remain in their original K groups; no ordinary GEMM replacement.
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @scales(%a: tensor<17x256xf8E4M3FN>,
                    %b: tensor<256x19xf8E4M3FN>,
                    %sa: tensor<17x2xf32>, %sb: tensor<2x1xf32>)
      -> tensor<17x19xf32>
      attributes {tessera.autodiff = "forward",
                  tessera.autodiff.wrt_indices = [2, 3]} {
    %y = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [128, 128], format = "fp32"},
      transposeA = false, transposeB = false
    } : (tensor<17x256xf8E4M3FN>, tensor<256x19xf8E4M3FN>,
         tensor<17x2xf32>, tensor<2x1xf32>) -> tensor<17x19xf32>
    return %y : tensor<17x19xf32>
  }
  func.func @transposed_lhs(%a: tensor<32x16xf8E4M3FN>,
                            %b: tensor<16x32xf8E4M3FN>,
                            %sa: tensor<16x1xi8>, %sb: tensor<1x16xi8>)
      -> tensor<16x16xf32>
      attributes {tessera.autodiff = "forward",
                  tessera.autodiff.wrt_indices = [0]} {
    %y = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [1, 32], format = "e8m0"},
      transposeA = true, transposeB = true
    } : (tensor<32x16xf8E4M3FN>, tensor<16x32xf8E4M3FN>,
         tensor<16x1xi8>, tensor<1x16xi8>) -> tensor<16x16xf32>
    return %y : tensor<16x16xf32>
  }
}
// CHECK-LABEL: func.func private @scales__jvp(
// CHECK-SAME: %[[A:.*]]: tensor<17x256xf8E4M3FN>, %[[B:.*]]: tensor<256x19xf8E4M3FN>, %[[SA:.*]]: tensor<17x2xf32>, %[[SB:.*]]: tensor<2x1xf32>, %[[DA:.*]]: tensor<17x2xf32>, %[[DB:.*]]: tensor<2x1xf32>
// CHECK: %[[P:.*]] = tessera.scaled_matmul %[[A]], %[[B]] scales(%[[SA]], %[[SB]])
// CHECK: %[[L:.*]] = tessera.scaled_matmul %[[A]], %[[B]] scales(%[[DA]], %[[SB]])
// CHECK-SAME: execution_mode = "exact_per_block"
// CHECK-SAME: block = [128, 128]
// CHECK: %[[R:.*]] = tessera.scaled_matmul %[[A]], %[[B]] scales(%[[SA]], %[[DB]])
// CHECK: %[[D:.*]] = tessera.add %[[L]], %[[R]]
// CHECK-NEXT: return %[[P]], %[[D]]
// CHECK-LABEL: func.func private @transposed_lhs__jvp(
// CHECK-SAME: %[[TA:.*]]: tensor<32x16xf8E4M3FN>, %[[TB:.*]]: tensor<16x32xf8E4M3FN>, %[[TSA:.*]]: tensor<16x1xi8>, %[[TSB:.*]]: tensor<1x16xi8>, %[[TDA:.*]]: tensor<32x16xf8E4M3FN>
// CHECK: %[[TP:.*]] = tessera.scaled_matmul %[[TA]], %[[TB]] scales(%[[TSA]], %[[TSB]])
// CHECK: %[[TD:.*]] = tessera.scaled_matmul %[[TDA]], %[[TB]] scales(%[[TSA]], %[[TSB]])
// CHECK-SAME: transposeA = true
// CHECK-SAME: transposeB = true
// CHECK-NEXT: return %[[TP]], %[[TD]]
