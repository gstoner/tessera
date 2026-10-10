// RUN: tessera-opt %s --tessera-autodiff-forward --tessera-graph-to-schedule --tessera-schedule-to-tile | FileCheck %s
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
}
// CHECK-LABEL: func.func private @scales__jvp(
// CHECK: tile.scaled_matmul_kernel
// CHECK-SAME: scale_k = 128
// CHECK-SAME: scale_fmt = "fp32"
// CHECK: tile.scaled_matmul_kernel
// CHECK-SAME: scale_k = 128
// CHECK: tile.scaled_matmul_kernel
// CHECK-SAME: scale_k = 128
// CHECK: tessera.add
// CHECK: return
