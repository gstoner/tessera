// RUN: not tessera-opt %s --tessera-autodiff-forward='export-scaled-program=true emit-storage-child=true' 2>&1 | FileCheck %s --check-prefix=MODE
// RUN: tessera-opt %s --tessera-autodiff-forward='export-scaled-program=true' | FileCheck %s

module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @scales(%a: tensor<17x256xf8E4M3FN>,
                    %b: tensor<256x19xf8E4M3FN>,
                    %sa: tensor<17x2xf32> {tessera.layout = "row_major", tessera.dim_names = ["row", "group"], tessera.model.parameter}, %sb: tensor<2x1xf32>)
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
// CHECK: tessera.autodiff.scaled_program
// CHECK-SAME: argument_count = 6
// CHECK-SAME: first_write = 0 : i64, id = 6 : i64, last_read = 4 : i64
// CHECK-SAME: ownership = "returned_output"
// CHECK-SAME: first_write = 1 : i64, id = 7 : i64, last_read = 3 : i64
// CHECK-SAME: ownership = "private_scratch"
// CHECK-SAME: first_write = 2 : i64, id = 8 : i64, last_read = 3 : i64
// CHECK-SAME: ownership = "private_scratch"
// CHECK-SAME: outputs = array<i64: 6, 9>
// CHECK-SAME: inputs = array<i64: 0, 1, 2, 3>, member = @scales__jvp__member_0, output = 6
// CHECK-SAME: inputs = array<i64: 0, 1, 4, 3>, member = @scales__jvp__member_1, output = 7
// CHECK-SAME: inputs = array<i64: 0, 1, 2, 5>, member = @scales__jvp__member_2, output = 8
// CHECK-SAME: inputs = array<i64: 7, 8>, member = @scales__jvp__member_3, output = 9
// CHECK-LABEL: func.func private @scales__jvp(
// CHECK-SAME: tessera.model.parameter
// CHECK-SAME: tensor<17x2xf32> {tessera.dim_names = ["row", "group"], tessera.layout = "row_major"}
// CHECK-LABEL: func.func private @scales__jvp__member_0(
// CHECK-SAME: tessera.model.parameter
// CHECK: tessera.scaled_matmul
// CHECK-SAME: block = [128, 128]
// CHECK-LABEL: func.func private @scales__jvp__member_1(
// CHECK-SAME: tensor<17x2xf32> {tessera.dim_names = ["row", "group"], tessera.layout = "row_major"}
// CHECK: tessera.scaled_matmul
// CHECK-LABEL: func.func private @scales__jvp__member_2(
// CHECK: tessera.scaled_matmul
// CHECK-LABEL: func.func private @scales__jvp__member_3(
// CHECK: tessera.add
// CHECK: return
// MODE: scaled JVP program export requires one request and no other export mode
