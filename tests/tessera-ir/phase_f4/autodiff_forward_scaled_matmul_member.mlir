// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt %s --tessera-autodiff-forward='export-scaled-program=true select-scaled-member=1' | FileCheck %s --check-prefix=MEMBER
// RUN: tessera-opt %s --pass-pipeline='builtin.module(tessera-autodiff-forward{export-scaled-program=true select-scaled-member=1},tessera-graph-to-schedule,tessera-schedule-to-tile)' | FileCheck %s --check-prefix=TILE
// RUN: tessera-opt %s --pass-pipeline='builtin.module(tessera-autodiff-forward{export-scaled-program=true select-scaled-member=3},tessera-graph-to-schedule,tessera-schedule-to-tile)' | FileCheck %s --check-prefix=SUM
// RUN: not tessera-opt %s --tessera-autodiff-forward='export-scaled-program=true select-scaled-member=4' 2>&1 | FileCheck %s --check-prefix=BOUND

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
// MEMBER: tessera.autodiff.scaled_member
// MEMBER-SAME: inputs = array<i64: 0, 1, 4, 3>
// MEMBER-SAME: output = 7
// MEMBER: tessera.autodiff.scaled_program_witness
// MEMBER-LABEL: {{^  }}func.func private @scales__jvp__member_1(
// MEMBER: tessera.scaled_matmul
// MEMBER-SAME: execution_mode = "exact_per_block"
// MEMBER-SAME: block = [128, 128]
// TILE-LABEL: {{^  }}func.func private @scales__jvp__member_1(
// TILE-NOT: tessera.scaled_matmul
// TILE: tile.scaled_matmul_kernel
// TILE-SAME: scale_k = 128
// TILE-SAME: scale_fmt = "fp32"
// SUM: tessera.autodiff.scaled_member
// SUM-SAME: inputs = array<i64: 7, 8>
// SUM-SAME: output = 9
// SUM: tessera.rocm_math_contract
// SUM: tile.elementwise_kernel
// SUM-SAME: kind = "add"
// BOUND: scaled JVP member index is outside its native program
