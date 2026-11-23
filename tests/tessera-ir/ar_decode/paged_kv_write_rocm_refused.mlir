// RUN: not tessera-opt --tessera-graph-to-schedule %s 2>&1 | FileCheck %s
//
// AR-DECODE-PROGRAM-1 S1a: only sm_120 has an owning Target consumer for the
// paged write; a ROCm module refuses at Schedule instead of falling through.

module attributes {tessera.target = "rocm_gfx1201", tessera.arch = "gfx1201"} {
  func.func @paged_write(%pages: tensor<4x16x2x16xf32>, %table: tensor<4xi32>,
                         %values: tensor<1x2x16xf32>, %position: tensor<1xi32>) -> tensor<4x16x2x16xf32>
      attributes {tessera.bindings = ["pages", "page_table", "values", "position"]} {
    %updated = tessera.paged_kv_write %pages, %table, %values, %position
      : (tensor<4x16x2x16xf32>, tensor<4xi32>, tensor<1x2x16xf32>, tensor<1xi32>) -> tensor<4x16x2x16xf32>
    return %updated : tensor<4x16x2x16xf32>
  }
}

// CHECK: paged write needs an owning Target consumer; only sm_120 has one
