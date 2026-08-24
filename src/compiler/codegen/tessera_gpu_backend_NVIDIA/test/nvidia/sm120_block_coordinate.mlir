// RUN: %tnv --lower-tile-to-nvidia='sm=120' %s | FileCheck %s

module {
  llvm.func @block_coordinate() attributes {nvvm.kernel} {
    %row, %col = "tessera_nvidia.block_coordinate"() {
      arch = "sm_120", tile_m = 16 : i64, tile_n = 8 : i64,
      grid_order = "column_major_xy"
    } : () -> (i64, i64)
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @block_coordinate
// CHECK: %[[X32:.*]] = nvvm.read.ptx.sreg.ctaid.x : i32
// CHECK: %[[X64:.*]] = arith.extui %[[X32]] : i32 to i64
// CHECK: %[[Y32:.*]] = nvvm.read.ptx.sreg.ctaid.y : i32
// CHECK: %[[Y64:.*]] = arith.extui %[[Y32]] : i32 to i64
// CHECK: %[[ROW:.*]] = arith.muli %[[Y64]], %{{.*}} : i64
// CHECK: %[[COL:.*]] = arith.muli %[[X64]], %{{.*}} : i64
// CHECK-NOT: tessera_nvidia.block_coordinate
