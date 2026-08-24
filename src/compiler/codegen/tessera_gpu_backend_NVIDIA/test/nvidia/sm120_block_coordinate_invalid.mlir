// RUN: not %tnv %s 2>&1 | FileCheck %s

module {
  llvm.func @bad_block_coordinate() attributes {nvvm.kernel} {
    %row, %col = "tessera_nvidia.block_coordinate"() {
      arch = "sm_120", tile_m = 32 : i64, tile_n = 8 : i64,
      grid_order = "column_major_xy"
    } : () -> (i64, i64)
    llvm.return
  }
}

// CHECK: error: 'tessera_nvidia.block_coordinate' op requires tile_m=16
