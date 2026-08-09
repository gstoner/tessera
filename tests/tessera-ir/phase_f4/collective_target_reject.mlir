// RUN: not tessera-opt --tessera-lower-tile-collectives %s 2>&1 | FileCheck %s

module {
  func.func @bad(%x: tensor<7xf32>) -> tensor<3xf32> {
    // Tile verification fails closed before target lowering: seven elements
    // cannot be reduce-scattered across two ranks.
    %s = "tile.reduce_scatter"(%x) {
      mesh_axis = "dp", tensor_axis = 0 : i64, reduction = "sum",
      world_size = 2 : i64
    } : (tensor<7xf32>) -> tensor<3xf32>
    return %s : tensor<3xf32>
  }
}

// CHECK: collective-axis extent disagrees with world_size
