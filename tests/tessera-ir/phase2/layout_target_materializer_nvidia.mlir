// RUN: tessera-opt --tessera-nvidia-materialize-layout-casts %s \
// RUN:   | FileCheck %s --check-prefix=NVIDIA
// RUN: tessera-opt --tessera-nvidia-materialize-layout-casts \
// RUN:   --tessera-tile-ir-lowering='sm=120' %s \
// RUN:   | FileCheck %s --check-prefix=TILE

// NVIDIA materializes the coarse Graph layout into an operand-indexed physical
// #tile.layout. Schedule-to-Tile consumes that structured attribute directly;
// no NVIDIA string convention crosses the boundary.

// NVIDIA-LABEL: func.func @matmul_layout
// NVIDIA-NOT: "tessera.cast"
// NVIDIA: tessera.matmul %arg0, %arg1
// NVIDIA-SAME: tile.operand_layout_0 = #tile.layout<shard = [4, 8] : [8, 1] on ["m", "m"]
// TILE-LABEL: func.func @matmul_layout
// TILE: tile.async_copy %arg0
// TILE-SAME: tile.layout = #tile.layout<shard = [4, 8] : [8, 1] on ["m", "m"]
func.func @matmul_layout(%arg0: tensor<4x8xf32>,
                         %arg1: tensor<8x6xf32>) -> tensor<4x6xf32> {
  %a = "tessera.cast"(%arg0) {
    tessera.layout = "row_major", tessera.source_layout = "tile"
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %0 = "tessera.matmul"(%a, %arg1) :
      (tensor<4x8xf32>, tensor<8x6xf32>) -> tensor<4x6xf32>
  return %0 : tensor<4x6xf32>
}

// NVIDIA-LABEL: func.func @attention_layout
// NVIDIA-NOT: "tessera.cast"
// NVIDIA: tessera.flash_attn %arg0, %arg1, %arg2
// NVIDIA-SAME: tile.operand_layout_0 = #tile.layout<shard = [8, 4] : [4, 1] on ["m", "m"]
// TILE-LABEL: func.func @attention_layout
// TILE: tile.async_copy %arg0
// TILE-SAME: tile.layout = #tile.layout<shard = [8, 4] : [4, 1] on ["m", "m"]
func.func @attention_layout(
    %arg0: tensor<8x4xf32>, %arg1: tensor<8x4xf32>,
    %arg2: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %q = "tessera.cast"(%arg0) {tessera.layout = "bhsd"} :
      (tensor<8x4xf32>) -> tensor<8x4xf32>
  %0 = "tessera.flash_attn"(%q, %arg1, %arg2)
      <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, head_dim = 4 : i64}> :
      (tensor<8x4xf32>, tensor<8x4xf32>, tensor<8x4xf32>) ->
      tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}
