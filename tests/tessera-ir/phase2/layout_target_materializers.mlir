// REQUIRES: tessera-apple-backend
//
// RUN: tessera-opt --tessera-apple-materialize-layout-casts %s | FileCheck %s --check-prefix=APPLE
// RUN: tessera-opt --tessera-x86-materialize-layout-casts %s | FileCheck %s --check-prefix=X86
// RUN: tessera-opt --tessera-nvidia-materialize-layout-casts %s | FileCheck %s --check-prefix=NVIDIA
// RUN: tessera-opt --tessera-nvidia-materialize-layout-casts --tessera-tile-ir-lowering='sm=120' %s | FileCheck %s --check-prefix=NVIDIA-TILE

// Architecture-owned Graph-cast consumers replace the same-type marker with
// its tensor value and leave an operand-indexed physical binding contract on
// the target consumer. Source-layout provenance survives the boundary.

// APPLE-LABEL: func.func @matmul_layout
// APPLE-NOT: "tessera.cast"
// APPLE: tessera.matmul %arg0, %arg1
// APPLE-SAME: tessera.apple.operand_layout_0 = "row_major"
// APPLE-SAME: tessera.apple.source_layout_0 = "tile"
// NVIDIA-LABEL: func.func @matmul_layout
// NVIDIA-NOT: "tessera.cast"
// NVIDIA: tessera.matmul %arg0, %arg1
// NVIDIA-SAME: tessera.nvidia.operand_layout_0 = "row_major"
// NVIDIA-SAME: tessera.nvidia.source_layout_0 = "tile"
// X86-LABEL: func.func @matmul_layout
// X86-NOT: "tessera.cast"
// X86: tessera.matmul %arg0, %arg1
// X86-SAME: tessera.x86.operand_layout_0 = "row_major"
// X86-SAME: tessera.x86.source_layout_0 = "tile"
// NVIDIA-TILE-LABEL: func.func @matmul_layout
// NVIDIA-TILE: tile.async_copy %arg0
// NVIDIA-TILE-SAME: tessera.nvidia.layout = "row_major"
func.func @matmul_layout(%arg0: tensor<4x8xf32>,
                         %arg1: tensor<8x6xf32>) -> tensor<4x6xf32> {
  %a = "tessera.cast"(%arg0) {
    tessera.layout = "row_major", tessera.source_layout = "tile"
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %0 = "tessera.matmul"(%a, %arg1) :
      (tensor<4x8xf32>, tensor<8x6xf32>) -> tensor<4x6xf32>
  return %0 : tensor<4x6xf32>
}

// APPLE-LABEL: func.func @attention_layout
// APPLE-NOT: "tessera.cast"
// APPLE: tessera.flash_attn %arg0, %arg1, %arg2
// APPLE-SAME: tessera.apple.operand_layout_0 = "bhsd"
// NVIDIA-LABEL: func.func @attention_layout
// NVIDIA-NOT: "tessera.cast"
// NVIDIA: tessera.flash_attn %arg0, %arg1, %arg2
// NVIDIA-SAME: tessera.nvidia.operand_layout_0 = "bhsd"
// NVIDIA-TILE-LABEL: func.func @attention_layout
// NVIDIA-TILE: tile.async_copy %arg0
// NVIDIA-TILE-SAME: tessera.nvidia.layout = "bhsd"
func.func @attention_layout(
    %arg0: tensor<1x2x8x4xf32>, %arg1: tensor<1x2x8x4xf32>,
    %arg2: tensor<1x2x8x4xf32>) -> tensor<1x2x8x4xf32> {
  %q = "tessera.cast"(%arg0) {tessera.layout = "bhsd"} :
      (tensor<1x2x8x4xf32>) -> tensor<1x2x8x4xf32>
  %0 = "tessera.flash_attn"(%q, %arg1, %arg2)
      <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, head_dim = 4 : i64}> :
      (tensor<1x2x8x4xf32>, tensor<1x2x8x4xf32>, tensor<1x2x8x4xf32>) ->
      tensor<1x2x8x4xf32>
  return %0 : tensor<1x2x8x4xf32>
}
