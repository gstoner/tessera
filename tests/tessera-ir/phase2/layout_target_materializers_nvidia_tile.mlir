// REQUIRES: tessera-nvidia-backend
//
// RUN: tessera-opt --tessera-nvidia-materialize-layout-casts --tessera-tile-ir-lowering='sm=120' %s | FileCheck %s

// The SM120 half of layout_target_materializers.mlir. SM120 Schedule delegation
// requires an explicit nvidia_sm120 target and sm_120 architecture on the module
// ([TILE_IR_LOWERING]), and a registered tessera_nvidia dialect, so this runs
// only in a build with the NVIDIA backend. The structured #tile.layout the
// NVIDIA materializer attaches (shard [4, 8] : [8, 1]) must survive the SM120
// route, which carries it as the composed layout feeding tile.view and the
// mma_sync fragments (the retired generic route used tile.async_copy).
// Matmul only: the SM120 Graph->Schedule route takes f16/bf16 inputs, and has no
// rank-2 flash_attn contract (the attention layout is still checked by the NVIDIA
// materialize RUN in layout_target_materializers.mlir).

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
// CHECK-LABEL: llvm.func @nvidia_sm120_scheduled_matmul_matmul_layout_kernel
// CHECK: tile.materialize_composed_layout {{.*}} <[4, 8], [8, 1],
// CHECK: tile.view %arg0
// CHECK: tile.fragment_pack {{.*}} role = "a"
// CHECK: tile.mma
func.func @matmul_layout(%arg0: tensor<4x8xbf16>,
                         %arg1: tensor<8x6xbf16>) -> tensor<4x6xf32> {
  %a = "tessera.cast"(%arg0) {
    tessera.layout = "row_major", tessera.source_layout = "tile"
  } : (tensor<4x8xbf16>) -> tensor<4x8xbf16>
  %0 = "tessera.matmul"(%a, %arg1) :
      (tensor<4x8xbf16>, tensor<8x6xbf16>) -> tensor<4x6xf32>
  return %0 : tensor<4x6xf32>
}
}
