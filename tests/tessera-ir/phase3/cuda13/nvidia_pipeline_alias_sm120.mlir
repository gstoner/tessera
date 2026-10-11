// REQUIRES: tessera-nvidia-backend
//
// RUN: tessera-opt --tessera-nvidia-pipeline-sm120 %s | FileCheck %s

// The SM120 half of nvidia_pipeline_alias.mlir. The SM120 route projects
// registered matmuls through Graph -> Schedule -> Tile before residual generic
// lowering, and refuses with "SM120 scheduled matmul requires the registered
// NVIDIA Target IR dialect" in a build without the NVIDIA backend, so it is
// gated here rather than in the core alias fixture.

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @entry(%A : tensor<64x16xbf16>,
                   %B : tensor<16x256xbf16>) -> tensor<64x256xf32> {
    %C = "tessera.matmul"(%A, %B) : (tensor<64x16xbf16>,
                                     tensor<16x256xbf16>) -> tensor<64x256xf32>
    return %C : tensor<64x256xf32>
  }
}

// CHECK-NOT: tessera.matmul
// CHECK-NOT: tile.async_copy
// CHECK: tile.fragment_pack {{.*}} : (!tile.tile) -> !tile.fragment
// CHECK: tile.fragment_pack {{.*}} : (!tile.tile) -> !tile.fragment
// CHECK: tile.mma {{.*}} -> !tile.fragment
// CHECK: tile.fragment_unpack
// CHECK-NOT: tessera.matmul
// CHECK-NOT: tile.async_copy
