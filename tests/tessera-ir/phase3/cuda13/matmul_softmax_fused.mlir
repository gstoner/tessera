// RUN: tessera-opt --tessera-lower-to-gpu --gpu-target=sm_90a %s | FileCheck %s
// REQUIRES: tessera_opt_built
//
// Sprint G-4 — Fused matmul→softmax kernel.  Score matrix stays in
// register file / SMEM — no DRAM round trip.  WGMMA (64, 256, 16)
// for the matmul; warp-shuffle row reduction for the softmax.

module attributes {tessera.target = "nvidia_sm90"} {
  func.func @matmul_softmax_fused(
      %A : memref<64x256xbf16, 3>,
      %B : memref<256x256xbf16, 3>,
      %Y : memref<64x256xbf16, 3>) {
    "tessera.tile.matmul_softmax"(%A, %B, %Y) {
      tile_m = 64 : i64,
      tile_n = 256 : i64,
      tile_k = 16 : i64,
      softmax_axis = -1 : i64,
      acc_dtype = "fp32",
      cuda_arch_min = "sm_90a"
    } : (memref<64x256xbf16, 3>,
         memref<256x256xbf16, 3>,
         memref<64x256xbf16, 3>) -> ()
    return
  }
}

// CHECK: tessera.tile.matmul_softmax
// CHECK-SAME: tile_m = 64
// CHECK-SAME: tile_n = 256
//
// WGMMA emits the score matrix in registers:
// CHECK-DAG: wgmma.mma_async.sync.aligned.m64n256k16
//
// No store of the intermediate score matrix — the softmax row reduce
// runs cooperatively in registers:
// CHECK-DAG: shfl.sync.bfly
