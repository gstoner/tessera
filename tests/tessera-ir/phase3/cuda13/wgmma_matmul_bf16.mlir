// RUN: tessera-opt --tessera-lower-to-gpu --gpu-target=sm_90a %s | FileCheck %s
// REQUIRES: tessera_opt_built
//
// Sprint G-4 (2026-05-11) — NVIDIA SM_90+ WGMMA matmul (bf16 storage,
// fp32 accumulator).  Validates the canonical Hopper tile shape
// (M=64, N=256, K=16) emitted by Tessera's WGMMA lowering pass under
// CUDA Toolkit 13.2 Update 1.
//
// Inputs: A[M, K] bf16, B[K, N] bf16, C[M, N] fp32 accumulator
// Tile shape: (64, 256, 16) — the cuBLAS-recommended WGMMA tile.
// Cluster: (1, 1, 1) — no producer-consumer specialization for plain matmul.

module attributes {tessera.target = "nvidia_sm90"} {
  func.func @wgmma_matmul_bf16(
      %A : memref<64x16xbf16, 3>,
      %B : memref<16x256xbf16, 3>,
      %C : memref<64x256xf32, 3>) {
    "tessera.tile.wgmma"(%A, %B, %C) {
      tile_m = 64 : i64,
      tile_n = 256 : i64,
      tile_k = 16 : i64,
      acc_dtype = "fp32",
      cuda_arch_min = "sm_90a"
    } : (memref<64x16xbf16, 3>,
         memref<16x256xbf16, 3>,
         memref<64x256xf32, 3>) -> ()
    return
  }
}

// CHECK: tessera.tile.wgmma
// CHECK-SAME: tile_m = 64
// CHECK-SAME: tile_n = 256
// CHECK-SAME: tile_k = 16
// CHECK-SAME: acc_dtype = "fp32"
// CHECK-SAME: cuda_arch_min = "sm_90a"
//
// PTX emission contract — the lowering pass must materialize the canonical
// Hopper bf16 WGMMA instruction:
// CHECK-DAG: wgmma.mma_async.sync.aligned.m64n256k16
// CHECK-DAG: .f32.bf16.bf16
// CHECK-DAG: mbarrier.arrive.expect_tx
