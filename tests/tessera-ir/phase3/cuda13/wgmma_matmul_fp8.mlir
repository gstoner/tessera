// RUN: tessera-opt --tessera-lower-to-gpu --gpu-target=sm_90a %s | FileCheck %s
// REQUIRES: tessera_opt_built
//
// Sprint G-4 — NVIDIA SM_90+ WGMMA matmul with FP8 storage (E4M3 inputs,
// fp32 accumulator).  FP8 doubles K relative to bf16: tile (64, 256, 32).

module attributes {tessera.target = "nvidia_sm90"} {
  func.func @wgmma_matmul_fp8(
      %A : memref<64x32xi8, 3>,    // fp8_e4m3 packed into i8 storage
      %B : memref<32x256xi8, 3>,
      %C : memref<64x256xf32, 3>) {
    "tessera.tile.wgmma"(%A, %B, %C) {
      tile_m = 64 : i64,
      tile_n = 256 : i64,
      tile_k = 32 : i64,
      elem_dtype = "fp8_e4m3",
      acc_dtype = "fp32",
      cuda_arch_min = "sm_90a"
    } : (memref<64x32xi8, 3>,
         memref<32x256xi8, 3>,
         memref<64x256xf32, 3>) -> ()
    return
  }
}

// CHECK: tessera.tile.wgmma
// CHECK-SAME: tile_k = 32
// CHECK-SAME: elem_dtype = "fp8_e4m3"
//
// PTX FP8 WGMMA — K is doubled to 32 vs bf16:
// CHECK-DAG: wgmma.mma_async.sync.aligned.m64n256k32
// CHECK-DAG: .f32.e4m3.e4m3
