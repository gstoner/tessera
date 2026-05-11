// RUN: tessera-opt --tessera-lower-to-gpu --gpu-target=sm_100a %s | FileCheck %s
// REQUIRES: tessera_opt_built
//
// Sprint G-4 — Blackwell SM_100+ tcgen05 matmul.  Validates:
//   - tcgen05.mma instruction emission (5th-gen tensor core contract)
//   - TMEM allocation / load / store
//   - CTA-pair scheduling for paired-MMA contracts
//   - NVFP4 block-scaled MMA (Blackwell exclusive)

module attributes {tessera.target = "nvidia_sm100"} {
  func.func @tcgen05_matmul_nvfp4(
      %A : memref<128x64xi8, 3>,           // nvfp4 packed
      %B : memref<64x128xi8, 3>,
      %C : memref<128x128xf32, 3>) {
    "tessera.tile.tcgen05_mma"(%A, %B, %C) {
      tile_m = 128 : i64,
      tile_n = 128 : i64,
      tile_k = 64 : i64,
      elem_dtype = "nvfp4",
      acc_dtype = "fp32",
      cta_group = 2 : i64,
      cuda_arch_min = "sm_100a",
      block_scaled = true
    } : (memref<128x64xi8, 3>,
         memref<64x128xi8, 3>,
         memref<128x128xf32, 3>) -> ()
    return
  }
}

// CHECK: tessera.tile.tcgen05_mma
// CHECK-SAME: elem_dtype = "nvfp4"
// CHECK-SAME: cta_group = 2
// CHECK-SAME: block_scaled = true
//
// PTX patterns:
// CHECK-DAG: tcgen05.alloc
// CHECK-DAG: tcgen05.mma.cta_group::2
// CHECK-DAG: tcgen05.commit
// CHECK-DAG: tcgen05.dealloc
