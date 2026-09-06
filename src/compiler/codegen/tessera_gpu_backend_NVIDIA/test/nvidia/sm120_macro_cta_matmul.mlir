// RUN: %tnv --lower-tile-to-nvidia='sm=120' %s | FileCheck %s
// RUN: %tnv --tessera-lower-to-nvidia-sm120 %s | FileCheck %s --check-prefix=NATIVE

module {
  llvm.func @macro(%a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr,
                   %m: i64, %n: i64, %k: i64) attributes {nvvm.kernel} {
    "tessera_nvidia.macro_cta_matmul"(%a, %b, %d, %m, %n, %k) {
      arch = "sm_120", cta_m = 32 : i64, cta_n = 32 : i64,
      tile_m = 16 : i64, tile_n = 8 : i64, tile_k = 16 : i64,
      warps = 4 : i64,
      warp_ownership = "quadrant_2x2_two_n_tiles",
      storage = "f16", accum = "f32",
      staging = "cp_async_shared_ab_16bit", stages = 2 : i64,
      completion = "wait_group_0_cta_barrier", bounds = "zero_fill_mnk_tail",
      grid_order = "column_major_xy",
      tessera.schedule_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    } : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    llvm.return
  }
}

// CHECK: llvm.mlir.global internal @__tessera_sm120_ab_stage_f16
// CHECK-LABEL: llvm.func @macro
// CHECK: nvvm.read.ptx.sreg.ctaid.x
// CHECK: nvvm.read.ptx.sreg.ctaid.y
// CHECK: nvvm.read.ptx.sreg.tid.x
// CHECK: nvgpu.device_async_copy
// CHECK: nvgpu.device_async_create_group
// CHECK: nvgpu.device_async_wait
// CHECK: nvvm.barrier
// CHECK: tessera_nvidia.mma_sync
// CHECK: nvvm.barrier
// CHECK-NOT: tessera_nvidia.macro_cta_matmul

// NATIVE: nvvm.cp.async.shared.global
// NATIVE: nvvm.cp.async.commit.group
// NATIVE: nvvm.cp.async.wait.group 0
// NATIVE: nvvm.mma.sync
// NATIVE-NOT: nvgpu.
