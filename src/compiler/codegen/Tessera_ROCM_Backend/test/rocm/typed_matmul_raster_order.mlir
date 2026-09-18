// RUN: %trop --allow-unregistered-dialect --generate-wmma-gemm-kernel='via-tile=true' --lower-tile-to-rocm='arch=gfx1151' %s | FileCheck %s
// RUN: %trop --allow-unregistered-dialect --generate-wmma-gemm-kernel='via-tile=true' --lower-tile-to-rocm='arch=gfx1201' %s | FileCheck %s
//
// The shared block-rasterization contract reaches the TYPED route
// (ROCM-RASTER-1, 2026-09-18). `tile.matmul_kernel` carries the Schedule's
// `tessera.raster_order` / `tessera.raster_group`; TileToROCM hands them to
// the generator under the directive lane's spelling, and the generator emits
// the grouped block-id remap (the `grouped_m` panel swizzle divides the
// flattened block id by the group) instead of the row-major identity. The
// selection itself stays row-major in Graph->Schedule until device timing
// and counters exist; this fixture proves the carriage, not a decision.

module {
  func.func @grouped(%a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr,
                     %m: i64, %n: i64, %k: i64) {
    tile.matmul_kernel %a, %b, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">,
      warps = 1 : i64, staging = "global",
      tessera.macro_tile_m = 32 : i64, tessera.macro_tile_n = 64 : i64,
      tessera.raster_order = "grouped_m", tessera.raster_group = 4 : i64
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    return
  }
}

// CHECK-NOT: tile.matmul_kernel
// CHECK: gpu.func @grouped(
// CHECK-DAG: tessera.rocm.schedule_raster_group = 4 : i64
// CHECK-DAG: tessera.rocm.schedule_raster_order = "grouped_m"
// CHECK: arith.divui
// CHECK: tessera_rocm.wmma
