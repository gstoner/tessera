// RUN: not %tnv %s 2>&1 | FileCheck %s

module {
  llvm.func @bad_macro(%a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr,
                       %m: i64, %n: i64, %k: i64) attributes {nvvm.kernel} {
    "tessera_nvidia.macro_cta_matmul"(%a, %b, %d, %m, %n, %k) {
      arch = "sm_120", cta_m = 64 : i64, cta_n = 32 : i64,
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

// CHECK: error: 'tessera_nvidia.macro_cta_matmul' op requires a 32x32 CTA tile
