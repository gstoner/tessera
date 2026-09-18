// RUN: %trop --allow-unregistered-dialect --generate-wmma-gemm-kernel='via-tile=true' --lower-tile-to-rocm='arch=gfx1151' %s | FileCheck %s --check-prefixes=CHECK,GFX11
// RUN: %trop --allow-unregistered-dialect --generate-wmma-gemm-kernel='via-tile=true' --lower-tile-to-rocm='arch=gfx1201' %s | FileCheck %s --check-prefixes=CHECK,GFX12
//
// Integer storage on the TYPED Tile route (GFX1201-PARITY slice 1b,
// 2026-09-18). `tile.matmul_kernel` with an int8/int8 -> i32 or int4/int4 ->
// i32 descriptor reaches the generator's typed body; TileToROCM packs the
// fragment per chip: gfx11 (rdna3_wmma) bitcasts 16 int8 lanes to
// vector<4xi32> and compacts 16 int4 nibbles to vector<2xi32>; gfx12
// (rdna4_wmma) packs 8 int8 to vector<2xi32> and 8 int4 nibbles to one i32
// word. Both lower to V_WMMA_I32_16X16X16_IU8 / IU4 through the same ROCDL
// selection the directive lane uses. int4 values ride i8 containers, one
// logical value per byte.

module {
  func.func @int8(%a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr,
                  %m: i64, %n: i64, %k: i64) {
    tile.matmul_kernel %a, %b, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16, a = "int8", b = "int8", acc = "i32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "i32">,
      warps = 1 : i64, staging = "global",
      tessera.macro_tile_m = 16 : i64, tessera.macro_tile_n = 16 : i64
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    return
  }
  func.func @int4(%a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr,
                  %m: i64, %n: i64, %k: i64) {
    tile.matmul_kernel %a, %b, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16, a = "int4", b = "int4", acc = "i32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "i32">,
      warps = 1 : i64, staging = "global",
      tessera.macro_tile_m = 16 : i64, tessera.macro_tile_n = 16 : i64
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    return
  }
}

// CHECK-NOT: tile.matmul_kernel
// CHECK: gpu.func @int8(%{{.*}}: memref<?xi8>, %{{.*}}: memref<?xi8>, %{{.*}}: memref<?xi32>
// CHECK: tessera_rocm.wmma
// GFX11-SAME: fragment_family = "rdna3_wmma"
// GFX12-SAME: fragment_family = "rdna4_wmma"
// CHECK-SAME: input_dtype = "int8"
// CHECK: gpu.func @int4(%{{.*}}: memref<?xi8>, %{{.*}}: memref<?xi8>, %{{.*}}: memref<?xi32>
// CHECK: tessera_rocm.wmma
// GFX11-SAME: fragment_family = "rdna3_wmma"
// GFX12-SAME: fragment_family = "rdna4_wmma"
// CHECK-SAME: input_dtype = "int4"
