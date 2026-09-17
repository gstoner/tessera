// RUN: %trop --allow-unregistered-dialect --generate-wmma-gemm-kernel='via-tile=true' --lower-tile-to-rocm='arch=gfx1201' %s | FileCheck %s --check-prefixes=CHECK,GFX12
// RUN: %trop --allow-unregistered-dialect --generate-wmma-gemm-kernel='via-tile=true' --lower-tile-to-rocm='arch=gfx1151' %s | FileCheck %s --check-prefixes=CHECK,GFX11
//
// The fused bias + activation epilogue on the TYPED Tile route. The generator's
// typed body used to refuse any epilogue ("typed via-tile pilot requires an
// unfused ... GEMM"), which left the fused epilogue gfx11-only: the untyped
// body applies it in its own element loop, and only the typed body reaches
// TileToROCM's per-architecture fragment layouts. Since 2026-09-17 the typed
// body hands the epilogue to the store (`tile.epilogue` + trailing bias
// operand) and TileToROCM applies it per element after resolving the row and
// column for the chip's own accumulator map -- so one implementation is right
// on gfx11's replicated rows and RDNA4's half-wave rows alike. Both RUN lines
// must show the bias load, the add and the activation next to the store.

module {
  func.func @fused(%a: !llvm.ptr, %b: !llvm.ptr, %bias: !llvm.ptr, %d: !llvm.ptr,
                   %m: i64, %n: i64, %k: i64) {
    tile.matmul_kernel %a, %b, %bias, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = true, activation = "relu", output = "f32">,
      warps = 1 : i64, staging = "global",
      tessera.macro_tile_m = 16 : i64, tessera.macro_tile_n = 16 : i64
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    return
  }
}

// CHECK-NOT: tile.matmul_kernel
// CHECK-NOT: tile.store
// CHECK: gpu.func @fused(%{{.*}}: memref<?xf16>, %{{.*}}: memref<?xf16>, %[[BIAS:.*]]: memref<?xf32>, %{{.*}}: memref<?xf32>
// GFX12: fragment_family = "rdna4_wmma"
// GFX11: fragment_family = "gfx11_wmma"
// CHECK: memref.load %[[BIAS]][
// CHECK: arith.addf
// CHECK: arith.maximumf
// CHECK: memref.store
