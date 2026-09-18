// RUN: %trop --allow-unregistered-dialect --generate-wmma-gemm-kernel='via-tile=true' --lower-tile-to-rocm='arch=gfx1201' %s | FileCheck %s
// RUN: not %trop --allow-unregistered-dialect --generate-wmma-gemm-kernel='via-tile=true' --lower-tile-to-rocm='arch=gfx1151' %s 2>&1 | FileCheck %s --check-prefix=GFX11
// RUN: not %trop --allow-unregistered-dialect --generate-wmma-gemm-kernel='via-tile=false' %s 2>&1 | FileCheck %s --check-prefix=DIRECT
//
// OCP FP8 storage on the TYPED Tile route (GFX1201-PARITY slice 5). The RDNA4
// WMMA datatype audit (2026-09-13) proved V_WMMA_F32_16X16X16_FP8_FP8 on the
// box; this is the first kernel-shaped consumer: `tile.matmul_kernel` with an
// e4m3/e4m3 -> f32 descriptor reaches the generator's typed body, whose
// fragments TileToROCM packs per chip (rdna4_wmma, vector<2xi32> per lane).
// gfx11 has no FP8 WMMA and refuses by name; so does the direct (untyped)
// gfx11 body.

module {
  func.func @fp8(%a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr,
                 %m: i64, %n: i64, %k: i64) {
    tile.matmul_kernel %a, %b, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16, a = "e4m3", b = "e4m3", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">,
      warps = 1 : i64, staging = "global",
      tessera.macro_tile_m = 16 : i64, tessera.macro_tile_n = 16 : i64
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    return
  }
}

// CHECK-NOT: tile.matmul_kernel
// CHECK: gpu.func @fp8(%{{.*}}: memref<?xf8E4M3FN>, %{{.*}}: memref<?xf8E4M3FN>, %{{.*}}: memref<?xf32>
// CHECK: tessera_rocm.wmma
// CHECK-SAME: fragment_family = "rdna4_wmma"
// CHECK: memref.store

// GFX11: OCP FP8 e4m3/e5m2 to f32 on gfx12
// DIRECT: FP8 storage ('e4m3') is a typed-route contract
