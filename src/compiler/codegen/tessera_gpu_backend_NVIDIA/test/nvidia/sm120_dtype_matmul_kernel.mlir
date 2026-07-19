// RUN: %tnv --lower-tile-to-nvidia='sm=120' --lower-tessera-nvidia-to-nvvm %s | FileCheck %s

module {
  llvm.func @tile_f64(%a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr,
                      %m: i64, %n: i64, %k: i64) attributes {nvvm.kernel} {
    tile.matmul_kernel %a, %b, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "mma_sync", m = 8, n = 8, k = 4, a = "f64", b = "f64", acc = "f64", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "f64">,
      warps = 1 : i64, staging = "global"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }

  llvm.func @tile_tf32(%a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr,
                       %m: i64, %n: i64, %k: i64) attributes {nvvm.kernel} {
    tile.matmul_kernel %a, %b, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 8, a = "tf32", b = "tf32", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">,
      warps = 1 : i64, staging = "global"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }

  llvm.func @tile_e4m3(%a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr,
                       %m: i64, %n: i64, %k: i64) attributes {nvvm.kernel} {
    tile.matmul_kernel %a, %b, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 32, a = "e4m3", b = "e4m3", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">,
      warps = 1 : i64, staging = "global"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }

  llvm.func @tile_s8(%a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr,
                     %m: i64, %n: i64, %k: i64) attributes {nvvm.kernel} {
    tile.matmul_kernel %a, %b, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 32, a = "s8", b = "s8", acc = "s32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "i32">,
      warps = 1 : i64, staging = "global"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @tile_f64
// CHECK: nvvm.mma.sync
// CHECK: llvm.intr.masked.store
// CHECK-LABEL: llvm.func @tile_tf32
// CHECK: nvvm.mma.sync
// CHECK-LABEL: llvm.func @tile_e4m3
// CHECK: mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
// CHECK-LABEL: llvm.func @tile_s8
// CHECK: nvvm.mma.sync
// CHECK: llvm.intr.masked.store
