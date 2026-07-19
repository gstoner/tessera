// RUN: %tnv --tessera-lower-to-nvidia-sm120 %s | FileCheck %s

// General-shape launch-level block-scaled matmul. Packed E2M1 A/B and logical
// UE4M3 scale views are explicit ABI operands; the lowering owns M16/N8 grid
// origins, K64 accumulation, ragged zero fill, and guarded f32 stores.
module {
  llvm.func @tile_matmul_nvfp4(
      %a: !llvm.ptr, %b: !llvm.ptr, %scale_a: !llvm.ptr,
      %scale_b: !llvm.ptr, %d: !llvm.ptr,
      %m: i64, %n: i64, %k: i64) attributes {nvvm.kernel} {
    tile.matmul_kernel %a, %b, %scale_a, %scale_b, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 64, a = "nvfp4", b = "nvfp4", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">,
      warps = 1 : i64, staging = "global"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @tile_matmul_nvfp4
// CHECK: nvvm.read.ptx.sreg.ctaid.x
// CHECK: nvvm.read.ptx.sreg.ctaid.y
// CHECK: scf.for
// CHECK: llvm.intr.masked.load
// CHECK: llvm.inline_asm
// CHECK-SAME: mxf4nvf4.block_scale
// CHECK: llvm.intr.masked.store
// CHECK-NOT: tile.matmul_kernel
