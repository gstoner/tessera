// RUN: not %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1151})' %s 2>&1 | FileCheck %s

module {
  llvm.func @wrong_workgroup(%input: !llvm.ptr, %output: !llvm.ptr) {
    %batch = llvm.mlir.constant(1 : i64) : i64
    %size = llvm.mlir.constant(32 : i64) : i64
    tile.coalition_butterfly_kernel %input, %output, %batch, %size {
      accum = "f64", arch = "gfx1151", half = 1 : i64, players = 5 : i64,
      sign = 1 : i64, stage_order = "ascending_bit_yates_v1", storage = "f32",
      tessera.schedule_hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      workgroup_size = 256 : i64
    } : !llvm.ptr, !llvm.ptr, i64, i64
    llvm.return
  }
}

// CHECK: error: gfx1151 coalition transform requires the shared radix-2 Yates contract
