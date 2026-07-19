// RUN: %tnv --tessera-lower-to-nvidia-sm120 %s | FileCheck %s

module {
  llvm.func @paged(%pages: !llvm.ptr, %table: !llvm.ptr, %o: !llvm.ptr,
                   %p: i64, %lp: i64, %ps: i64, %h: i64, %d: i64,
                   %start: i64, %tokens: i64) attributes {nvvm.kernel} {
    tile.paged_kv_read_kernel %pages, %table, %o, %p, %lp, %ps, %h, %d, %start, %tokens {
      storage = "f32", table_storage = "i32", route = "direct"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @paged
// CHECK: nvvm.read.ptx.sreg.ctaid.x
// CHECK: llvm.load
// CHECK: llvm.store
// CHECK-NOT: tile.paged_kv_read_kernel
