// RUN: %tnv --tessera-lower-to-nvidia-sm120 %s | FileCheck %s
//
// AR-DECODE-PROGRAM-1 S1a: the in-place paged write lowers to one guarded
// store per value element; the status word is written only on refusal.

module {
  llvm.func @paged_write(%pages: !llvm.ptr, %table: !llvm.ptr, %values: !llvm.ptr,
                         %position: !llvm.ptr, %status: !llvm.ptr,
                         %p: i64, %lp: i64, %ps: i64, %h: i64, %d: i64,
                         %tokens: i64) attributes {nvvm.kernel} {
    tile.paged_kv_write_kernel %pages, %table, %values, %position, %status, %p, %lp, %ps, %h, %d, %tokens {
      storage = "f32", table_storage = "i32", route = "direct", page_ownership = "in_place"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @paged_write
// CHECK: nvvm.read.ptx.sreg.ctaid.x
// CHECK: llvm.load %{{.*}} : !llvm.ptr -> i32
// CHECK: llvm.icmp "sge"
// CHECK: llvm.store %{{.*}}, %{{.*}} : f32, !llvm.ptr
// CHECK: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr
// CHECK-NOT: tile.paged_kv_write_kernel
