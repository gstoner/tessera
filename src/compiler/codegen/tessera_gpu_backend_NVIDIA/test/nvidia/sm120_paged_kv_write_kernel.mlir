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
// CHECK: arith.cmpi ult
// CHECK: scf.if
// The logical position is read from device memory and checked as signed.
// CHECK: llvm.load %arg3 {{.*}} : !llvm.ptr -> i32
// CHECK: arith.cmpi sge
// CHECK: scf.if
// CHECK: llvm.load {{.*}} : !llvm.ptr -> i32
// CHECK: arith.cmpi ult, %{{.*}}, %arg5 : i64
// CHECK: scf.if
// CHECK: llvm.load {{.*}} : !llvm.ptr -> f32
// CHECK: llvm.store {{.*}} : f32, !llvm.ptr
// Physical page out of range, then position out of range: both set status.
// CHECK: llvm.store %{{.*}}, %arg4 {{.*}} : i32, !llvm.ptr
// CHECK: llvm.store %{{.*}}, %arg4 {{.*}} : i32, !llvm.ptr
// CHECK-NOT: tile.paged_kv_write_kernel
