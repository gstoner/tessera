// RUN: %trop %s --pass-pipeline='builtin.module(gpu.module(rocm-materialize-dynamic-lds))' | FileCheck %s

module {
  gpu.module @kernels {
    // Sequential non-overlapping lifetimes now share one interference slot.
    // CHECK-LABEL: llvm.func @sequential_dynamic_lds
    // CHECK-SAME: tessera.rocm.dynamic_lds_launch_reduction = "aligned_sum_of_slot_maxima"
    // CHECK-SAME: tessera.rocm.dynamic_lds_slots
    // CHECK-COUNT-2: llvm.getelementptr
    // CHECK-NOT: llvm.alloca
    llvm.func @sequential_dynamic_lds(%lhs: i64, %rhs: i64)
        attributes {gpu.kernel, rocdl.kernel} {
      llvm.br ^later
    ^later:
      %a = llvm.alloca %lhs x i8 : (i64) -> !llvm.ptr<3>
      llvm.br ^final
    ^final:
      %b = llvm.alloca %rhs x i8 : (i64) -> !llvm.ptr<3>
      llvm.return
    }

    // A pointer forwarded through a block argument remains an analyzable SSA
    // alias. Once its final load has completed, a later arena can reuse the
    // same slot; forwarding alone is not an escape.
    // CHECK-LABEL: llvm.func @forwarded_alias_dynamic_lds
    // CHECK-SAME: tessera.rocm.dynamic_lds_slots
    // CHECK-SAME: slot = 0
    // CHECK-NOT: slot = 1
    // CHECK: llvm.br ^bb1
    // CHECK: llvm.load
    // CHECK: llvm.getelementptr
    // CHECK-NOT: llvm.alloca
    llvm.func @forwarded_alias_dynamic_lds(%lhs: i64, %rhs: i64)
        attributes {gpu.kernel, rocdl.kernel} {
      %a = llvm.alloca %lhs x i8 : (i64) -> !llvm.ptr<3>
      llvm.br ^bb1(%a : !llvm.ptr<3>)
    ^bb1(%alias: !llvm.ptr<3>):
      %observed = llvm.load %alias : !llvm.ptr<3> -> i8
      llvm.br ^bb2
    ^bb2:
      %b = llvm.alloca %rhs x i8 : (i64) -> !llvm.ptr<3>
      llvm.return
    }
  }
}
