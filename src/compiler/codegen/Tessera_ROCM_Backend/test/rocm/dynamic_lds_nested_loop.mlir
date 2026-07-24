// RUN: %trop %s --pass-pipeline='builtin.module(gpu.module(rocm-materialize-dynamic-lds))' | FileCheck %s

module {
  gpu.module @kernels {
    // The outer arena is live through the loop and therefore interferes with
    // loop-local storage. The loop-local and nested branch-local arenas have
    // disjoint pointer lifetimes and reuse the second slot.
    // CHECK-LABEL: llvm.func @nested_loop_dynamic_lds
    // CHECK-SAME: tessera.rocm.dynamic_lds_launch_reduction = "aligned_sum_of_slot_maxima"
    // CHECK-SAME: tessera.rocm.dynamic_lds_slots
    // CHECK-SAME: slot = 0
    // CHECK-SAME: slot = 1
    // CHECK-COUNT-4: llvm.getelementptr
    // CHECK-NOT: llvm.alloca
    llvm.func @nested_loop_dynamic_lds(
        %outer_bytes: i64, %loop_bytes: i64, %then_bytes: i64,
        %else_bytes: i64, %choose: i1, %iterations: i64)
        attributes {gpu.kernel, rocdl.kernel} {
      %outer = llvm.alloca %outer_bytes x i8 : (i64) -> !llvm.ptr<3>
      %zero = llvm.mlir.constant(0 : i64) : i64
      %outer_value = llvm.mlir.constant(3 : i8) : i8
      llvm.store %outer_value, %outer : i8, !llvm.ptr<3>
      llvm.br ^loop(%zero, %loop_bytes : i64, i64)
    ^loop(%iv: i64, %loop_size: i64):
      %local = llvm.alloca %loop_size x i8 : (i64) -> !llvm.ptr<3>
      %local_value = llvm.mlir.constant(4 : i8) : i8
      llvm.store %local_value, %local : i8, !llvm.ptr<3>
      %local_observed = llvm.load %local : !llvm.ptr<3> -> i8
      llvm.cond_br %choose, ^then(%then_bytes : i64),
          ^else(%else_bytes : i64)
    ^then(%then_size: i64):
      %then_arena = llvm.alloca %then_size x i8 : (i64) -> !llvm.ptr<3>
      %then_value = llvm.mlir.constant(5 : i8) : i8
      llvm.store %then_value, %then_arena : i8, !llvm.ptr<3>
      %then_observed = llvm.load %then_arena : !llvm.ptr<3> -> i8
      llvm.br ^latch
    ^else(%else_size: i64):
      %else_arena = llvm.alloca %else_size x i8 : (i64) -> !llvm.ptr<3>
      %else_value = llvm.mlir.constant(7 : i8) : i8
      llvm.store %else_value, %else_arena : i8, !llvm.ptr<3>
      %else_observed = llvm.load %else_arena : !llvm.ptr<3> -> i8
      llvm.br ^latch
    ^latch:
      %one = llvm.mlir.constant(1 : i64) : i64
      %next = llvm.add %iv, %one : i64
      %again = llvm.icmp "ult" %next, %iterations : i64
      llvm.cond_br %again, ^loop(%next, %loop_size : i64, i64), ^exit
    ^exit:
      %outer_observed = llvm.load %outer : !llvm.ptr<3> -> i8
      llvm.return
    }
  }
}
