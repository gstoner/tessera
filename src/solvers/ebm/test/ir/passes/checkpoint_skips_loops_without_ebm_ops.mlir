// RUN: ts-ebm-opt --tessera-ebm-checkpoint-inner-loop %s | FileCheck %s
//
// Loops that don't contain any ebm step ops are left untouched —
// only loops actually running an ebm inner-step / langevin-step get
// the checkpoint annotations.

module {
  func.func @plain_loop(%n : index) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %result = scf.for %i = %c0 to %n step %c1
        iter_args(%acc = %c0) -> index {
      %next = arith.addi %acc, %c1 : index
      scf.yield %next : index
    }
    return %result : index
  }
}

// CHECK: scf.for
// CHECK-NOT: tessera.ebm.checkpoint_loop
// CHECK-NOT: tessera.ebm.checkpoint_budget
