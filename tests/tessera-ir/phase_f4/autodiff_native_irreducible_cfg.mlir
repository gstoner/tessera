// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s
//
// A two-entry SCC is irreducible: entry may jump to either bb1 or bb2, while
// bb1/bb2 form a cycle.  W4 lowers it to a bounded program-counter state
// machine and differentiates the resulting canonical SCF instead of guessing
// a loop header or rejecting all irreducible control flow.

module {
  func.func @irreducible(%enter_left: i1, %x: tensor<4xf32>)
      -> tensor<4xf32> attributes {tessera.autodiff = "reverse"} {
    %out = scf.execute_region -> tensor<4xf32> {
      %c0 = arith.constant 0 : index
      cf.cond_br %enter_left, ^bb1(%c0, %x : index, tensor<4xf32>),
                              ^bb2(%c0, %x : index, tensor<4xf32>)
    ^bb1(%i: index, %state: tensor<4xf32>):
      %next = "tessera.tanh"(%state) :
          (tensor<4xf32>) -> tensor<4xf32>
      cf.br ^bb2(%i, %next : index, tensor<4xf32>)
    ^bb2(%j: index, %right_state: tensor<4xf32>):
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %next_i = arith.addi %j, %c1 : index
      %continue = arith.cmpi slt, %next_i, %c2 : index
      cf.cond_br %continue, ^bb1(%next_i, %right_state : index, tensor<4xf32>),
                                  ^bb3(%right_state : tensor<4xf32>)
    ^bb3(%result: tensor<4xf32>):
      scf.yield %result : tensor<4xf32>
    } {tessera.structured_cfg.digest = "3333333333333333333333333333333333333333333333333333333333333333",
       tessera.structured_cfg.max_steps = 8 : i64}
    return %out : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @irreducible(
// CHECK-NOT: scf.execute_region
// CHECK: scf.for
// CHECK: } {
// CHECK-SAME: tessera.autodiff.native_multiblock_structurized = true
// CHECK-SAME: tessera.structured_cfg.execution = "bounded_state_machine_v1"
// CHECK: cf.assert
// CHECK-LABEL: func.func @irreducible__bwd(
// CHECK: scf.for
// CHECK: return
