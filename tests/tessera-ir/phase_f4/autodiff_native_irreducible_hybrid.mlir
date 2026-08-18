// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s
//
// HYBRID retention for an irreducible CFG tapes only differentiable tensor
// slots.  Program-counter and completion state remain explicit primal SSA and
// are replayed from the retained tensor checkpoints.

module {
  func.func @irreducible_hybrid(%enter_left: i1, %x: tensor<4xf32>)
      -> tensor<4xf32> attributes {tessera.autodiff = "reverse"} {
    %out = scf.execute_region -> tensor<4xf32> {
      %c0 = arith.constant 0 : index
      cf.cond_br %enter_left, ^bb1(%c0, %x : index, tensor<4xf32>),
                              ^bb2(%c0, %x : index, tensor<4xf32>)
    ^bb1(%i: index, %state: tensor<4xf32>):
      %next = scf.if %enter_left -> tensor<4xf32> {
        %then = "tessera.tanh"(%state) :
            (tensor<4xf32>) -> tensor<4xf32>
        scf.yield %then : tensor<4xf32>
      } else {
        %else = "tessera.tanh"(%state) :
            (tensor<4xf32>) -> tensor<4xf32>
        scf.yield %else : tensor<4xf32>
      }
      cf.br ^bb2(%i, %next : index, tensor<4xf32>)
    ^bb2(%j: index, %right_state: tensor<4xf32>):
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %next_i = arith.addi %j, %c1 : index
      %continue = arith.cmpi slt, %next_i, %c2 : index
      cf.cond_br %continue,
          ^bb1(%next_i, %right_state : index, tensor<4xf32>),
          ^exit(%right_state : tensor<4xf32>)
    ^exit(%result: tensor<4xf32>):
      scf.yield %result : tensor<4xf32>
    } {tessera.autodiff.checkpoint_indices = array<i64: 2, 4, 6>,
       tessera.autodiff.checkpoint_policy = "hybrid",
       tessera.structured_cfg.digest = "5555555555555555555555555555555555555555555555555555555555555555",
       tessera.structured_cfg.max_steps = 8 : i64}
    return %out : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @irreducible_hybrid(
// CHECK: scf.for
// CHECK: } {
// CHECK-SAME: tessera.autodiff.residual_owner = "generic_for"
// CHECK-SAME: tessera.autodiff.residual_primal_iter_arg_indices = array<i64: 0, 1, 2, 3, 4, 5, 6, 7>
// CHECK-SAME: tessera.autodiff.residual_result_indices = array<i64: 8, 9, 10, 11, 12, 13, 14, 15>
// CHECK-SAME: tessera.structured_cfg.execution = "bounded_state_machine_v1"
// CHECK-LABEL: func.func @irreducible_hybrid__bwd(
// CHECK: scf.for
// CHECK: tensor.extract {{.*}} : tensor<3xindex>
// CHECK: tensor.extract_slice
// CHECK: return
