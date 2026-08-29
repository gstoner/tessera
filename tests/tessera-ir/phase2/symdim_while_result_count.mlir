// RUN: tessera-opt --tessera-symdim-equality %s | FileCheck %s
//
// An `scf.while`'s RESULT count is what its condition forwards, which need not
// equal its init/yield count. Here two values are carried and yielded but the
// condition forwards one, so the loop has one result. The pass seeded result
// dim-names from inside the loop over YIELD operands, so it indexed
// `whileOp.getResult(1)` on a one-result op — out of bounds on IR that is
// perfectly valid.

// CHECK-LABEL: func.func @while_yields_more_than_it_returns
// CHECK: scf.while
func.func @while_yields_more_than_it_returns(
    %x: tensor<4xf32>, %y: tensor<4xf32>) -> tensor<4xf32> {
  %r = scf.while (%a = %x, %b = %y)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> {
    %c = arith.constant true
    scf.condition(%c) %a : tensor<4xf32>
  } do {
  ^bb0(%arg: tensor<4xf32>):
    scf.yield %arg, %arg : tensor<4xf32>, tensor<4xf32>
  }
  return %r : tensor<4xf32>
}

// A condition that forwards a NON-PREFIX slice of the carried state. Result 0
// is %b (init position 1), so seeding it from position 0 would attach %x's
// dim-names to %y's value — and the positional mismatch check would compare
// two unrelated sequences and could reject a valid loop. Results are seeded
// from the condition's forwarded values, and the check only fires where the
// forward really is positional.

// CHECK-LABEL: func.func @while_forwards_non_prefix_state
// CHECK: scf.while
func.func @while_forwards_non_prefix_state(
    %x: tensor<4xf32>, %y: tensor<8xf32>) -> tensor<8xf32> {
  %r = scf.while (%a = %x, %b = %y)
      : (tensor<4xf32>, tensor<8xf32>) -> tensor<8xf32> {
    %c = arith.constant true
    scf.condition(%c) %b : tensor<8xf32>
  } do {
  ^bb0(%arg: tensor<8xf32>):
    scf.yield %x, %arg : tensor<4xf32>, tensor<8xf32>
  }
  return %r : tensor<8xf32>
}
