// RUN: tessera-opt --tessera-to-linalg %s | FileCheck %s
//
// `!=` on floats is the NEGATION of `==`, so it is TRUE whenever either operand
// is NaN. Every other comparison is ordered — false on NaN — but `ne` is not,
// and it was lowered to `arith.cmpf one` (ordered not-equal). That made
// `NaN != NaN` false and silently defeated `x != x`, the idiomatic NaN test,
// while matching numpy for every non-NaN input so nothing else looked wrong.
//
// numpy, which is this project's reference semantics:
//   np.not_equal(nan, nan) -> True      np.equal(nan, nan) -> False

func.func @float_ne(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xi1> {
  // CHECK: arith.cmpf une
  // CHECK-NOT: arith.cmpf one
  %0 = "tessera.ne"(%a, %b) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}

func.func @float_eq(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xi1> {
  // `eq` stays ORDERED: NaN == NaN is false, which oeq already gives.
  // CHECK: arith.cmpf oeq
  %0 = "tessera.eq"(%a, %b) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}

func.func @float_lt(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xi1> {
  // ...and so do the orderings.
  // CHECK: arith.cmpf olt
  %0 = "tessera.lt"(%a, %b) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}
