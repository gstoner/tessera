// RUN: tessera-opt --tessera-control-flow-to-scf %s | FileCheck %s
//
// A boolean loop condition is the most natural form to write, and it used to
// crash the lowering: `extractPredicateI1` built `getFloatAttr` on the element
// type unconditionally, so a `tensor<i1>` condition asked an integer type for a
// float attribute. Only a `tensor<f32>` condition avoided it.
//
// The predicate type is now checked BEFORE any IR is created — a failed
// lowering is not rolled back, so refusing at the point of use would leave a
// half-built scf.while behind.

func.func @cond_i1(%c: tensor<4xf32>) -> tensor<i1> {
  %t = arith.constant dense<true> : tensor<i1>
  return %t : tensor<i1>
}
func.func @body_i1(%c: tensor<4xf32>) -> tensor<4xf32> { return %c : tensor<4xf32> }

// CHECK-LABEL: func.func @while_with_boolean_condition
// CHECK: scf.while
// CHECK-NOT: tessera.control_while
func.func @while_with_boolean_condition(%x: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "tessera.control_while"(%x)
      {cond = @cond_i1, body = @body_i1, max_iters = 4 : i64, carry_arg_index = 0 : i64}
      : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

func.func @cond_f32(%c: tensor<4xf32>) -> tensor<f32> {
  %t = arith.constant dense<1.0> : tensor<f32>
  return %t : tensor<f32>
}
func.func @body_f32(%c: tensor<4xf32>) -> tensor<4xf32> { return %c : tensor<4xf32> }

// The float form is unchanged.
// CHECK-LABEL: func.func @while_with_float_condition
// CHECK: scf.while
func.func @while_with_float_condition(%x: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "tessera.control_while"(%x)
      {cond = @cond_f32, body = @body_f32, max_iters = 4 : i64, carry_arg_index = 0 : i64}
      : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
