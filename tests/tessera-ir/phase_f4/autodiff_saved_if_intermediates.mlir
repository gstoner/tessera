// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s
//
// W4-PRODUCT-1: SAVE on scf.if carries the predicate plus each branch-local
// differentiable SSA value.  The pullback consumes only the taken branch's
// values; inactive-branch placeholders never participate in differentiation.

module {
  func.func @saved_branch(%pred: i1, %x: tensor<4xf32>) -> tensor<4xf32>
      attributes {tessera.autodiff = "reverse"} {
    %out = scf.if %pred -> tensor<4xf32> {
      %e = "tessera.tanh"(%x) : (tensor<4xf32>) -> tensor<4xf32>
      %square = "tessera.mul"(%e, %e) :
          (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      scf.yield %square : tensor<4xf32>
    } else {
      %s = "tessera.sigmoid"(%x) : (tensor<4xf32>) -> tensor<4xf32>
      %double = "tessera.add"(%s, %s) :
          (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      scf.yield %double : tensor<4xf32>
    } {tessera.autodiff.checkpoint_policy = "save"}
    return %out : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @saved_branch(
// CHECK-SAME: -> (tensor<4xf32>, i1, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-SAME: tessera.autodiff.residual_sources = ["scf.if:predicate", "scf.if:branch_value:0", "scf.if:branch_value:1", "scf.if:branch_value:2", "scf.if:branch_value:3"]
// CHECK: %[[IF:.+]]:5 = scf.if
// CHECK: tessera.autodiff.else_saved_count = 2
// CHECK: tessera.autodiff.residual_owner = "scf_if"
// CHECK: tessera.autodiff.residual_result_indices = array<i64: 1, 2, 3, 4>
// CHECK: tessera.autodiff.then_saved_count = 2
// CHECK: return %[[IF]]#0, %arg0, %[[IF]]#1, %[[IF]]#2, %[[IF]]#3, %[[IF]]#4
// CHECK-LABEL: func.func @saved_branch__bwd(
// CHECK-SAME: tessera.autodiff.residual_policy = "save"
// CHECK: scf.if %arg3
// CHECK: return
