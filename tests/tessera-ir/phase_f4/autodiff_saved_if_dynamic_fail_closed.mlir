// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s
//
// Dynamic branch-local residuals use initialized zero-extent sentinels in the
// inactive branch. Predicate replay ensures the sentinel is never consumed.

module {
  func.func @dynamic_branch(%pred: i1, %x: tensor<?xf32>) -> tensor<?xf32>
      attributes {tessera.autodiff = "reverse"} {
    %out = scf.if %pred -> tensor<?xf32> {
      %a = "tessera.tanh"(%x) : (tensor<?xf32>) -> tensor<?xf32>
      %square = "tessera.mul"(%a, %a) :
          (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
      scf.yield %square : tensor<?xf32>
    } else {
      %b = "tessera.sigmoid"(%x) : (tensor<?xf32>) -> tensor<?xf32>
      %double = "tessera.mul"(%b, %b) :
          (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
      scf.yield %double : tensor<?xf32>
    } {tessera.autodiff.checkpoint_policy = "save"}
    return %out : tensor<?xf32>
  }
}

// CHECK-LABEL: func.func @dynamic_branch(
// CHECK-SAME: -> (tensor<?xf32>, i1, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>)
// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[EMPTY:.+]] = tensor.empty(%[[C0]]) : tensor<?xf32>
// CHECK: %[[ZERO:.+]] = linalg.fill
// CHECK: scf.yield {{.*}}%[[ZERO]]{{.*}} : tensor<?xf32>
// CHECK-LABEL: func.func @dynamic_branch__bwd(
// CHECK: scf.if %arg3
// CHECK: tessera.mul %arg2, %arg4
// CHECK: tessera.mul %arg2, %arg6
// CHECK: return
