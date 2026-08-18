// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s
//
// W4.3: implicit SCF captures are returned by RegionAdjointInterface as
// explicit (primal, cotangent) pairs. The backward must replay only the taken
// branch and merge the captured %x cotangent with an scf.if.

module {
  func.func @branch(%pred: i1, %x: tensor<4xf32>) -> tensor<4xf32>
      attributes {tessera.autodiff = "reverse"} {
    %out = scf.if %pred -> tensor<4xf32> {
      %square = "tessera.mul"(%x, %x) :
          (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      scf.yield %square : tensor<4xf32>
    } else {
      %double = "tessera.add"(%x, %x) :
          (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      scf.yield %double : tensor<4xf32>
    }
    return %out : tensor<4xf32>
  }

  // The taken predicate is an explicit residual result, not an implicit
  // promise that backward can rediscover a data-dependent branch.
  // CHECK-LABEL: func.func @branch(
  // CHECK-SAME: -> (tensor<4xf32>, i1)
  // CHECK-SAME: tessera.autodiff.residual_sources = ["scf.if:predicate"]
  // CHECK: return {{.*}}, %arg0 : tensor<4xf32>, i1
  // CHECK-LABEL: func.func @branch__bwd(
  // CHECK-SAME: %[[PRED:[^:]+]]: i1)
  // CHECK-SAME: tessera.autodiff.residual_sources = ["scf.if:predicate"]
  // Recomputed primal branch followed by the branch-selecting pullback.
  // CHECK: %[[PRIMAL:.+]] = scf.if
  // CHECK: %[[DX:.+]] = scf.if %[[PRED]]
  // CHECK: tessera.mul
  // CHECK: arith.addf
  // The predicate remains nondifferentiable; %x receives the region capture.
  // CHECK: return {{.*}}, %[[DX]] : i1, tensor<4xf32>
}
