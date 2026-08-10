// RUN: tessera-opt %s --tessera-autodiff-forward | FileCheck %s

module {
  func.func @forward_core(
      %x: tensor<2x3xf32>, %w: tensor<3x4xf32>) -> tensor<2x4xf32>
      attributes {tessera.autodiff = "forward"} {
    %mm = tessera.matmul %x, %w :
      (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    %sq = tessera.mul %mm, %mm :
      (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    return %sq : tensor<2x4xf32>
  }

  func.func @barrier(%x: tensor<4xf32>) -> tensor<4xf32>
      attributes {tessera.autodiff = "forward"} {
    %y = tessera.stop_gradient %x : (tensor<4xf32>) -> tensor<4xf32>
    return %y : tensor<4xf32>
  }

  func.func @subset(
      %x: tensor<2x3xf32>, %w: tensor<3x4xf32>) -> tensor<2x4xf32>
      attributes {tessera.autodiff = "forward",
                  tessera.autodiff.wrt_indices = [1]} {
    %y = tessera.matmul %x, %w :
      (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    return %y : tensor<2x4xf32>
  }
}

// CHECK-LABEL: func.func @forward_core(
// CHECK-SAME: tessera.autodiff = "forward"
// CHECK-SAME: tessera.autodiff.jvp = @forward_core__jvp
// CHECK-LABEL: func.func private @forward_core__jvp(
// CHECK-SAME: %{{.*}}: tensor<2x3xf32>, %{{.*}}: tensor<3x4xf32>, %{{.*}}: tensor<2x3xf32>, %{{.*}}: tensor<3x4xf32>
// CHECK-SAME: -> (tensor<2x4xf32>, tensor<2x4xf32>)
// CHECK-SAME: tessera.autodiff.forward = @forward_core
// CHECK-SAME: tessera.autodiff.role = "jvp"
// CHECK: %[[P:.*]] = tessera.matmul
// CHECK: %[[DL:.*]] = tessera.matmul
// CHECK: %[[DR:.*]] = tessera.matmul
// CHECK: %[[DMM:.*]] = tessera.add %[[DL]], %[[DR]]
// CHECK: %[[SQ:.*]] = tessera.mul %[[P]], %[[P]]
// CHECK: %[[ML:.*]] = tessera.mul %[[DMM]], %[[P]]
// CHECK: %[[MR:.*]] = tessera.mul %[[P]], %[[DMM]]
// CHECK: %[[DSQ:.*]] = tessera.add %[[ML]], %[[MR]]
// CHECK: return %[[SQ]], %[[DSQ]]
// CHECK-LABEL: func.func private @barrier__jvp(
// CHECK: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<4xf32>
// CHECK: return %{{.*}}, %[[ZERO]]
// CHECK-LABEL: func.func private @subset__jvp(
// CHECK-SAME: %{{.*}}: tensor<2x3xf32>, %{{.*}}: tensor<3x4xf32>, %{{.*}}: tensor<3x4xf32>
// CHECK-SAME: -> (tensor<2x4xf32>, tensor<2x4xf32>)
// CHECK: %[[SUBP:.*]] = tessera.matmul %{{.*}}, %{{.*}}
// CHECK: %[[SUBD:.*]] = tessera.matmul %{{.*}}, %{{.*}}
// CHECK-NEXT: return %[[SUBP]], %[[SUBD]]
