// RUN: tessera-opt %s --tessera-autodiff-hvp-pipeline | FileCheck %s
//
// AD-HIGHER-1: the Hessian product is the exact JVP of the compiler-emitted
// paired reverse program.  There is no epsilon, finite-difference helper, or
// second execution of the Python function in this pipeline.

module {
  func.func @square(%x: tensor<4xf32>) -> tensor<4xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = tessera.mul %x, %x :
        (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %y : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @square(
// CHECK-SAME: tessera.autodiff.hvp = @square__bwd__jvp
// CHECK-SAME: tessera.autodiff.paired = @square__bwd

// The reverse takes (x, dy) and returns dx.  Only x is tangent-active; dy is
// the fixed cotangent seed, which is the forward-over-reverse HVP contract.
// CHECK-LABEL: func.func @square__bwd(
// CHECK-SAME: tessera.autodiff = "forward"
// CHECK-SAME: tessera.autodiff.hvp_parent = @square
// CHECK-SAME: tessera.autodiff.jvp = @square__bwd__jvp
// CHECK-SAME: tessera.autodiff.wrt_indices = [0]

// CHECK-LABEL: func.func private @square__bwd__jvp(
// CHECK-SAME: %{{.*}}: tensor<4xf32>, %{{.*}}: tensor<4xf32>, %{{.*}}: tensor<4xf32>
// CHECK-SAME: -> (tensor<4xf32>, tensor<4xf32>)
// CHECK-SAME: tessera.autodiff.forward = @square__bwd
// CHECK-SAME: tessera.autodiff.role = "jvp"
// CHECK: tessera.mul
// CHECK: tessera.add
// CHECK: return {{.*}} : tensor<4xf32>, tensor<4xf32>
