// RUN: tessera-opt --tessera-autodiff-paired --allow-unregistered-dialect %s | FileCheck %s

// AD-CORE-EFFECT-CONTROL-1: activity is SSA-backward reachability.  A stopped
// input remains a forward identity, is erased before lowering, and returns a
// zero input cotangent from the paired backward function.  An unrelated nested
// region is inactive and therefore legal.

module {
  func.func @stopped_input(%x: tensor<4xf32>) -> tensor<4xf32>
      attributes {tessera.autodiff = "reverse"} {
    "test.inactive_region"() ({
      "test.terminator"() : () -> ()
    }) : () -> ()
    %s = "tessera.stop_gradient"(%x)
        : (tensor<4xf32>) -> tensor<4xf32>
    return %s : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @stopped_input(
// CHECK: "test.inactive_region"() ({
// CHECK: }) {tessera.autodiff.activity = "inactive"}
// CHECK-NOT: tessera.stop_gradient
// CHECK-LABEL: func.func @stopped_input__bwd(
// CHECK: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<4xf32>
// CHECK: return %[[ZERO]] : tensor<4xf32>
// CHECK-NOT: tessera.stop_gradient
