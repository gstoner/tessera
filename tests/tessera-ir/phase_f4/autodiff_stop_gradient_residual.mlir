// RUN: not tessera-opt --tessera-autodiff-paired %s 2>&1 | FileCheck %s

// The paired pass advertises recompute_all.  Replaying an inactive producer
// behind stop_gradient could repeat state or random effects, so an intermediate
// barrier fails closed until a saved-residual policy exists.

module {
  func.func @stopped_intermediate(%x: tensor<4xf32>) -> tensor<4xf32>
      attributes {tessera.autodiff = "reverse"} {
    %p = "tessera.relu"(%x) : (tensor<4xf32>) -> tensor<4xf32>
    %s = "tessera.stop_gradient"(%p)
        : (tensor<4xf32>) -> tensor<4xf32>
    // CHECK: AUTODIFF_STOP_GRADIENT_RESIDUAL_REQUIRED:
    return %s : tensor<4xf32>
  }
}
