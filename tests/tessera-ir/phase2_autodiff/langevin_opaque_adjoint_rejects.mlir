// RUN: not tessera-opt --tessera-ebm-lower-langevin %s 2>&1 | FileCheck %s
// REQUIRES: tessera-ebm
//
// An energy whose adjoint is a host callback. The gradient has to be the
// compiler's all the way down: an opaque adjoint means a host round trip per
// step, and this lane — one compiled loop, one launch — does not have one.
//
// This was documented as refused from the first slice and was not checked. On
// the CPU the JIT failed later with a pipeline error that named nothing, and on
// the device route the row-program emitter refused an op it could not place. The
// refusal now happens here, where both the energy and the callback can be named.

// CHECK: is not the compiler's
// CHECK-SAME: calls out to the host (user_energy_vjp)
module {
  func.func private @Ecb(%y: tensor<4x8xf32>, %x: tensor<4x8xf32>) -> tensor<4xf32>
  func.func @Ecb__bwd(%y: tensor<4x8xf32>, %x: tensor<4x8xf32>, %cot: tensor<4xf32>)
      -> (tensor<4x8xf32>, tensor<4x8xf32>) {
    %g = "tessera.custom_adjoint_call"(%y) { name = "user_energy_vjp" }
        : (tensor<4x8xf32>) -> tensor<4x8xf32>
    return %g, %g : tensor<4x8xf32>, tensor<4x8xf32>
  }
  func.func @opaque_adjoint(%y: tensor<4x8xf32>, %x: tensor<4x8xf32>, %k: tensor<2xi64>) -> tensor<4x8xf32> {
    %n:2 = "tessera_ebm.langevin_step"(%y, %k, %x) {
        operandSegmentSizes = array<i32: 1, 1, 0, 1>,
        energy_fn = @Ecb, eta = 1.000000e-01 : f64, temperature = 5.000000e-01 : f64,
        manifold = "euclidean"
    } : (tensor<4x8xf32>, tensor<2xi64>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<2xi64>)
    return %n#0 : tensor<4x8xf32>
  }
}
