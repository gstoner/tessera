// RUN: tessera-opt --allow-unregistered-dialect --tessera-newton-autodiff='generate-jvp=true' %s | FileCheck %s

module {
  func.func private @sqrt_residual(%theta: tensor<4xf32>, %x: tensor<4xf32>)
      -> tensor<4xf32> {
    %xx = arith.mulf %x, %x : tensor<4xf32>
    %r = arith.subf %xx, %theta : tensor<4xf32>
    return %r : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @solve
  // CHECK: "tessera_solver.implicit"
  // CHECK-SAME: tessera.solver.autodiff_ready
  // CHECK-SAME: tessera.solver.ift_contract_version = 1
  // CHECK-SAME: tessera.solver.jvp = @solve__implicit_0_jvp
  // CHECK-SAME: tessera.solver.vjp = @solve__implicit_0_vjp
  func.func @solve(%theta: tensor<4xf32>) -> tensor<4xf32> {
    %x = "tessera_solver.implicit"(%theta) {
      residual = @sqrt_residual
    } : (tensor<4xf32>) -> tensor<4xf32>
    return %x : tensor<4xf32>
  }

  // CHECK-LABEL: func.func private @solve__implicit_0_vjp
  // CHECK-SAME: (%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<4xf32>
  // CHECK: %[[R:.*]] = "tessera_solver.residual"(%arg0, %arg1) <{callee = @sqrt_residual}> : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // CHECK: %[[LAMBDA:.*]] = "tessera_solver.linear_solve"(%arg0, %arg1, %[[R]], %arg2)
  // CHECK-SAME: linearization = "solution"
  // CHECK-SAME: residual = @sqrt_residual
  // CHECK-SAME: transpose = true
  // CHECK-SAME: algorithm = "gmres"
  // CHECK-SAME: matrix_free = true
  // CHECK-SAME: max_iterations = 256
  // CHECK-SAME: restart = 32
  // CHECK: %[[DTHETA:.*]] = "tessera_solver.residual_adjoint"(%arg0, %arg1, %[[LAMBDA]])
  // CHECK-SAME: residual = @sqrt_residual
  // CHECK-SAME: scale = -1.000000e+00
  // CHECK-SAME: wrt = "parameters"
  // CHECK: return %[[DTHETA]] : tensor<4xf32>

  // CHECK-LABEL: func.func private @solve__implicit_0_jvp
  // CHECK: %[[JR:.*]] = "tessera_solver.residual"(%arg0, %arg1)
  // CHECK: %[[RHS:.*]] = "tessera_solver.residual_jvp"(%arg0, %arg1, %arg2)
  // CHECK-SAME: wrt = "parameters"
  // CHECK: %[[DX:.*]] = "tessera_solver.linear_solve"(%arg0, %arg1, %[[JR]], %[[RHS]])
  // CHECK-SAME: rhs_scale = -1.000000e+00
  // CHECK-SAME: transpose = false
  // CHECK-SAME: algorithm = "gmres"
  // CHECK-SAME: matrix_free = true
  // CHECK: return %[[DX]] : tensor<4xf32>
}
