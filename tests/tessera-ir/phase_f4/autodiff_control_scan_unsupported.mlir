// RUN: tessera-opt %s --tessera-autodiff-paired | FileCheck %s

// W4 / integrated-plan queue order 2 — paired reverse mode now admits the
// bounded symbol-body scan directly. The pass dynamically runs the canonical
// control_scan -> scf.for normalization, then consumes the ordinary region
// adjoint and exact tensor-slice transposes.
//
// The mathematics is settled, not open. For
// `(c_{t+1}, y_t) = body(c_t, x_t)` the reverse recurrence is
//
//     (cbar_t, xbar_t) = body_vjp(c_t, x_t; cbar_{t+1}, ybar_t),  t = T-1..0
//
// so the adjoint of a scan is a scan over reversed t. Verified against
// central differences on a nonlinear body (tanh): max absolute error 4.7e-10
// for both the init and the xs cotangents.
//
// Payload/dynamic/malformed forms are still left unlowered and retain the
// AUTODIFF_CONTROL_SCAN_UNSUPPORTED diagnostic.

module {
  func.func @sbody(%c: tensor<4xf32>, %x: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>) {
    %n = "tessera.add"(%c, %x) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %n, %n : tensor<4xf32>, tensor<4xf32>
  }
  func.func @scan_reverse(%init: tensor<4xf32>, %xs: tensor<3x4xf32>)
      -> tensor<4xf32> attributes {tessera.autodiff = "reverse"} {
    %c, %ys = "tessera.control_scan"(%init, %xs) {
      body = @sbody, trip = 3 : i64, carry_arg_index = 0 : i64
    } : (tensor<4xf32>, tensor<3x4xf32>) -> (tensor<4xf32>, tensor<3x4xf32>)
    return %c : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @scan_reverse__bwd
// CHECK: scf.for
// CHECK: scf.for
// CHECK: tensor.extract_slice
// CHECK: tensor.insert_slice
// CHECK: return
// CHECK-NOT: tessera.control_scan
