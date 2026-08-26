// RUN: not tessera-opt %s --tessera-autodiff-paired 2>&1 | FileCheck %s

// W4 / integrated-plan queue order 2 — `tessera.control_scan` is the fourth
// control primitive and the only one with no reverse rule. Bounded `if`,
// counted `for`, and canonical bounded `while` all differentiate through the
// scf region machinery; scan does not, and previously said only that some
// interface was missing.
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
// What blocks it is structural. The reverse scan needs the BODY's paired
// backward and a residual tape of the intermediate carries, which the forward
// scan does not stack — and `AdjointInterface::buildAdjoint` receives only an
// OpBuilder at the forward site and may only emit ops. So the rule belongs in
// the paired pass beside the scf region handling. The diagnostic says that, so
// the next reader does not implement it in the wrong place.

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

// It fails CLOSED, and names the op rather than an absent interface.
// CHECK: error: AUTODIFF_CONTROL_SCAN_UNSUPPORTED: tessera.control_scan
// The notes carry the settled mathematics and the reason for the location, so
// the next implementation starts from them rather than re-deriving.
// CHECK: note: {{.*}}adjoint of a scan is a scan
// CHECK: note: {{.*}}SAVE/RECOMPUTE/HYBRID
// CHECK-NOT: func.func @scan_reverse__bwd
