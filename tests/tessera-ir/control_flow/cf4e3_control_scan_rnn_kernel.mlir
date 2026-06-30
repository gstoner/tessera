// CF4e-3 — GenerateROCMControlScanRnnKernel lowers a nonlinear RNN-cell scan
// (h_t = tanh(h_{t-1} @ W + x_t @ U + b): two GEMV captures W/U + a bias + tanh +
// per-step xs) to one cooperative-workgroup gpu.func. On-device execution on
// gfx1151 is proven by tests/unit/test_rocm_control_scan_rnn_exec.py.
//
// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt %s -split-input-file --generate-rocm-control-scan-rnn-kernel \
// RUN:   | FileCheck %s

// ─── RNN cell body: (h, x, W, U, b) -> (tanh(h@W + x@U + b), same) ──────────
func.func @sb(%h: tensor<1x4xf32>, %x: tensor<1x4xf32>, %w: tensor<4x4xf32>,
    %u: tensor<4x4xf32>, %b: tensor<1x4xf32>) -> (tensor<1x4xf32>, tensor<1x4xf32>) {
  %m1 = "tessera.matmul"(%h, %w) : (tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<1x4xf32>
  %m2 = "tessera.matmul"(%x, %u) : (tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<1x4xf32>
  %s1 = "tessera.add"(%m1, %m2) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  %s2 = "tessera.add"(%s1, %b) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  %t = "tessera.tanh"(%s2) : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %t, %t : tensor<1x4xf32>, tensor<1x4xf32>
}

// CHECK-LABEL: func.func @f
// kernel ABI (INIT, XS, W, U, B, COUT, YS : memref<?xf32>, K : index); h in LDS,
// then the trip loop: the fused two-GEMV reduction (mulf+mulf+addf), + bias,
// tanh, two barriers, new carry + stacked ys store; final carry → cout.
// CHECK:       gpu.func @tessera_control_scan_rnn_0(%{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: index) workgroup({{.*}}memref<256xf32, #gpu.address_space<workgroup>>) kernel
// CHECK:         gpu.barrier
// CHECK:         scf.for
// CHECK:           arith.mulf
// CHECK:           arith.mulf
// CHECK:           math.tanh
// CHECK:           gpu.barrier
// CHECK:           memref.store
// CHECK:           gpu.barrier
// CHECK:         gpu.return
func.func @f(%init: tensor<1x4xf32>, %xs: tensor<3x1x4xf32>, %W: tensor<4x4xf32>,
    %U: tensor<4x4xf32>, %b: tensor<1x4xf32>) -> (tensor<1x4xf32>, tensor<3x1x4xf32>) {
  %c, %ys = "tessera.control_scan"(%init, %xs, %W, %U, %b) {
    body = @sb, trip = 3 : i64, carry_arg_index = 0 : i64
  } : (tensor<1x4xf32>, tensor<3x1x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<1x4xf32>)
      -> (tensor<1x4xf32>, tensor<3x1x4xf32>)
  return %c, %ys : tensor<1x4xf32>, tensor<3x1x4xf32>
}

// -----
// ─── a scan missing the tanh (linear, CF4e-2's form) is NOT this RNN cell —
// ─── left untouched (the CF4e-2 scan-gemv pass / guard handles it) ──────────
func.func @sb2(%h: tensor<1x4xf32>, %x: tensor<1x4xf32>, %w: tensor<4x4xf32>)
    -> (tensor<1x4xf32>, tensor<1x4xf32>) {
  %m = "tessera.matmul"(%h, %w) : (tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<1x4xf32>
  %s = "tessera.add"(%m, %x) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  return %s, %s : tensor<1x4xf32>, tensor<1x4xf32>
}

// CHECK-LABEL: func.func @g
// CHECK:       tessera.control_scan
// CHECK-NOT:   gpu.func
func.func @g(%init: tensor<1x4xf32>, %xs: tensor<3x1x4xf32>, %W: tensor<4x4xf32>)
    -> (tensor<1x4xf32>, tensor<3x1x4xf32>) {
  %c, %ys = "tessera.control_scan"(%init, %xs, %W) {
    body = @sb2, trip = 3 : i64, carry_arg_index = 0 : i64
  } : (tensor<1x4xf32>, tensor<3x1x4xf32>, tensor<4x4xf32>)
      -> (tensor<1x4xf32>, tensor<3x1x4xf32>)
  return %c, %ys : tensor<1x4xf32>, tensor<3x1x4xf32>
}
