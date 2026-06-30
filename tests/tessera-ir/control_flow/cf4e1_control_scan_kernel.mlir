// CF4e-1 — GenerateROCMControlScanKernel lowers an elementwise-body
// tessera.control_scan (the 4th control primitive: per-step xs input, stacked ys
// output) to one per-thread gpu.func device kernel for gfx1151. On-device
// execution is proven by tests/unit/test_rocm_control_scan_exec.py.
//
// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt %s -split-input-file --generate-rocm-control-scan-kernel \
// RUN:   | FileCheck %s

// ─── elementwise scan body: carry' = carry + xt ; y = carry' ────────────────
func.func @sb(%c: tensor<4xf32>, %x: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = "tessera.add"(%c, %x) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0, %0 : tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: func.func @f
// kernel ABI (INIT, XS, YS, COUT : memref<?xf32>, N : index); init[gid] load,
// then the trip loop: xs[t*N+gid] load, body, ys[t*N+gid] store, yield carry';
// finally the carry → cout store.
// CHECK:       gpu.func @tessera_control_scan_0(%{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: index) kernel
// CHECK:         memref.load
// CHECK:         scf.for
// CHECK:           memref.load
// CHECK:           arith.addf
// CHECK:           memref.store
// CHECK:           scf.yield
// CHECK:         memref.store
// CHECK:         gpu.return
func.func @f(%init: tensor<4xf32>, %xs: tensor<3x4xf32>) -> (tensor<4xf32>, tensor<3x4xf32>) {
  %c, %ys = "tessera.control_scan"(%init, %xs) {
    body = @sb, trip = 3 : i64, carry_arg_index = 0 : i64
  } : (tensor<4xf32>, tensor<3x4xf32>) -> (tensor<4xf32>, tensor<3x4xf32>)
  return %c, %ys : tensor<4xf32>, tensor<3x4xf32>
}

// -----
// ─── a cross-element (matmul) scan body is NOT elementwise — left untouched ──
func.func @sb2(%c: tensor<4xf32>, %x: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = "tessera.add"(%c, %x) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %1 = "tessera.softmax"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  return %1, %1 : tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: func.func @g
// CHECK:       tessera.control_scan
// CHECK-NOT:   gpu.func
func.func @g(%init: tensor<4xf32>, %xs: tensor<3x4xf32>) -> (tensor<4xf32>, tensor<3x4xf32>) {
  %c, %ys = "tessera.control_scan"(%init, %xs) {
    body = @sb2, trip = 3 : i64, carry_arg_index = 0 : i64
  } : (tensor<4xf32>, tensor<3x4xf32>) -> (tensor<4xf32>, tensor<3x4xf32>)
  return %c, %ys : tensor<4xf32>, tensor<3x4xf32>
}
