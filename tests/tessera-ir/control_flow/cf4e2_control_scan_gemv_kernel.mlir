// CF4e-2 — GenerateROCMControlScanGemvKernel lowers a linear state-space scan
// (h_t = h_{t-1} @ W + x_t : a GEMV body + a W capture + per-step xs) to one
// cooperative-workgroup gpu.func: h in LDS, GEMV + per-step input add, barrier
// per step, stacked ys. On-device execution on gfx1151 is proven by
// tests/unit/test_rocm_control_scan_gemv_exec.py.
//
// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt %s -split-input-file --generate-rocm-control-scan-gemv-kernel \
// RUN:   | FileCheck %s

// ─── linear SSM body: (h, x, W) -> (h@W + x, same) ──────────────────────────
func.func @sb(%h: tensor<1x4xf32>, %x: tensor<1x4xf32>, %w: tensor<4x4xf32>)
    -> (tensor<1x4xf32>, tensor<1x4xf32>) {
  %m = "tessera.matmul"(%h, %w) : (tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<1x4xf32>
  %s = "tessera.add"(%m, %x) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  return %s, %s : tensor<1x4xf32>, tensor<1x4xf32>
}

// CHECK-LABEL: func.func @f
// kernel ABI (INIT, XS, W, COUT, YS : memref<?xf32>, K : index); h in LDS, then
// the trip loop: the GEMV reduction (mulf+addf), per-step xs add, two barriers,
// new carry + stacked ys store; finally the carry → cout store.
// CHECK:       gpu.func @tessera_control_scan_gemv_0(%{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: index) workgroup({{.*}}memref<256xf32, #gpu.address_space<workgroup>>) kernel
// CHECK:         gpu.barrier
// CHECK:         scf.for
// CHECK:           arith.mulf
// CHECK:           arith.addf
// CHECK:           gpu.barrier
// CHECK:           memref.store
// CHECK:           gpu.barrier
// CHECK:         gpu.return
func.func @f(%init: tensor<1x4xf32>, %xs: tensor<3x4xf32>, %W: tensor<4x4xf32>)
    -> (tensor<1x4xf32>, tensor<3x4xf32>) {
  %c, %ys = "tessera.control_scan"(%init, %xs, %W) {
    body = @sb, trip = 3 : i64, carry_arg_index = 0 : i64
  } : (tensor<1x4xf32>, tensor<3x4xf32>, tensor<4x4xf32>) -> (tensor<1x4xf32>, tensor<3x4xf32>)
  return %c, %ys : tensor<1x4xf32>, tensor<3x4xf32>
}

// -----
// ─── a no-capture scan (no W) is the elementwise CF4e-1 form — left untouched ─
func.func @sb2(%h: tensor<1x4xf32>, %x: tensor<1x4xf32>)
    -> (tensor<1x4xf32>, tensor<1x4xf32>) {
  %s = "tessera.add"(%h, %x) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  return %s, %s : tensor<1x4xf32>, tensor<1x4xf32>
}

// CHECK-LABEL: func.func @g
// CHECK:       tessera.control_scan
// CHECK-NOT:   gpu.func
func.func @g(%init: tensor<1x4xf32>, %xs: tensor<3x4xf32>)
    -> (tensor<1x4xf32>, tensor<3x4xf32>) {
  %c, %ys = "tessera.control_scan"(%init, %xs) {
    body = @sb2, trip = 3 : i64, carry_arg_index = 0 : i64
  } : (tensor<1x4xf32>, tensor<3x4xf32>) -> (tensor<1x4xf32>, tensor<3x4xf32>)
  return %c, %ys : tensor<1x4xf32>, tensor<3x4xf32>
}
