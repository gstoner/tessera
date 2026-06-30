// CF4d-1 — GenerateROCMControlForGemvKernel lowers a GEMV-recurrence control_for
// (carry = carry @ W, carry 1xK / W KxK) to ONE cooperative-workgroup gpu.func:
// the carry lives in LDS, thread j computes o[j] = Σ_k carry[k]·W[k][j] by a
// serial dot product, and a gpu.barrier separates loop iterations. The first
// CROSS-ELEMENT control body (per-thread can't express it). On-device execution
// on gfx1151 is proven by tests/unit/test_rocm_control_for_gemv_exec.py.
//
// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt %s -split-input-file --generate-rocm-control-for-gemv-kernel \
// RUN:   --allow-unregistered-dialect | FileCheck %s

// ─── carry (1x4) @ W (4x4), looped 3x ───────────────────────────────────────
func.func @wb(%c: tensor<1x4xf32>, %w: tensor<4x4xf32>) -> tensor<1x4xf32> {
  %0 = "tessera.matmul"(%c, %w) : (tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// CHECK:       gpu.func @tessera_control_for_gemv_0(%{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: index) workgroup(%[[LDS:.*]] : memref<256xf32, #gpu.address_space<workgroup>>) kernel
// load carry into LDS, then a barrier
// CHECK:         memref.store %{{.*}}, %[[LDS]]
// CHECK:         gpu.barrier
// the control loop: per-thread dot product, barrier, write-back, barrier
// CHECK:         scf.for
// CHECK:           scf.if %{{.*}} -> (f32) {
// CHECK:             scf.for %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (f32)
// CHECK:               memref.load %[[LDS]]
// CHECK:               arith.mulf
// CHECK:               arith.addf
// CHECK:           gpu.barrier
// CHECK:           memref.store %{{.*}}, %[[LDS]]
// CHECK:           gpu.barrier
// CHECK:         memref.store
// CHECK:         gpu.return
func.func @f(%init: tensor<1x4xf32>, %w: tensor<4x4xf32>) -> tensor<1x4xf32> {
  %r = "tessera.control_for"(%init, %w) {
    body = @wb, start = 0 : i64, stop = 3 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64
  } : (tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<1x4xf32>
  return %r : tensor<1x4xf32>
}

// -----
// ─── a transposed matmul body is NOT this GEMV form — left untouched ────────
func.func @tb(%c: tensor<1x4xf32>, %w: tensor<4x4xf32>) -> tensor<1x4xf32> {
  %0 = "tessera.matmul"(%c, %w) {transposeB = true} : (tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// CHECK-LABEL: func.func @g
// CHECK:       tessera.control_for
// CHECK-NOT:   gpu.func
func.func @g(%init: tensor<1x4xf32>, %w: tensor<4x4xf32>) -> tensor<1x4xf32> {
  %r = "tessera.control_for"(%init, %w) {
    body = @tb, start = 0 : i64, stop = 3 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64
  } : (tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<1x4xf32>
  return %r : tensor<1x4xf32>
}

// -----
// ─── an elementwise body is NOT a GEMV — this pass leaves it (CF4b owns it) ──
func.func @eb(%c: tensor<1x4xf32>, %w: tensor<4x4xf32>) -> tensor<1x4xf32> {
  %0 = "tessera.add"(%c, %c) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// CHECK-LABEL: func.func @h
// CHECK:       tessera.control_for
// CHECK-NOT:   gpu.func
func.func @h(%init: tensor<1x4xf32>, %w: tensor<4x4xf32>) -> tensor<1x4xf32> {
  %r = "tessera.control_for"(%init, %w) {
    body = @eb, start = 0 : i64, stop = 3 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64
  } : (tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<1x4xf32>
  return %r : tensor<1x4xf32>
}
