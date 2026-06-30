// CF4d-2 — GenerateROCMControlForNormKernel lowers a norm-in-loop control_for
// (carry = rmsnorm(carry) / layer_norm(carry), carry 1xK) to ONE
// cooperative-workgroup gpu.func on the CF4d-1 substrate: the carry lives in LDS,
// each thread computes the normalization statistic over the whole LDS-resident
// carry and normalizes its own element, and a gpu.barrier separates iterations.
// On-device execution on gfx1151 is proven by
// tests/unit/test_rocm_control_for_norm_exec.py.
//
// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt %s -split-input-file --generate-rocm-control-for-norm-kernel \
// RUN:   --allow-unregistered-dialect | FileCheck %s

// ─── rmsnorm(carry) looped: x / sqrt(mean(x²) + eps) ────────────────────────
func.func @nb(%c: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "tessera.rmsnorm"(%c) {eps = 1.000000e-05 : f64} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// CHECK:       gpu.func @tessera_control_for_norm_0({{.*}}index) workgroup({{.*}}memref<256xf32, #gpu.address_space<workgroup>>) kernel
// CHECK:         memref.store {{.*}}memref<256xf32, #gpu.address_space<workgroup>>
// CHECK:         gpu.barrier
// CHECK:         scf.for
// the rmsnorm statistic (Σ x²) + normalize
// CHECK:           scf.if %{{.*}} -> (f32) {
// CHECK:             scf.for {{.*}} iter_args
// CHECK:               memref.load {{.*}}memref<256xf32, #gpu.address_space<workgroup>>
// CHECK:               arith.mulf
// CHECK:             math.sqrt
// CHECK:             arith.divf
// CHECK:           gpu.barrier
// CHECK:           memref.store {{.*}}memref<256xf32, #gpu.address_space<workgroup>>
// CHECK:           gpu.barrier
// CHECK:         gpu.return
func.func @f(%init: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %r = "tessera.control_for"(%init) {
    body = @nb, start = 0 : i64, stop = 2 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64
  } : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %r : tensor<1x4xf32>
}

// -----
// ─── layer_norm(carry) looped: (x-μ)/sqrt(mean((x-μ)²)+eps) — two reductions ─
func.func @lb(%c: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %0 = "tessera.layer_norm"(%c) {eps = 1.000000e-05 : f64} : (tensor<1x8xf32>) -> tensor<1x8xf32>
  return %0 : tensor<1x8xf32>
}

// CHECK-LABEL: func.func @g
// CHECK:       gpu.func @tessera_control_for_norm_0
// CHECK:         scf.if %{{.*}} -> (f32) {
// CHECK:           arith.subf
// CHECK:           math.sqrt
func.func @g(%init: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %r = "tessera.control_for"(%init) {
    body = @lb, start = 0 : i64, stop = 2 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64
  } : (tensor<1x8xf32>) -> tensor<1x8xf32>
  return %r : tensor<1x8xf32>
}

// -----
// ─── a non-norm (elementwise) body is NOT lowered by this pass — left alone ──
func.func @eb(%c: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "tessera.add"(%c, %c) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// CHECK-LABEL: func.func @h
// CHECK:       tessera.control_for
// CHECK-NOT:   gpu.func
func.func @h(%init: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %r = "tessera.control_for"(%init) {
    body = @eb, start = 0 : i64, stop = 2 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64
  } : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %r : tensor<1x4xf32>
}
