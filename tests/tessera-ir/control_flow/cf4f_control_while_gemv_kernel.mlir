// CF4f — GenerateROCMControlWhileGemvKernel lowers a CROSS-ELEMENT control_while
// (power iteration: h = h @ W while Σh > eps, bounded by max_iters) to one
// cooperative-workgroup gpu.func: h in LDS, a scf.while whose before-region is a
// UNIFORM reduction-cond (Σ lds > eps AND i < max — every thread computes the
// same predicate so the per-iteration barriers are safe), after-region a
// cooperative GEMV. On-device execution on gfx1151 is proven by
// tests/unit/test_rocm_control_while_gemv_exec.py.
//
// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt %s -split-input-file --generate-rocm-control-while-gemv-kernel \
// RUN:   | FileCheck %s

func.func @b(%h: tensor<1x4xf32>, %w: tensor<4x4xf32>) -> tensor<1x4xf32> {
  %m = "tessera.matmul"(%h, %w) : (tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<1x4xf32>
  return %m : tensor<1x4xf32>
}
func.func @c(%h: tensor<1x4xf32>) -> tensor<1x1xf32> {
  %s = "tessera.reduce"(%h) {kind = "sum", axis = 1 : i64} : (tensor<1x4xf32>) -> tensor<1x1xf32>
  return %s : tensor<1x1xf32>
}

// CHECK-LABEL: func.func @f
// kernel ABI (H, W, OUT : memref<?xf32>, K : index); h → LDS, barrier, then the
// scf.while: before = uniform reduction cond (addf reduction, cmpf ogt eps,
// andi), after = GEMV (mulf) + barriers; finally the carry → OUT store.
// CHECK:       gpu.func @tessera_control_while_gemv_0(%{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: index) workgroup({{.*}}memref<256xf32, #gpu.address_space<workgroup>>) kernel
// CHECK:         gpu.barrier
// CHECK:         scf.while
// CHECK:           arith.cmpf ogt
// CHECK:           arith.andi
// CHECK:           scf.condition
// CHECK:           arith.mulf
// CHECK:           gpu.barrier
// CHECK:         gpu.return
func.func @f(%init: tensor<1x4xf32>, %W: tensor<4x4xf32>) -> tensor<1x4xf32> {
  %r = "tessera.control_while"(%init, %W) {
    body = @b, cond = @c, carry_arg_index = 0 : i64, max_iters = 16 : i64,
    tessera.while_cond_eps = 1.000000e-01 : f32
  } : (tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<1x4xf32>
  return %r : tensor<1x4xf32>
}

// -----
// ─── a non-reduce cond (the elementwise CF4c-cont form) is NOT this uniform
// ─── reduction-cond — left untouched for CF4c-cont / the guard ──────────────
func.func @b2(%h: tensor<1x4xf32>, %w: tensor<4x4xf32>) -> tensor<1x4xf32> {
  %m = "tessera.matmul"(%h, %w) : (tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<1x4xf32>
  return %m : tensor<1x4xf32>
}
func.func @c2(%h: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %s = "tessera.sigmoid"(%h) : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %s : tensor<1x4xf32>
}

// CHECK-LABEL: func.func @g
// CHECK:       tessera.control_while
// CHECK-NOT:   gpu.func
func.func @g(%init: tensor<1x4xf32>, %W: tensor<4x4xf32>) -> tensor<1x4xf32> {
  %r = "tessera.control_while"(%init, %W) {
    body = @b2, cond = @c2, carry_arg_index = 0 : i64, max_iters = 16 : i64
  } : (tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<1x4xf32>
  return %r : tensor<1x4xf32>
}

// -----
// ─── a reduce over the WRONG axis (axis=0 → 1×K, K predicate elements, not the
// ─── whole-carry sum the kernel computes) is left untouched — never lowered to
// ─── a cond that disagrees with the kernel's total-Σ loop ───────────────────
func.func @b3(%h: tensor<1x4xf32>, %w: tensor<4x4xf32>) -> tensor<1x4xf32> {
  %m = "tessera.matmul"(%h, %w) : (tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<1x4xf32>
  return %m : tensor<1x4xf32>
}
func.func @c3(%h: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %s = "tessera.reduce"(%h) {kind = "sum", axis = 0 : i64} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %s : tensor<1x4xf32>
}

// CHECK-LABEL: func.func @h
// CHECK:       tessera.control_while
// CHECK-NOT:   gpu.func
func.func @h(%init: tensor<1x4xf32>, %W: tensor<4x4xf32>) -> tensor<1x4xf32> {
  %r = "tessera.control_while"(%init, %W) {
    body = @b3, cond = @c3, carry_arg_index = 0 : i64, max_iters = 16 : i64,
    tessera.while_cond_eps = 1.000000e-01 : f32
  } : (tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<1x4xf32>
  return %r : tensor<1x4xf32>
}
