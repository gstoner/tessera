// CF4b — GenerateROCMControlForKernel: an elementwise-body tessera.control_for
// becomes ONE gpu.func device kernel — grid over the carry's elements, each
// thread running the loop's scf.for (K iterations) with the body as scalar ops.
// One dispatch, not one launch per iteration. (On-device execution on gfx1151
// is proven by tests/unit/test_rocm_control_for_exec.py.)
//
// RUN: tessera-opt %s -split-input-file --generate-rocm-control-for-kernel \
// RUN:   --allow-unregistered-dialect | FileCheck %s

// ─── elementwise body (add → doubling) lowers to a device kernel ────────────
func.func @loop_body(%c: tensor<8xf32>) -> tensor<8xf32> {
  %0 = "tessera.add"(%c, %c) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK:       gpu.module @tessera_control_for_0_mod
// CHECK:       gpu.func @tessera_control_for_0(%{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: index) kernel
// CHECK:         gpu.block_id  x
// CHECK:         gpu.thread_id  x
// CHECK:         %[[G:.*]] = arith.addi
// CHECK:         arith.cmpi slt, %[[G]]
// CHECK:         scf.if
// CHECK:           memref.load
// CHECK:           %[[R:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[A:.*]] = %{{.*}}) -> (f32)
// CHECK:             arith.addf %[[A]], %[[A]] : f32
// CHECK:             scf.yield
// CHECK:           memref.store %[[R]]
// CHECK:         gpu.return
func.func @f(%init: tensor<8xf32>) -> tensor<8xf32> {
  %r = "tessera.control_for"(%init) {
    body = @loop_body, start = 0 : i64, stop = 4 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %r : tensor<8xf32>
}

// -----
// ─── silu body also lowers (unary elementwise → scalar math) ────────────────
func.func @silu_body(%c: tensor<8xf32>) -> tensor<8xf32> {
  %0 = "tessera.silu"(%c) : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK:       gpu.func @tessera_control_for_0
// CHECK:         scf.for
// CHECK:           math.exp
// CHECK:           arith.mulf
func.func @g(%init: tensor<8xf32>) -> tensor<8xf32> {
  %r = "tessera.control_for"(%init) {
    body = @silu_body, start = 0 : i64, stop = 3 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %r : tensor<8xf32>
}

// -----
// ─── a non-elementwise (matmul) body is NOT lowered — left for the guard ────
func.func @mm_body(%c: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = "tessera.matmul"(%c, %c) : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-NOT:   gpu.func
// CHECK:       tessera.control_for
func.func @h(%init: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %r = "tessera.control_for"(%init) {
    body = @mm_body, start = 0 : i64, stop = 2 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %r : tensor<8x8xf32>
}

// -----
// ─── a rank>1 carry is NOT lowered — the kernel ABI is a flat memref<?xf32> ──
func.func @r2_body(%c: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %0 = "tessera.add"(%c, %c) : (tensor<1x8xf32>, tensor<1x8xf32>) -> tensor<1x8xf32>
  return %0 : tensor<1x8xf32>
}

// CHECK-NOT:   gpu.func
// CHECK:       tessera.control_for
func.func @r2(%init: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %r = "tessera.control_for"(%init) {
    body = @r2_body, start = 0 : i64, stop = 4 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64
  } : (tensor<1x8xf32>) -> tensor<1x8xf32>
  return %r : tensor<1x8xf32>
}
