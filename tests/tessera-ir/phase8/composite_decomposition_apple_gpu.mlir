// REQUIRES: tessera-apple-backend
//
// ODS triage WIRE slice 1: the Apple GPU -runtime pipeline runs the composite
// rewrite first, so target_verify reaches the softmax MSL runtime call and
// ntk_rope reaches the rope MSL runtime call.
//
// RUN: tessera-opt %s --pass-pipeline='builtin.module(tessera-lower-to-apple_gpu-runtime)' | FileCheck %s

// CHECK-DAG: func.func private @tessera_apple_gpu_softmax_f32_status(i64, i64, i32, i32) -> i32
// CHECK-DAG: func.func private @tessera_apple_gpu_rope_f32(i64, i64, i64, i32, i32)

// CHECK-LABEL: func.func @target_verify
// CHECK-NOT: tessera.target_verify
// CHECK-NOT: tessera.softmax
// CHECK: call @tessera_apple_gpu_softmax_f32_status
func.func @target_verify(%t: tensor<3xi32>, %l: tensor<3x8xf32>) -> tensor<3x8xf32> {
  %0 = tessera.target_verify %t, %l : (tensor<3xi32>, tensor<3x8xf32>) -> tensor<3x8xf32>
  return %0 : tensor<3x8xf32>
}

// CHECK-LABEL: func.func @ntk_rope_unit
// CHECK-NOT: tessera.ntk_rope
// CHECK-NOT: tessera.rope
// CHECK: call @tessera_apple_gpu_rope_f32
func.func @ntk_rope_unit(%x: tensor<4x8xf32>, %th: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = tessera.ntk_rope %x, %th {scale = 1.0 : f64} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// A scaled theta: the rope call consumes theta / scale. This pipeline has no
// Graph `tessera.div` lowering, so the division stays a Graph op here -- the
// recorded gap that keeps the ntk_rope Target row from borrowing rope's
// device-verified ABI evidence (ODS triage, ntk_rope row).
// CHECK-LABEL: func.func @ntk_rope_scaled
// CHECK-NOT: tessera.ntk_rope
// CHECK: %[[D:.*]] = tessera.div
// CHECK: bufferization.to_buffer %[[D]]
// CHECK: call @tessera_apple_gpu_rope_f32
func.func @ntk_rope_scaled(%x: tensor<4x8xf32>, %th: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = tessera.ntk_rope %x, %th {scale = 2.0 : f64} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}
