// REQUIRES: tessera-x86-target-ir
//
// ODS triage WIRE slice 1: the named x86 pipeline reaches the composite
// rewrite through tessera-canonicalize (addGraphIRPreLoweringPasses), so no
// target_verify / ntk_rope survives its Graph pre-lowering stage.
//
// RUN: %tessera_strict_opt %s --pass-pipeline='builtin.module(tessera-lower-to-x86)' | FileCheck %s

// CHECK-LABEL: func.func @target_verify_to_softmax
// CHECK-NOT: tessera.target_verify
// CHECK: tessera.softmax
func.func @target_verify_to_softmax(%t: tensor<3xi32>, %l: tensor<3x8xf32>) -> tensor<3x8xf32> {
  %0 = tessera.target_verify %t, %l : (tensor<3xi32>, tensor<3x8xf32>) -> tensor<3x8xf32>
  return %0 : tensor<3x8xf32>
}

// CHECK-LABEL: func.func @ntk_rope_scaled
// CHECK-NOT: tessera.ntk_rope
// CHECK: tessera.div
// CHECK: tessera.rope
func.func @ntk_rope_scaled(%x: tensor<4x8xf32>, %th: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = tessera.ntk_rope %x, %th {scale = 2.0 : f64} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}
