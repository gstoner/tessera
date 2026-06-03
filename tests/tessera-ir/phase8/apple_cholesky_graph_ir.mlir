// L-series linalg pilot — L1: tessera.cholesky Graph IR op parse/print roundtrip.
// Negative verifier cases live in apple_cholesky_graph_ir_invalid.mlir.
//
// RUN: tessera-opt %s --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @chol_square
// CHECK: tessera.cholesky %{{.*}} : (tensor<4x4xf32>) -> tensor<4x4xf32>
func.func @chol_square(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = tessera.cholesky %a : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// Dynamic dims are accepted (the verifier only rejects statically-known
// non-square / shape-mismatched cases).
// CHECK-LABEL: func.func @chol_dynamic
// CHECK: tessera.cholesky %{{.*}} : (tensor<?x?xf32>) -> tensor<?x?xf32>
func.func @chol_dynamic(%a: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tessera.cholesky %a : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// The `lower` attribute round-trips (default true; here set explicitly false).
// CHECK-LABEL: func.func @chol_upper
// CHECK: tessera.cholesky %{{.*}} {lower = false}
func.func @chol_upper(%a: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %0 = tessera.cholesky %a {lower = false} : (tensor<3x3xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}
