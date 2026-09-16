// RUN: tessera-opt %s --tessera-tiling --allow-unregistered-dialect | FileCheck %s
//
// A Graph IR matmul carrying its `bias` / `residual` epilogue operands (the
// tracer's keyword-operand form) and its `activation` attribute is tiled as
// the plain reduction, and the epilogue is re-applied once on the logical
// product in the contract's order: bias broadcast per output column, then the
// activation, then the residual. Before 2026-09-15 the shared TilingPass kept
// the marker attrs on the inner tile op but dropped the operands, so the pass
// failed its own verifier and no backend ever saw the epilogue; the activation
// then stayed on every inner K step, where a consumer honoring it would have
// applied it per partial product.

// CHECK-LABEL: func.func @matmul_bias
// CHECK: scf.for
// CHECK: tessera.matmul %{{.*}}, %{{.*}} {
// CHECK-NOT: bias =
// CHECK-SAME: tessera.canonical_k_step
// CHECK: %[[BB:.*]] = tessera.broadcast %arg2 {shape = [32, 48]} : (tensor<48xf32>) -> tensor<32x48xf32>
// CHECK: tessera.add %{{.*}}, %[[BB]] : (tensor<32x48xf32>, tensor<32x48xf32>) -> tensor<32x48xf32>
// CHECK: return
func.func @matmul_bias(%a: tensor<32x64xf32>, %b: tensor<64x48xf32>, %bias: tensor<48xf32>) -> tensor<32x48xf32> {
  %0 = tessera.matmul %a, %b, %bias {bias = "row"} : (tensor<32x64xf32>, tensor<64x48xf32>, tensor<48xf32>) -> tensor<32x48xf32>
  return %0 : tensor<32x48xf32>
}

// Bias first, residual second, on the ragged logical product (M = 30 is
// padded to a tile multiple inside the nest and sliced back before the epilogue).
// CHECK-LABEL: func.func @matmul_bias_residual
// CHECK: tensor.extract_slice %{{.*}} : tensor<32x48xf32> to tensor<30x48xf32>
// CHECK: %[[BB:.*]] = tessera.broadcast %arg2 {shape = [30, 48]}
// CHECK: %[[S:.*]] = tessera.add %{{.*}}, %[[BB]]
// CHECK: tessera.add %[[S]], %arg3 : (tensor<30x48xf32>, tensor<30x48xf32>) -> tensor<30x48xf32>
func.func @matmul_bias_residual(%a: tensor<30x64xf32>, %b: tensor<64x48xf32>, %bias: tensor<48xf32>, %res: tensor<30x48xf32>) -> tensor<30x48xf32> {
  %0 = tessera.matmul %a, %b, %bias, %res {bias = "row", residual = "res"} : (tensor<30x64xf32>, tensor<64x48xf32>, tensor<48xf32>, tensor<30x48xf32>) -> tensor<30x48xf32>
  return %0 : tensor<30x48xf32>
}

// Residual alone (no bias operand at index 2).
// CHECK-LABEL: func.func @matmul_residual
// CHECK-NOT: tessera.broadcast
// CHECK: tessera.add %{{.*}}, %arg2 : (tensor<32x48xf32>, tensor<32x48xf32>) -> tensor<32x48xf32>
func.func @matmul_residual(%a: tensor<32x64xf32>, %b: tensor<64x48xf32>, %res: tensor<32x48xf32>) -> tensor<32x48xf32> {
  %0 = tessera.matmul %a, %b, %res {residual = "res"} : (tensor<32x64xf32>, tensor<64x48xf32>, tensor<32x48xf32>) -> tensor<32x48xf32>
  return %0 : tensor<32x48xf32>
}

// Activation between bias and residual, and never on the inner K step.
// CHECK-LABEL: func.func @matmul_bias_gelu_residual
// CHECK: tessera.matmul %{{.*}}, %{{.*}} {
// CHECK-NOT: activation =
// CHECK-SAME: tessera.canonical_k_step
// CHECK: %[[BB:.*]] = tessera.broadcast %arg2 {shape = [32, 48]}
// CHECK: %[[S:.*]] = tessera.add %{{.*}}, %[[BB]]
// CHECK: %[[G:.*]] = tessera.gelu %[[S]] : (tensor<32x48xf32>) -> tensor<32x48xf32>
// CHECK: tessera.add %[[G]], %arg3 : (tensor<32x48xf32>, tensor<32x48xf32>) -> tensor<32x48xf32>
func.func @matmul_bias_gelu_residual(%a: tensor<32x64xf32>, %b: tensor<64x48xf32>, %bias: tensor<48xf32>, %res: tensor<32x48xf32>) -> tensor<32x48xf32> {
  %0 = tessera.matmul %a, %b, %bias, %res {activation = "gelu", bias = "bias", residual = "residual"} : (tensor<32x64xf32>, tensor<64x48xf32>, tensor<48xf32>, tensor<32x48xf32>) -> tensor<32x48xf32>
  return %0 : tensor<32x48xf32>
}

// Activation alone: no epilogue operands, the relu still leaves the K step.
// CHECK-LABEL: func.func @matmul_relu
// CHECK: tessera.matmul %{{.*}}, %{{.*}} {
// CHECK-NOT: activation =
// CHECK: tessera.relu %{{.*}} : (tensor<32x48xf32>) -> tensor<32x48xf32>
func.func @matmul_relu(%a: tensor<32x64xf32>, %b: tensor<64x48xf32>) -> tensor<32x48xf32> {
  %0 = tessera.matmul %a, %b {activation = "relu"} : (tensor<32x64xf32>, tensor<64x48xf32>) -> tensor<32x48xf32>
  return %0 : tensor<32x48xf32>
}

// An unknown activation is not guessed: the nest is left for the verifier.
// CHECK-LABEL: func.func @matmul_unknown_activation
// CHECK-NOT: scf.for
// CHECK: tessera.matmul %arg0, %arg1 {activation = "swish2"}
func.func @matmul_unknown_activation(%a: tensor<32x64xf32>, %b: tensor<64x48xf32>) -> tensor<32x48xf32> {
  %0 = tessera.matmul %a, %b {activation = "swish2"} : (tensor<32x64xf32>, tensor<64x48xf32>) -> tensor<32x48xf32>
  return %0 : tensor<32x48xf32>
}

// A non-string activation attribute (a programmatic enum) is not "absent":
// the nest is left untouched rather than stripped into a plain matmul.
// CHECK-LABEL: func.func @matmul_enum_activation
// CHECK-NOT: scf.for
// CHECK: tessera.matmul %arg0, %arg1 {activation = 2 : i64}
func.func @matmul_enum_activation(%a: tensor<32x64xf32>, %b: tensor<64x48xf32>) -> tensor<32x48xf32> {
  %0 = tessera.matmul %a, %b {activation = 2 : i64} : (tensor<32x64xf32>, tensor<64x48xf32>) -> tensor<32x48xf32>
  return %0 : tensor<32x48xf32>
}
