// RUN: tessera-opt --canonicalize --cse %s | FileCheck %s
//
// Phase 1 (front-to-back closure plan): Tessera-dialect per-op folders /
// canonicalizers + DCE + CSE — the exact passes now wired into the tessera_jit
// CPU pipeline (pm1: canonicalize → cse → tessera-to-linalg), so they are
// observable end-to-end on the executed path. See
// docs/audit/compiler/COMPILER_AUDIT.md (Phase 1 / Phase 4).

// transpose(transpose(x)) -> x  (a no-perm transpose is its own inverse).
// CHECK-LABEL: func.func @transpose_sq
// CHECK-NOT: tessera.transpose
// CHECK: return %arg0
func.func @transpose_sq(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = "tessera.transpose"(%x) : (tensor<4x8xf32>) -> tensor<8x4xf32>
  %1 = "tessera.transpose"(%0) : (tensor<8x4xf32>) -> tensor<4x8xf32>
  return %1 : tensor<4x8xf32>
}

// cast(x): T -> T  folds to x (identity cast, no numeric policy).
// CHECK-LABEL: func.func @identity_cast
// CHECK-NOT: tessera.cast
// CHECK: return %arg0
func.func @identity_cast(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = "tessera.cast"(%x) : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// A real (type-changing) cast must NOT fold.
// CHECK-LABEL: func.func @real_cast
// CHECK: tessera.cast
func.func @real_cast(%x: tensor<4x8xf32>) -> tensor<4x8xf16> {
  %0 = "tessera.cast"(%x) : (tensor<4x8xf32>) -> tensor<4x8xf16>
  return %0 : tensor<4x8xf16>
}

// A single transpose must survive (nothing to fold against).
// CHECK-LABEL: func.func @single_transpose
// CHECK: tessera.transpose
func.func @single_transpose(%x: tensor<4x8xf32>) -> tensor<8x4xf32> {
  %0 = "tessera.transpose"(%x) : (tensor<4x8xf32>) -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// DCE: a dead pure op (its result is unused) is eliminated.
// CHECK-LABEL: func.func @dce_dead_op
// CHECK-NOT: tessera.gelu
// CHECK: tessera.matmul
func.func @dce_dead_op(%x: tensor<4x8xf32>, %w: tensor<8x6xf32>) -> tensor<4x6xf32> {
  %dead = "tessera.gelu"(%x) : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %0 = "tessera.matmul"(%x, %w) : (tensor<4x8xf32>, tensor<8x6xf32>) -> tensor<4x6xf32>
  return %0 : tensor<4x6xf32>
}

// CSE: two identical pure matmuls (the shared-QKV-projection pattern) collapse
// to one.
// CHECK-LABEL: func.func @cse_dup
// CHECK-COUNT-1: tessera.matmul
// CHECK-NOT: tessera.matmul
func.func @cse_dup(%x: tensor<4x8xf32>, %w: tensor<8x6xf32>) -> tensor<4x6xf32> {
  %a = "tessera.matmul"(%x, %w) : (tensor<4x8xf32>, tensor<8x6xf32>) -> tensor<4x6xf32>
  %b = "tessera.matmul"(%x, %w) : (tensor<4x8xf32>, tensor<8x6xf32>) -> tensor<4x6xf32>
  %0 = "tessera.add"(%a, %b) : (tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<4x6xf32>
  return %0 : tensor<4x6xf32>
}
