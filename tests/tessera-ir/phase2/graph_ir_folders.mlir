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

// ── Phase 1 identity folders (2026-06-16) — elementwise binary family ────────

// x + 0 -> x  (constant zero splat on the rhs).
// CHECK-LABEL: func.func @add_zero
// CHECK-NOT: tessera.add
// CHECK: return %arg0
func.func @add_zero(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %z = arith.constant dense<0.0> : tensor<4x8xf32>
  %0 = "tessera.add"(%x, %z) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// 0 + x -> x  (constant zero splat on the lhs).
// CHECK-LABEL: func.func @zero_add
// CHECK-NOT: tessera.add
// CHECK: return %arg0
func.func @zero_add(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %z = arith.constant dense<0.0> : tensor<4x8xf32>
  %0 = "tessera.add"(%z, %x) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// x - 0 -> x.
// CHECK-LABEL: func.func @sub_zero
// CHECK-NOT: tessera.sub
// CHECK: return %arg0
func.func @sub_zero(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %z = arith.constant dense<0.0> : tensor<4x8xf32>
  %0 = "tessera.sub"(%x, %z) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// x * 1 -> x.
// CHECK-LABEL: func.func @mul_one
// CHECK-NOT: tessera.mul
// CHECK: return %arg0
func.func @mul_one(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %one = arith.constant dense<1.0> : tensor<4x8xf32>
  %0 = "tessera.mul"(%x, %one) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// x / 1 -> x.
// CHECK-LABEL: func.func @div_one
// CHECK-NOT: tessera.div
// CHECK: return %arg0
func.func @div_one(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %one = arith.constant dense<1.0> : tensor<4x8xf32>
  %0 = "tessera.div"(%x, %one) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// Negative: x * 2 is NOT an identity fold — the mul must survive.
// CHECK-LABEL: func.func @mul_two_survives
// CHECK: tessera.mul
func.func @mul_two_survives(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %two = arith.constant dense<2.0> : tensor<4x8xf32>
  %0 = "tessera.mul"(%x, %two) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// Negative: 1 - x is NOT x (sub only folds the rhs-zero case).
// CHECK-LABEL: func.func @one_sub_survives
// CHECK: tessera.sub
func.func @one_sub_survives(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %one = arith.constant dense<1.0> : tensor<4x8xf32>
  %0 = "tessera.sub"(%one, %x) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// transpose-into-matmul: matmul(transpose(A), B) folds the transpose into the
// transposeA flag (per-op MatmulOp canonicalizer — the twin of
// CanonicalizeTesseraIR's TransposeIntoMatmul, now firing under --canonicalize
// so it reaches the executed tessera_jit CPU path).
// CHECK-LABEL: func.func @transpose_into_matmul
// CHECK-NOT: tessera.transpose
// CHECK: tessera.matmul
// CHECK-SAME: transposeA = true
func.func @transpose_into_matmul(%a: tensor<8x4xf32>, %b: tensor<8x16xf32>) -> tensor<4x16xf32> {
  %at = "tessera.transpose"(%a) : (tensor<8x4xf32>) -> tensor<4x8xf32>
  %0 = "tessera.matmul"(%at, %b) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// Double-transpose into matmul composes by XOR: an explicit transposeB=true with
// a folded transpose(B) cancels back to transposeB=false (the default-false attr
// is elided on print, so it must NOT survive as transposeB = true).
// CHECK-LABEL: func.func @xor_flag_compose
// CHECK-NOT: tessera.transpose
// CHECK: tessera.matmul
// CHECK-NOT: transposeB = true
func.func @xor_flag_compose(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> tensor<4x16xf32> {
  %bt = "tessera.transpose"(%b) : (tensor<8x16xf32>) -> tensor<16x8xf32>
  %0 = "tessera.matmul"(%a, %bt) {transposeB = true} : (tensor<4x8xf32>, tensor<16x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}
