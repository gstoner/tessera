// RUN: tessera-opt --allow-unregistered-dialect --tessera-linalg-mixed-precision %s | FileCheck %s --check-prefix=MP
// RUN: tessera-opt --allow-unregistered-dialect --pass-pipeline='builtin.module(func.func(tessera-linalg-iterative-refinement))' %s | FileCheck %s --check-prefix=IR
//
// TILE-LATENT-DEFECTS-2026-09-27. The linalg solver passes select ops by op
// identity. They used to match substrings: `contains("solve")` is true for
// every `tessera_solver.*` op through the dialect prefix, and MixedPrecision's
// `contains("lu")` / `contains("factor")` matched `tessera.gelu`,
// `tessera.relu` and `tessera.adafactor`. The first function is the negative
// case (Decision #10a): names that contain "solve", "lu" or "chol" but are not
// solver ops must come out UNannotated.

module {
  // MP-LABEL: func.func @not_solver_ops
  // MP-NOT: tessera.mixed_precision_annotated
  // MP-NOT: tessera.compute_dtype
  // MP: return
  // IR-LABEL: func.func @not_solver_ops
  // IR-NOT: tessera_solver.ir_annotated
  // IR: return
  func.func @not_solver_ops(%x: tensor<4xf32>) {
    %g = tessera.gelu %x : (tensor<4xf32>) -> tensor<4xf32>
    %r = tessera.relu %x : (tensor<4xf32>) -> tensor<4xf32>
    %n = "test.resolve_names"(%x) : (tensor<4xf32>) -> tensor<4xf32>
    %c = "test.cholesky_like_but_not"(%x) : (tensor<4xf32>) -> tensor<4xf32>
    return
  }

  func.func private @res(%t: tensor<4xf32>, %x: tensor<4xf32>) -> tensor<4xf32> {
    %d = arith.subf %x, %t : tensor<4xf32>
    return %d : tensor<4xf32>
  }

  // MP-LABEL: func.func @solver_ops
  // MP: tessera_solver.potrf
  // MP-SAME: tessera.compute_dtype = "f32"
  // MP: tessera_solver.potrs
  // MP-SAME: tessera.compute_dtype = "f16"
  // MP: tessera_solver.trsm
  // MP-SAME: tessera.compute_dtype = "f16"
  // MP: tessera.cholesky
  // MP-SAME: tessera.compute_dtype = "f32"
  // MP: tessera.cholesky_solve
  // MP-SAME: tessera.compute_dtype = "f16"
  // MP: "tessera_solver.residual"
  // MP-SAME: tessera.compute_dtype = "f32"
  // The matrix-free IFT solve is not a dense factor/solve kernel; the old
  // prefix match stamped it fp16.
  // MP: "tessera_solver.implicit"
  // MP-NOT: tessera.compute_dtype
  // MP: return
  //
  // IR-LABEL: func.func @solver_ops
  // A factorization has no solution to refine.
  // IR: tessera_solver.potrf
  // IR-NOT: ir_annotated
  // IR: tessera_solver.potrs
  // IR-SAME: tessera_solver.ir_annotated
  // IR: tessera_solver.trsm
  // IR-SAME: tessera_solver.ir_annotated
  // IR: tessera.cholesky %
  // IR-NOT: ir_annotated
  // IR: tessera.cholesky_solve
  // IR-SAME: tessera_solver.ir_annotated
  // IR: "tessera_solver.residual"
  // IR-NOT: ir_annotated
  // IR: "tessera_solver.implicit"
  // IR-SAME: tessera_solver.ir_annotated
  func.func @solver_ops(%a: tensor<4x4xf32>, %b: tensor<4x2xf32>,
                        %t: tensor<4xf32>) {
    %l = tessera_solver.potrf %a : tensor<4x4xf32> -> tensor<4x4xf32>
    %x = tessera_solver.potrs %l, %b : tensor<4x4xf32>, tensor<4x2xf32> -> tensor<4x2xf32>
    %y = tessera_solver.trsm %l, %b : tensor<4x4xf32>, tensor<4x2xf32> -> tensor<4x2xf32>
    %c = tessera.cholesky %a : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %s = tessera.cholesky_solve %a, %b : (tensor<4x4xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>
    %r = "tessera_solver.residual"(%t, %t) {callee = @res} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %i = "tessera_solver.implicit"(%t) {residual = @res} : (tensor<4xf32>) -> tensor<4xf32>
    return
  }
}
