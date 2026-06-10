// RUN: tessera-opt --tessera-verify="forbid-ops=tessera.matmul" -split-input-file -verify-diagnostics %s
// RUN: tessera-opt --tessera-verify="forbid-ops=tessera.flash_attn" %s -split-input-file | FileCheck %s --check-prefix=CLEAN
// 2026-06-10 audit — pipeline-completeness option on tessera-verify.
//
// Lowering passes gate per-op via notifyMatchFailure, so a partially-lowered
// module still "succeeds" and unmatched ops silently survive. Appending
// `tessera-verify{forbid-ops=...}` to a pipeline turns a survivor into a
// stable named diagnostic (Decision #21).

// CLEAN-LABEL: func.func @still_has_matmul
// CLEAN: tessera.matmul

module attributes {tessera.ir.version = "1.0"} {
  func.func @still_has_matmul(
      %A: tensor<8x16xf32>,
      %B: tensor<16x8xf32>
  ) -> tensor<8x8xf32> {
    // expected-error @+1 {{[TESSERA_VFY_FORBIDDEN_OP] op survived lowering but is forbidden at this pipeline stage}}
    %C = "tessera.matmul"(%A, %B)
        : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
    return %C : tensor<8x8xf32>
  }
}
