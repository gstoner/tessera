// RUN: tessera-opt --tessera-lower-to-x86 %s -split-input-file -verify-diagnostics

// Sprint V6b (2026-05-22) — `--tessera-symdim-equality` integrated into
// the `tessera-lower-to-x86` and `tessera-lower-to-gpu` named pipelines.
// Inserted AFTER DistributionLoweringPass.
//
// This fixture proves the pass actually fires inside the named pipeline
// (not just standalone via `--tessera-symdim-equality`).  A function
// carrying a broken `tessera.dim_bindings` clause must fail the
// pipeline with the same stable diagnostic the standalone pass emits.

// ─────────────────────────────────────────────────────────────────────────
// NEGATIVE: SYMDIM_BINDING_VIOLATION inside the full x86 pipeline.
// dim_sizes claims D = 65 while D = H * Dh = 4 * 16 = 64.
// The pipeline runs canonicalize → fusion passes → distribution
// lowering → SymbolicDimEqualityPass — which catches the broken
// equation before tiling/x86 lowering ever runs.
// ─────────────────────────────────────────────────────────────────────────

// expected-error @+1 {{SYMDIM_BINDING_VIOLATION: binding 'D = H * Dh' violated: D = 65 but product of RHS = 64}}
func.func @broken_binding_caught_in_pipeline(%x: tensor<2x4x65xf32>) -> tensor<2x4x65xf32>
    attributes {
      tessera.dim_bindings = ["D = H * Dh"],
      tessera.dim_sizes = { H = 4 : i64, Dh = 16 : i64, D = 65 : i64 }
    } {
  return %x : tensor<2x4x65xf32>
}
