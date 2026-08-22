// RUN: %tessera_strict_opt --split-input-file --tessera-nvwgmma-lowering='sm=90' %s \
// RUN:   | FileCheck %s
//
// W1.1 step 2b (compatibility boundary) — NVWGMMALoweringPass threads the
// accumulator instead of silently dropping it.
//
// The WGMMA path builds `ValueRange{A, B}`: it never passes a third
// (accumulator) operand, hardcodes `m64n64k16`, and infers the dtype through
// `dyn_cast<ShapedType>` — which a `!tile.fragment` is not, so it silently
// defaults to bf16. For the two-operand mma every existing fixture emits, that
// is correct and `nvwgmma_lowering.mlir` still checks exactly that call.
//
// For an mma that HAS an accumulator, the accumulator was discarded: a K-loop
// recomputed A x B from nothing each step and the GEMM returned the last
// partial product — rc=0, no diagnostic, a wrong number.
//
// Measured on merged main, this was NOT specific to the typed fragment form,
// which is why the guard keys on "has an accumulator" rather than "is typed".
// No fixture in the tree exercised either case, which is how it survived.
//
// ROCm already fails closed at the same seam; x86 and Apple leave `tile.mma`
// unlowered. NVIDIA was the only backend that dropped it silently.

// The LEGACY bare form with three data operands — exactly what
// `LowerKReductionAddToTileMMA` emits for the canonical K-step. This case
// predates the typed fragment work entirely.
func.func @legacy_k_step_accumulator(%a: tensor<64x64xbf16>,
                                     %b: tensor<64x64xbf16>,
                                     %c: tensor<64x64xf32>) -> tensor<64x64xf32> {
  // CHECK-LABEL: func.func @legacy_k_step_accumulator
  // CHECK: call @tessera_nvidia_wgmma_mma_async_bf16_m64n64k16(%arg0, %arg1, %arg2)
  %r = "tile.mma"(%a, %b, %c) {sm = 90 : i32, tessera.canonical_k_step}
      : (tensor<64x64xbf16>, tensor<64x64xbf16>, tensor<64x64xf32>)
      -> tensor<64x64xf32>
  return %r : tensor<64x64xf32>
}

// -----

!fa  = !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32",
                      role = "a", layout = "row_major", family = "auto">
!fb  = !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32",
                      role = "b", layout = "col_major", family = "auto">
!fac = !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32",
                      role = "acc", layout = "row_major", family = "auto">

// The typed form carries the same three-data-operand ABI.
func.func @typed_accumulator(%x: !fa, %y: !fb, %z: !fac) -> !fac {
  // CHECK-LABEL: func.func @typed_accumulator
  // CHECK: call @tessera_nvidia_wgmma_mma_async_bf16_m64n64k16(%arg0, %arg1, %arg2)
  %r = tile.mma %x, %y, %z : (!fa, !fb, !fac) -> !fac
  return %r : !fac
}

// -----

// Five raw operands, but only THREE data operands: `dataOperands()` filters
// tile control types before constructing the backend call (the #491 rule).
//
// The complementary case, a token-synced mma with only A and B, must still
// LOWER, or this guard would break every warp-specialized kernel. That one is
// covered by nvwgmma_lowering.mlir, whose emitted call is byte-identical before
// and after this change.
func.func @tokens_are_not_accumulators(%a: tensor<64x64xbf16>,
                                       %b: tensor<64x64xbf16>,
                                       %c: tensor<64x64xf32>,
                                       %t1: !tile.async_token,
                                       %t2: !tile.async_token)
    -> tensor<64x64xf32> {
  // CHECK-LABEL: func.func @tokens_are_not_accumulators
  // CHECK: call @tessera_nvidia_wgmma_mma_async_bf16_m64n64k16(%arg0, %arg1, %arg2)
  %r = "tile.mma"(%a, %b, %c, %t1, %t2) {sm = 90 : i32}
      : (tensor<64x64xbf16>, tensor<64x64xbf16>, tensor<64x64xf32>,
         !tile.async_token, !tile.async_token) -> tensor<64x64xf32>
  return %r : tensor<64x64xf32>
}
