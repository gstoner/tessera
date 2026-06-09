// RUN: tessera-opt %s | FileCheck %s

// Step 3: the DeepGEMM-inspired grouped-layout contract is a first-class
// Graph-IR op + attributes (tessera.grouped_gemm), so it round-trips and is
// verifiable at the IR level — not just runtime kwargs.  The IR op is
// target-agnostic (all four families are valid IR); the Apple runtime gate
// rejects the families it can't execute (masked / k_grouped) separately.

// CHECK-LABEL: func.func @moe_grouped
// CHECK:       tessera.grouped_gemm
// CHECK-SAME:  grouped_alignment = 128 : i64
// CHECK-SAME:  grouped_kind = "masked"
// CHECK-SAME:  quant = "nvfp4"
func.func @moe_grouped(%x: tensor<256x64xbf16>, %w: tensor<3x64x16xbf16>,
                       %gs: tensor<3xi64>) -> tensor<256x16xf32> {
  %o = tessera.grouped_gemm %x, %w, %gs {
         grouped_kind = "masked", grouped_alignment = 128 : i64, quant = "nvfp4"
       } : (tensor<256x64xbf16>, tensor<3x64x16xbf16>, tensor<3xi64>)
             -> tensor<256x16xf32>
  return %o : tensor<256x16xf32>
}

// Default grouped_kind ("contiguous") is elided on print; alignment/quant
// are optional and omitted here.
// CHECK-LABEL: func.func @moe_default
// CHECK:       tessera.grouped_gemm
func.func @moe_default(%x: tensor<12x8xbf16>, %w: tensor<3x8x6xbf16>,
                       %gs: tensor<3xi64>) -> tensor<12x6xf32> {
  %o = tessera.grouped_gemm %x, %w, %gs
       : (tensor<12x8xbf16>, tensor<3x8x6xbf16>, tensor<3xi64>) -> tensor<12x6xf32>
  return %o : tensor<12x6xf32>
}
