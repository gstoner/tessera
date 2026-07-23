// x86 now defaults layout assignment on and consumes its physical row-major
// contract before tiling:
// RUN: tessera-opt %s -pass-pipeline='builtin.module(tessera-lower-to-x86)' | FileCheck %s --check-prefix=ON
//
// With assign-layouts=true the assignment half runs inside the named pipeline
// (just before LayoutLegalityPass, which verifies it), stamping tessera.layout:
// RUN: tessera-opt %s -pass-pipeline='builtin.module(tessera-lower-to-x86{assign-layouts=true})' | FileCheck %s --check-prefix=ON
//
// Phase 1 (front-to-back closure plan): LayoutAssignmentPass is wired into the
// named x86/GPU/CUDA-13 pipelines behind the `assign-layouts` option. It is
// off by default because the inserted same-type cast{layout} markers are not
// consumed by any backend yet, so the executing lowering path stays
// byte-identical; turning it on exercises the two-sided (assign + verify)
// layout contract end-to-end. See src/transforms/lib/Passes.cpp
// (TesseraLoweringPipelineOptions) and docs/audit/compiler/COMPILER_AUDIT.md.

// ON-LABEL: func.func @mm
// ON: tessera.matmul
// ON-SAME: tessera.layout = "row_major"
// ON: tessera.relu
// ON-SAME: tessera.layout = "row_major"
func.func @mm(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> tensor<4x16xf32> {
  %c = "tessera.matmul"(%a, %b) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  %r = "tessera.relu"(%c) : (tensor<4x16xf32>) -> tensor<4x16xf32>
  return %r : tensor<4x16xf32>
}
