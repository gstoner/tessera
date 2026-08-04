// RUN: tessera-opt %s --tessera-record-metadata --tessera-tile-ir-lowering \
// RUN:   --tessera-verify-metadata-obligation | FileCheck %s
//
// W1.3 / Decision #32 — the metadata lowering obligation, checked on the REAL
// Graph IR -> Tile IR boundary rather than on a synthetic pair of passes.
//
// This fixture is the regression test for a defect the verifier found the first
// time it was pointed at a real program. `TileIRLoweringPass` has two patterns
// that produce `tile.mma`:
//
//   * `LowerKReductionAddToTileMMA` (the fused K-step) forwarded
//     `numeric_policy`, and always had.
//   * `LowerMatmulToTileMMA` (the MAIN matmul producer) did not.
//
// So `numeric_policy = {storage = "bf16", accum = "fp32"}` was stated at Graph
// IR, validated there by `IRContractLegalityPass`'s
// DTYPE_LEGALITY_LOWP_WITHOUT_WIDE_ACCUM rule, and then ceased to exist one
// level down — codegen chose an MMA instruction with no record of the required
// accumulator width. That is Decision #32's motivating example, and it was live
// in this tree.
//
// The RUN line is the whole point of the two-pass design: record, lower, verify
// in ONE invocation. A `PassInstrumentation` straddling runBeforePass /
// runAfterPass would be the more obvious MLIR idiom and could not be written as
// a fixture at all.

// CHECK-LABEL: func.func @gemm
func.func @gemm(%a: tensor<128x256xbf16>, %b: tensor<256x128xbf16>)
    -> tensor<128x128xf32> {
  // The accumulator contract must still be readable on the Tile op. Checking
  // for it HERE, on `tile.mma`, is what makes this a test of the boundary and
  // not merely of the verifier: if the attribute stopped being forwarded, the
  // verify pass would fail the run AND this line would stop matching.
  // CHECK: tile.mma
  // CHECK-SAME: numeric_policy = {accum = "fp32", storage = "bf16"}
  %0 = "tessera.matmul"(%a, %b) {
    numeric_policy = {storage = "bf16", accum = "fp32"}, layout = "row_major"
  } : (tensor<128x256xbf16>, tensor<256x128xbf16>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// The mechanism's own cases -- a declared drop being ACCEPTED, and the four
// rejections -- need a controlled drop, which no pass in this pipeline performs
// now that the bug above is fixed. They hand-write the snapshot instead and run
// the verifier alone: `metadata_obligation_declared.mlir` and
// `metadata_obligation_invalid.mlir`.
