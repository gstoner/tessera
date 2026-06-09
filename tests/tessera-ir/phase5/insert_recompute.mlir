// RUN: tessera-opt --tessera-insert-recompute="memory-budget-mb=1" --allow-unregistered-dialect %s | FileCheck %s

// Test: InsertRecomputePass tags pure ops with tessera_sr.checkpoint /
// tessera_sr.checkpoint_id whenever the accumulated live-tensor memory exceeds
// the budget, and records the total on the module.
//
// 2026-06: un-XFAIL'd.  The fixture moved to value-semantics tensor matmul
// (the MLIR-22 TesseraMatmulOp verifier now requires one tensor result) and
// adds --allow-unregistered-dialect so the emitted tessera_sr.* markers (a
// separate, intentionally-unregistered annotation dialect) round-trip.

// CHECK: tessera_sr.num_checkpoints
module {
  // CHECK-LABEL: func.func @train_step
  func.func @train_step(%x: tensor<1024x1024xbf16>, %w: tensor<1024x1024xbf16>)
      -> tensor<1024x1024xbf16> {

    // These matmuls accumulate live tensors.  With a 1 MiB budget and each
    // output being 2 MiB (1024*1024*2 bytes), the pass instruments them with
    // checkpoint markers.

    // CHECK: tessera.matmul
    // CHECK-SAME: tessera_sr.checkpoint
    // CHECK-SAME: tessera_sr.checkpoint_id
    %a = "tessera.matmul"(%x, %w) {tessera.effect = "pure"} :
        (tensor<1024x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16>
    %b = "tessera.matmul"(%a, %w) {tessera.effect = "pure"} :
        (tensor<1024x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16>
    %c = "tessera.matmul"(%b, %w) {tessera.effect = "pure"} :
        (tensor<1024x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16>

    return %c : tensor<1024x1024xbf16>
  }
}
