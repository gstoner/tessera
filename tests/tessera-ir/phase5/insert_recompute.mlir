// RUN: tessera-opt --tessera-insert-recompute="memory-budget-mb=1" %s | FileCheck %s

// Test: InsertRecomputePass inserts tessera_sr.checkpoint markers whenever
// the accumulated live-tensor memory exceeds the budget, and tags pure ops
// between checkpoints with tessera_sr.recompute_hint = "recomputable".

module {
  // CHECK-LABEL: func.func @train_step
  func.func @train_step(%x: memref<1024x1024xbf16>) -> memref<1024x1024xbf16> {

    // These matmuls accumulate live tensors.  With a 1 MiB budget and each
    // output being 2 MiB (1024*1024*2 bytes), the first op should trigger a
    // checkpoint on or after the second result.

    %a = memref.alloc() : memref<1024x1024xbf16>
    "tessera.matmul"(%x, %a) {tessera.effect = "pure"} :
        (memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()

    // CHECK: tessera_sr.checkpoint
    // CHECK: tessera_sr.recompute_hint
    %b = memref.alloc() : memref<1024x1024xbf16>
    "tessera.matmul"(%a, %b) {tessera.effect = "pure"} :
        (memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()

    %c = memref.alloc() : memref<1024x1024xbf16>
    "tessera.matmul"(%b, %c) {tessera.effect = "pure"} :
        (memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()

    return %c : memref<1024x1024xbf16>
  }
}
