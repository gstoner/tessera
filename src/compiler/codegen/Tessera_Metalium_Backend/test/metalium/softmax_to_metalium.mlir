// RUN: tessera-metalium-opt %s -pass-pipeline="tessera-metalium" | FileCheck %s
// REQUIRES: tessera_metalium_opt
//
// Sprint I-1 (2026-05-11): exercises the softmax lowering path through
// the `tessera_metalium` dialect.  Softmax decomposes to:
//   1. DMA DRAM→SRAM (load the row into on-core SRAM)
//   2. matmul-shaped reduction for max + sum (tile-local FMA over
//      vectors; Metalium has no dedicated reduce intrinsic, so we
//      lower through the matmul intrinsic with a 1xN×Nx1 contraction
//      against a broadcast identity vector)
//   3. elementwise exp + multiply by reciprocal sum
//   4. DMA SRAM→DRAM (store the row back)
//
// We check that:
//   - the row-tile DMA descriptor is emitted with the right shape
//   - a tile-local matmul tracks the row reduction
//   - element_size_bytes matches bf16 (2 bytes) for the canonical
//     reasoning-model storage dtype.

module {
  func.func @softmax_row(%src: memref<1x256xbf16, #tessera_metalium.memspace<"dram">>,
                         %dst: memref<1x256xbf16, #tessera_metalium.memspace<"dram">>) {
    "tessera.tile.softmax"(%src, %dst) {
      axis = -1 : i64,
      tile_m = 1 : i64,
      tile_n = 256 : i64
    } : (memref<1x256xbf16, #tessera_metalium.memspace<"dram">>,
         memref<1x256xbf16, #tessera_metalium.memspace<"dram">>) -> ()
    return
  }
}

// CHECK: tessera_metalium.dma
// CHECK-SAME: direction = "dram_to_sram"
// CHECK-SAME: element_size_bytes = 2
// CHECK: tessera_metalium.matmul
// CHECK-SAME: tile_shape
// CHECK: tessera_metalium.dma
// CHECK-SAME: direction = "sram_to_dram"
