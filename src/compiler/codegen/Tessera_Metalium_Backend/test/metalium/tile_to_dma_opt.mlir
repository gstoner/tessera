\
// RUN: tessera-metalium-opt %s -pass-pipeline="tessera-metalium" | FileCheck %s
// REQUIRES: tessera_metalium_opt

module {
  func.func @copy(%src: memref<64x64xf16, #tessera_metalium.memspace<"dram">>,
                  %dst: memref<64x64xf16, #tessera_metalium.memspace<"sram">>) {
    "tessera.tile.copy"(%src, %dst) : (memref<64x64xf16, #tessera_metalium.memspace<"dram">>,
                                       memref<64x64xf16, #tessera_metalium.memspace<"sram">>) -> ()
    return
  }
}

// CHECK: tessera_metalium.dma
// CHECK-SAME: direction = "dram_to_sram"
// CHECK-SAME: element_size_bytes = 2
// CHECK-SAME: [64, 64]
