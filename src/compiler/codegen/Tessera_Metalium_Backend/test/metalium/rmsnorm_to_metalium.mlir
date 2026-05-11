// RUN: tessera-metalium-opt %s -pass-pipeline="tessera-metalium" | FileCheck %s
// REQUIRES: tessera_metalium_opt
//
// Sprint I-1 (2026-05-11): RMSNorm row reduction lowered to
// `tessera_metalium.dma + matmul`.  Simpler than LayerNorm — no mean
// subtraction, just the RMS scale:
//
//   y[i] = x[i] · γ[i] / sqrt((1/N) · Σ x[i]² + eps)
//
// One reduction (Σ x²) → one tile-local matmul.

module {
  func.func @rmsnorm_row(%src:   memref<1x512xbf16, #tessera_metalium.memspace<"dram">>,
                         %gamma: memref<1x512xbf16, #tessera_metalium.memspace<"dram">>,
                         %dst:   memref<1x512xbf16, #tessera_metalium.memspace<"dram">>) {
    "tessera.tile.rmsnorm"(%src, %gamma, %dst) {
      axis = -1 : i64,
      tile_m = 1 : i64,
      tile_n = 512 : i64,
      eps = 1.000000e-06 : f32
    } : (memref<1x512xbf16, #tessera_metalium.memspace<"dram">>,
         memref<1x512xbf16, #tessera_metalium.memspace<"dram">>,
         memref<1x512xbf16, #tessera_metalium.memspace<"dram">>) -> ()
    return
  }
}

// Load x and γ into SRAM:
// CHECK-COUNT-2: tessera_metalium.dma
// CHECK-SAME: direction = "dram_to_sram"
// CHECK-SAME: element_size_bytes = 2
//
// One reduction Σx²:
// CHECK: tessera_metalium.matmul
//
// Store y back to DRAM:
// CHECK: tessera_metalium.dma
// CHECK-SAME: direction = "sram_to_dram"
