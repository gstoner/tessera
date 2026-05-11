// RUN: tessera-metalium-opt %s -pass-pipeline="tessera-metalium" | FileCheck %s
// REQUIRES: tessera_metalium_opt
//
// Sprint I-1 (2026-05-11): layer_norm row reduction lowered to
// `tessera_metalium.dma + matmul` chain.  Layer norm decomposes as:
//
//   μ      = (1/N) · Σ x[i]
//   σ²     = (1/N) · Σ (x[i] - μ)²
//   y[i]   = (x[i] - μ) / sqrt(σ² + eps)   * γ[i] + β[i]
//
// The two Σ reductions each lower to a tile-local matmul (1xN × Nx1
// against an identity vector); the elementwise compose runs in SRAM
// between the DRAM→SRAM load and the SRAM→DRAM store.

module {
  func.func @layer_norm_row(%src:   memref<1x512xbf16, #tessera_metalium.memspace<"dram">>,
                            %gamma: memref<1x512xbf16, #tessera_metalium.memspace<"dram">>,
                            %beta:  memref<1x512xbf16, #tessera_metalium.memspace<"dram">>,
                            %dst:   memref<1x512xbf16, #tessera_metalium.memspace<"dram">>) {
    "tessera.tile.layer_norm"(%src, %gamma, %beta, %dst) {
      axis = -1 : i64,
      tile_m = 1 : i64,
      tile_n = 512 : i64,
      eps = 1.000000e-05 : f32
    } : (memref<1x512xbf16, #tessera_metalium.memspace<"dram">>,
         memref<1x512xbf16, #tessera_metalium.memspace<"dram">>,
         memref<1x512xbf16, #tessera_metalium.memspace<"dram">>,
         memref<1x512xbf16, #tessera_metalium.memspace<"dram">>) -> ()
    return
  }
}

// Load x, γ, β into SRAM:
// CHECK-COUNT-3: tessera_metalium.dma
// CHECK-SAME: direction = "dram_to_sram"
// CHECK-SAME: element_size_bytes = 2
//
// Two reductions: Σx for μ, Σ(x-μ)² for σ²
// CHECK-COUNT-2: tessera_metalium.matmul
//
// Store y back to DRAM:
// CHECK: tessera_metalium.dma
// CHECK-SAME: direction = "sram_to_dram"
