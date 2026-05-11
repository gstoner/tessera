// RUN: tessera-opt --tessera-lower-to-gpu --gpu-target=sm_90a %s | FileCheck %s
// REQUIRES: tessera_opt_built
//
// Sprint G-4 — Fused SwiGLU MLP: matmul → silu·mul → matmul.
// Three matmuls collapsed into a two-WGMMA-pass kernel:
//   tmp = silu(x @ W_gate) * (x @ W_up)
//   y   = tmp @ W_down
// First WGMMA computes both gate + up branches in one tile (two
// accumulators); the silu·mul happens in registers; the second WGMMA
// projects through W_down.

module attributes {tessera.target = "nvidia_sm90"} {
  func.func @swiglu_mlp(
      %X : memref<64x256xbf16, 3>,
      %W_gate : memref<256x1024xbf16, 3>,
      %W_up : memref<256x1024xbf16, 3>,
      %W_down : memref<1024x256xbf16, 3>,
      %Y : memref<64x256xbf16, 3>) {
    "tessera.tile.swiglu_mlp"(%X, %W_gate, %W_up, %W_down, %Y) {
      tile_m = 64 : i64,
      tile_n = 256 : i64,
      tile_k = 16 : i64,
      hidden_dim = 1024 : i64,
      cluster = array<i64: 2, 1, 1>,
      acc_dtype = "fp32",
      cuda_arch_min = "sm_90a"
    } : (memref<64x256xbf16, 3>,
         memref<256x1024xbf16, 3>,
         memref<256x1024xbf16, 3>,
         memref<1024x256xbf16, 3>,
         memref<64x256xbf16, 3>) -> ()
    return
  }
}

// CHECK: tessera.tile.swiglu_mlp
//
// Two WGMMA passes (gate+up combined, then down):
// CHECK-COUNT-2: wgmma.mma_async.sync.aligned
//
// Cluster for producer/consumer warp staging:
// CHECK-DAG: cluster
