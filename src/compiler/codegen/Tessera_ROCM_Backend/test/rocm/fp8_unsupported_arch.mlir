// RUN: not %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1151})' %s 2>&1 | FileCheck %s
//
// A6 / B4 (Decision #21) — FP8 matmul on an arch with no FP8 matrix path is a
// hard, named error, never a silent flavor guess.  gfx1151 (RDNA 3.5 / Strix
// Halo) has NO FP8 WMMA instruction (the load-bearing distinction from gfx1200).

module {
  func.func @fp8_mma(%a: f8E4M3FN, %b: f8E4M3FN) -> f8E4M3FN {
    %m = "tile.mma"(%a, %b) : (f8E4M3FN, f8E4M3FN) -> f8E4M3FN
    return %m : f8E4M3FN
  }
}

// CHECK: error: ROCm lowering: FP8 matmul requested on arch 'gfx1151' which has no FP8 matrix path
