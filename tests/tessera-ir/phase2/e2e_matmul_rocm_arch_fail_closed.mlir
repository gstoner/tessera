// RUN: not tessera-opt --tessera-graph-to-schedule --split-input-file %s 2>&1 | FileCheck %s --check-prefix=ARCH --implicit-check-not=schedule.matmul

// The shared Schedule/Tile vocabulary is architecture-neutral, but the
// physical schedule is not.  Future RDNA targets must provide their own
// instruction and macro-tile profiles rather than inheriting gfx1151 values.

module attributes {tessera.target = "rocm", tessera.arch = "gfx1200"} {
  func.func @gfx1200_requires_profile(%a: tensor<32x32xf16>,
      %b: tensor<32x32xf16>) -> tensor<32x32xf32> {
    // ARCH: E2E-REAL-2 Graph->Schedule requires static rank-2
    %0 = tessera.matmul %a, %b
        : (tensor<32x32xf16>, tensor<32x32xf16>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}

// -----

module attributes {tessera.target = "rocm", tessera.arch = "gfx1250"} {
  func.func @gfx1250_requires_profile(%a: tensor<32x32xf16>,
      %b: tensor<32x32xf16>) -> tensor<32x32xf32> {
    // ARCH: E2E-REAL-2 Graph->Schedule requires static rank-2
    %0 = tessera.matmul %a, %b
        : (tensor<32x32xf16>, tensor<32x32xf16>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}
