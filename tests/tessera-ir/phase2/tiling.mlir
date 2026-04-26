// RUN: tessera-opt --tessera-tiling='tile-m=16 tile-n=16' %s | FileCheck %s --check-prefix=T16
// RUN: tessera-opt --tessera-tiling='tile-m=32 tile-n=32' %s | FileCheck %s --check-prefix=T32

// ── T16: matmul 64x128 @ 128x64 tiled into 16x16 blocks ──────────────────
// T16-LABEL: func.func @matmul_64x128x64
// T16:       tensor.empty
// T16:       scf.for
// T16:       scf.for
// T16:       tensor.extract_slice
// T16:       tensor.extract_slice
// T16:       tessera.matmul
// T16:       tensor.insert_slice
// T16:       scf.yield
// T16:       scf.yield
// T16-NOT:   tessera.matmul{{.*}}tensor<64x128xbf16>

// ── T32: same op, 32x32 tiles ─────────────────────────────────────────────
// T32-LABEL: func.func @matmul_64x128x64
// T32:       scf.for
// T32:       tensor.extract_slice
// T32:       tessera.matmul

// ── Already tile-sized op is left alone ───────────────────────────────────
// T16-LABEL: func.func @matmul_tile_sized
// T16-NOT:   scf.for
// T16:       tessera.matmul

module attributes {tessera.ir.version = "1.0"} {

  func.func @matmul_64x128x64(
      %A: tensor<64x128xbf16>,
      %B: tensor<128x64xbf16>
  ) -> tensor<64x64xf32> {
    %C = "tessera.matmul"(%A, %B) : (tensor<64x128xbf16>, tensor<128x64xbf16>)
                                    -> tensor<64x64xf32>
    return %C : tensor<64x64xf32>
  }

  // Tile-sized input: 16x32 @ 32x16 with tile-m=16 tile-n=16 → no wrapping.
  func.func @matmul_tile_sized(
      %A: tensor<16x32xbf16>,
      %B: tensor<32x16xbf16>
  ) -> tensor<16x16xf32> {
    %C = "tessera.matmul"(%A, %B) : (tensor<16x32xbf16>, tensor<32x16xbf16>)
                                    -> tensor<16x16xf32>
    return %C : tensor<16x16xf32>
  }
}
