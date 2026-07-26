// RUN: tessera-opt --tessera-tiling='tile-m=16 tile-n=16 tile-k=16' %s | FileCheck %s --check-prefix=T16
// RUN: tessera-opt --tessera-tiling='tile-m=32 tile-n=32 tile-k=32' %s | FileCheck %s --check-prefix=T32
// RUN: tessera-opt --tessera-tiling='tile-m=16 tile-n=16 tile-k=16' --tessera-tile-ir-lowering='sm=120' %s | FileCheck %s --check-prefix=TILE
// 2026-06: un-XFAIL'd — TilingPass now builds tensor.extract_slice/insert_slice
// with static OpFoldResult sizes (matching the static tile result type) instead
// of dynamic Values, which the MLIR-23 slice verifier rejected.

// (T16 path) matmul 64x128 @ 128x64 tiled into 16x16 blocks
// T16-LABEL: func.func @matmul_64x128x64
// T16:       arith.constant dense<0.000000e+00>
// T16:       scf.for
// T16:       scf.for
// T16:       tile.pipeline_init
// T16:       scf.for
// T16:       tensor.extract_slice
// T16:       tensor.extract_slice
// T16:       tessera.matmul
// T16-SAME:  tessera.canonical_k_step
// T16:       tessera.add
// T16:       tensor.insert_slice
// T16:       tile.pipeline_advance
// T16:       scf.yield
// T16:       scf.yield
// T16:       scf.yield
// T16-NOT:   tessera.matmul{{.*}}tensor<64x128xbf16>

// TILE-LABEL: func.func @matmul_64x128x64
// TILE:       scf.for
// TILE:       scf.for
// TILE:       tile.pipeline_init
// TILE:       scf.for
// TILE-SAME:  iter_args(
// TILE-SAME:  !tile.pipeline_state
// TILE:       tile.async_copy
// TILE-SAME:  tile.layout = #tile.layout
// TILE:       tile.async_copy
// TILE-SAME:  tile.layout = #tile.layout
// TILE:       tile.wait_async
// TILE:       tile.mma
// TILE-SAME:  tessera.canonical_k_step
// TILE:       tile.pipeline_advance
// TILE:       scf.yield
// TILE-NOT:   tessera.matmul
// TILE-NOT:   tessera.add

// (T32 path) same op, 32x32 tiles
// T32-LABEL: func.func @matmul_64x128x64
// T32:       scf.for
// T32:       tensor.extract_slice
// T32:       tessera.matmul

// A tile-sized M/N still receives the canonical K reduction loop.
// T16-LABEL: func.func @matmul_tile_sized
// T16:       scf.for
// T16:       scf.for
// T16:       scf.for
// T16:       tessera.matmul

// Ragged M/N/K are zero-padded to physical tiles and sliced back to the
// logical result. This makes every physical extract_slice in-bounds.
// T16-LABEL: func.func @matmul_ragged
// T16-DAG:   arith.constant dense<0.000000e+00> : tensor<32x48xbf16>
// T16-DAG:   arith.constant dense<0.000000e+00> : tensor<48x32xbf16>
// T16-DAG:   arith.constant dense<0.000000e+00> : tensor<32x32xf32>
// T16:       tensor.insert_slice %arg0 into
// T16:       tensor.insert_slice %arg1 into
// T16:       scf.for
// T16:       scf.for
// T16:       scf.for
// T16:       tensor.extract_slice
// T16:       tensor.extract_slice
// T16:       tensor.extract_slice
// T16:       return

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

  func.func @matmul_ragged(
      %A: tensor<17x35xbf16>,
      %B: tensor<35x19xbf16>
  ) -> tensor<17x19xf32> {
    %C = "tessera.matmul"(%A, %B) : (tensor<17x35xbf16>, tensor<35x19xbf16>)
                                    -> tensor<17x19xf32>
    return %C : tensor<17x19xf32>
  }
}
