
// RUN: mlir-opt %s -linalg-tile="tile-sizes=128,128,16" -linalg-bufferize -convert-linalg-to-loops -lower-affine | mlir-opt
// Minimal MLIR demonstrating tiling+bufferization on a GEMM-like op.
module {
  func.func @gemm(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) {
    linalg.matmul ins(%A, %B : memref<1024x1024xf32>, memref<1024x1024xf32>)
                  outs(%C : memref<1024x1024xf32>)
    return
  }
}
