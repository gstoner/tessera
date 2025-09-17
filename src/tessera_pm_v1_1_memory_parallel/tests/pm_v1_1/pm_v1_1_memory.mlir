// RUN: tessera-opt %s -pm-v1_1-verify | FileCheck %s

module {
  // CHECK-LABEL: tile.alloc_shared
  %buf = tile.alloc_shared : memref<64x65xbf16, 3> { swizzle = "xor", bank_pad = 1 }

  // CHECK: tile.async_copy
  tile.async_copy %src, %buf { stage = 0, vector = 16 } : (memref<?xf16,1>, memref<64x65xbf16,3>) -> ()
  // CHECK: tile.wait_async
  tile.wait_async { stage = 0 }

  // Deterministic reduction
  // CHECK: tile.reduce {{.*}} order = "tree"
  %r = tile.reduce %x { op = "sum", order = "tree" } : (vector<128xf32>) -> f32
}
