module attributes {tessera.arch = "gfx1201", tessera.frontend.authority = "tracer", tessera.ir.version = "1.0", tessera.target = "rocm"} {
  func.func @normalized(%arg0: tensor<5x4x32xf32>) -> tensor<5x4x32xf32> attributes {tessera.frontend.authority = "tracer", tessera.structured_cfg.blocks = 2 : i64, tessera.structured_cfg.digest = "4f06b08357b7ef2800d4287f2daacefa4ec19fb2ca2ecf11cbeae13330757ed6", tessera.structured_cfg.schema = "tessera.structured_cfg.v1"} {
    %0 = bufferization.to_buffer %arg0 : tensor<5x4x32xf32> to memref<5x4x32xf32>
    %intptr = memref.extract_aligned_pointer_as_index %0 : memref<5x4x32xf32> -> index
    %1 = arith.index_cast %intptr : index to i64
    %2 = llvm.inttoptr %1 : i64 to !llvm.ptr
    %alloc = memref.alloc() : memref<5x4x32xf32>
    %intptr_0 = memref.extract_aligned_pointer_as_index %alloc : memref<5x4x32xf32> -> index
    %3 = arith.index_cast %intptr_0 : index to i64
    %4 = llvm.inttoptr %3 : i64 to !llvm.ptr
    %c20_i64 = arith.constant 20 : i64
    %c32_i64 = arith.constant 32 : i64
    tile.softmax_kernel %2, %4, %c20_i64, %c32_i64 {accum = "f32", axis = -1 : i64, exp_mode = "accurate", ftz = false, storage = "f32", tessera.schedule_hash = "45508c6fdb8335883e40672da408026c9c1b1eb2aafd7c06327ed309bf17be3a", tessera.workgroup_size = 256 : i64} : !llvm.ptr, !llvm.ptr, i64, i64
    %5 = bufferization.to_tensor %alloc : memref<5x4x32xf32> to tensor<5x4x32xf32>
    return %5 : tensor<5x4x32xf32>
  }
}

