module attributes {tessera.arch = "gfx1201", tessera.frontend.authority = "tracer", tessera.ir.version = "1.0", tessera.target = "rocm"} {
  func.func @normalized(%arg0: tensor<1024x8x128xf32>) -> tensor<1024x8x128xf32> attributes {tessera.frontend.authority = "tracer", tessera.structured_cfg.blocks = 2 : i64, tessera.structured_cfg.digest = "4f06b08357b7ef2800d4287f2daacefa4ec19fb2ca2ecf11cbeae13330757ed6", tessera.structured_cfg.schema = "tessera.structured_cfg.v1"} {
    %0 = bufferization.to_buffer %arg0 : tensor<1024x8x128xf32> to memref<1024x8x128xf32>
    %intptr = memref.extract_aligned_pointer_as_index %0 : memref<1024x8x128xf32> -> index
    %1 = arith.index_cast %intptr : index to i64
    %2 = llvm.inttoptr %1 : i64 to !llvm.ptr
    %alloc = memref.alloc() : memref<1024x8x128xf32>
    %intptr_0 = memref.extract_aligned_pointer_as_index %alloc : memref<1024x8x128xf32> -> index
    %3 = arith.index_cast %intptr_0 : index to i64
    %4 = llvm.inttoptr %3 : i64 to !llvm.ptr
    %c8192_i64 = arith.constant 8192 : i64
    %c128_i64 = arith.constant 128 : i64
    tile.softmax_kernel %2, %4, %c8192_i64, %c128_i64 {accum = "f32", axis = -1 : i64, exp_mode = "accurate", ftz = false, storage = "f32", tessera.schedule_hash = "f9686bbac2a053bc712b289f962fb7ee9d0dfd15be6d5ebac1aa1f76f0faea3a", tessera.workgroup_size = 256 : i64} : !llvm.ptr, !llvm.ptr, i64, i64
    %5 = bufferization.to_tensor %alloc : memref<1024x8x128xf32> to tensor<1024x8x128xf32>
    return %5 : tensor<1024x8x128xf32>
  }
}

