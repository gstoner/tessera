module attributes {gpu.container_module} {
  gpu.module @scale_rounding {
    gpu.func @scale_rounding(%a: !llvm.ptr<1>, %b: !llvm.ptr<1>, %out: !llvm.ptr<1>) kernel attributes {known_block_size = array<i32: 64, 1, 1>} {
      %tid = gpu.thread_id x
      %index = arith.index_cast %tid : index to i64
      %pa = llvm.getelementptr %a[%index] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %pb = llvm.getelementptr %b[%index] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %po = llvm.getelementptr %out[%index] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
      %x = llvm.load %pa : !llvm.ptr<1> -> f64
      %y = llvm.load %pb : !llvm.ptr<1> -> f64
      %product = arith.mulf %x, %y : f64
      %narrow = arith.truncf %product : f64 to f32
      %bits = arith.bitcast %narrow : f32 to i32
      llvm.store %bits, %po : i32, !llvm.ptr<1>
      gpu.return
    }
  }
}
