module attributes {gpu.container_module, tessera.arch = "gfx1201"} {
  gpu.module @probe {
    gpu.func @fp8_cast(%arg0: memref<?xf32>, %arg1: memref<?xf8E4M3FN>, %arg2: index) kernel {
      %thread_id_x = gpu.thread_id x
      %block_id_x = gpu.block_id x
      %c256 = arith.constant 256 : index
      %0 = arith.muli %block_id_x, %c256 : index
      %1 = arith.addi %0, %thread_id_x : index
      %2 = arith.cmpi ult, %1, %arg2 : index
      scf.if %2 {
        %3 = memref.load %arg0[%1] : memref<?xf32>
        %c0_i32 = arith.constant 0 : i32
        %4 = rocdl.cvt.pk.fp8.f32 %3, %3 -> %c0_i32[false] : i32
        %5 = arith.trunci %4 : i32 to i8
        %6 = arith.bitcast %5 : i8 to f8E4M3FN
        memref.store %6, %arg1[%1] : memref<?xf8E4M3FN>
      }
      gpu.return
    }
  }
}

