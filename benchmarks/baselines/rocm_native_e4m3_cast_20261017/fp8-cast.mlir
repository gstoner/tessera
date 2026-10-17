module attributes {gpu.container_module} {
 gpu.module @probe {
  gpu.func @fp8_cast(%input: memref<?xf32>, %codes: memref<?xf8E4M3FN>, %count: index) kernel {
   %tid = gpu.thread_id x
   %bid = gpu.block_id x
   %c256 = arith.constant 256 : index
   %base = arith.muli %bid, %c256 : index
   %index = arith.addi %base, %tid : index
   %active = arith.cmpi ult, %index, %count : index
   scf.if %active {
    %value = memref.load %input[%index] : memref<?xf32>
    %code = arith.truncf %value : f32 to f8E4M3FN
    memref.store %code, %codes[%index] : memref<?xf8E4M3FN>
   }
   gpu.return
  }
 }
}
