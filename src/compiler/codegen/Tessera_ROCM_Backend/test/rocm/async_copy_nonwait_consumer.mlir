// RUN: not %trop --allow-unregistered-dialect --lower-rocm-async-copy %s 2>&1 | FileCheck %s

module {
  gpu.module @m {
    gpu.func @bad(%src: memref<?xf32>, %n: i64)
        workgroup(%lds: memref<256xf32, #gpu.address_space<workgroup>>) kernel {
      // CHECK: error: lower-rocm-async-copy: token has a non-wait consumer
      %tok = tessera_rocm.async_copy %lds, %src, %n
          : memref<256xf32, #gpu.address_space<workgroup>>, memref<?xf32>
            -> !tessera_rocm.token
      "test.consume"(%tok) : (!tessera_rocm.token) -> ()
      gpu.return
    }
  }
}
