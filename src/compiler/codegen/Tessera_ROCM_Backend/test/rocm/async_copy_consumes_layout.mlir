// RUN: %trop --lower-rocm-async-copy %s | FileCheck %s

module {
  gpu.module @m {
    gpu.func @layout_copy(%src: memref<?xf32>, %n: i64)
        workgroup(%lds: memref<256xf32, #gpu.address_space<workgroup>>) kernel {
      // The logical flat index is delinearized as [8,4], then rebuilt with
      // strides [16,2] and offset 3 before the LDS store.
      // CHECK-LABEL: gpu.func @layout_copy
      // CHECK: arith.divui
      // CHECK: arith.remui
      // CHECK: arith.muli
      // CHECK: arith.addi
      // CHECK: memref.store %{{.*}}, %{{.*}}[%{{.*}}]
      %tok = tessera_rocm.async_copy %lds, %src, %n {
        tile.layout = #tile.layout<shard = [8, 4] : [16, 2] on ["lds", "lds"], replica = [] : [] on [], offset = 3>
      } : memref<256xf32, #gpu.address_space<workgroup>>, memref<?xf32>
          -> !tessera_rocm.token
      tessera_rocm.wait %tok : !tessera_rocm.token
      gpu.return
    }
  }
}
