// RUN: %trop --lower-rocm-async-copy %s | FileCheck %s
//
// Runnable async_copy: tessera_rocm.async_copy lowers to a REAL cooperative
// global→LDS copy loop (not the artifact-only llvm.amdgcn.raw.buffer.copy.
// contract marker), and tessera_rocm.wait lowers to gpu.barrier. The
// memref.load/store then lower to real global_load / ds_store through the
// standard gpu.module → ROCDL path.

module {
  gpu.module @m {
    // CHECK-LABEL: gpu.func @copy_demo
    gpu.func @copy_demo(%src: memref<?xf32>, %out: memref<?xf32>, %n: i64)
        workgroup(%lds: memref<256xf32, #gpu.address_space<workgroup>>) kernel {
      // CHECK: scf.for
      // CHECK: memref.load %{{.*}} : memref<?xf32>
      // CHECK: memref.store %{{.*}} : memref<256xf32, #gpu.address_space<workgroup>>
      // CHECK: gpu.barrier
      // CHECK-NOT: tessera_rocm.async_copy
      // CHECK-NOT: tessera_rocm.wait
      // CHECK-NOT: contract
      %tok = tessera_rocm.async_copy %lds, %src, %n
          : memref<256xf32, #gpu.address_space<workgroup>>, memref<?xf32>
            -> !tessera_rocm.token
      tessera_rocm.wait %tok : !tessera_rocm.token
      // read the staged LDS tile back out (proves the copy landed).
      %tid = gpu.thread_id x
      %bdim = gpu.block_dim x
      %ni = arith.index_cast %n : i64 to index
      scf.for %i = %tid to %ni step %bdim {
        %v = memref.load %lds[%i]
            : memref<256xf32, #gpu.address_space<workgroup>>
        memref.store %v, %out[%i] : memref<?xf32>
      }
      gpu.return
    }
  }
}
