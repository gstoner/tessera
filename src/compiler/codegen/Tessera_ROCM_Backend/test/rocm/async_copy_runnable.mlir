// RUN: %trop --lower-rocm-async-copy %s | FileCheck %s
//
// Runnable async_copy: tessera_rocm.async_copy lowers to a REAL cooperative
// global→LDS copy loop (not the artifact-only llvm.amdgcn.raw.buffer.copy.
// contract marker), and tessera_rocm.wait lowers to a real targeted wait plus
// gpu.barrier. The
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
      // CHECK: rocdl.s.waitcnt 0
      // CHECK: gpu.barrier
      // CHECK-NOT: tessera_rocm.async_copy
      // CHECK-NOT: tessera_rocm.wait
      // CHECK-NOT: contract
      %tok = tessera_rocm.async_copy %lds, %src, %n
          : memref<256xf32, #gpu.address_space<workgroup>>, memref<?xf32>
            -> !tessera_rocm.token
      tessera_rocm.wait %tok : !tessera_rocm.token
      // Pin the gfx1151 counter encodings as executable ROCDL operations.
      // vmcnt(1)=0x07f7; lgkmcnt(2)=0xfc27.
      // CHECK: rocdl.s.waitcnt 2039
      // CHECK: gpu.barrier
      %tok_vm = tessera_rocm.async_copy %lds, %src, %n
          : memref<256xf32, #gpu.address_space<workgroup>>, memref<?xf32>
            -> !tessera_rocm.token
      tessera_rocm.wait %tok_vm {counter = "vmcnt", threshold = 1 : i64}
          : !tessera_rocm.token
      // CHECK: rocdl.s.waitcnt 64551
      // CHECK: gpu.barrier
      %tok_lgkm = tessera_rocm.async_copy %lds, %src, %n
          : memref<256xf32, #gpu.address_space<workgroup>>, memref<?xf32>
            -> !tessera_rocm.token
      tessera_rocm.wait %tok_lgkm {counter = "lgkmcnt", threshold = 2 : i64}
          : !tessera_rocm.token
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
