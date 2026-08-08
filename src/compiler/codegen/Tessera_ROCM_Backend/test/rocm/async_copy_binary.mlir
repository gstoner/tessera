// REQUIRES: rocm-device-libs
// RUN: %trop --pass-pipeline='builtin.module(tessera-rocm-executable{family=softmax input=tile output=binary arch=gfx1151})' %s | FileCheck %s
//
// Final packaging is a separate evidence layer from host-free Target/ROCDL
// lowering: it runs only when a real ROCm SDK supplies AMD device libraries.

module {
  gpu.module @m {
    gpu.func @copy_demo(%src: memref<?xf32>, %out: memref<?xf32>, %n: i64)
        workgroup(%lds: memref<256xf32, #gpu.address_space<workgroup>>) kernel {
      %tok = tessera_rocm.async_copy %lds, %src, %n
          : memref<256xf32, #gpu.address_space<workgroup>>, memref<?xf32>
            -> !tessera_rocm.token
      tessera_rocm.wait %tok : !tessera_rocm.token
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

// CHECK: gpu.binary
