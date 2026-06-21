// RUN: %trop --lower-tessera-target-to-rocdl %s | FileCheck %s

// Decoupled waits (moonmath CDNA3 attention writeup, §"Decoupled Waits"): the
// `counter` attribute on tessera_rocm.wait selects a targeted AMDGCN wait
// counter — vmcnt for vector-memory (a global→LDS copy), lgkmcnt for an LDS
// read — instead of draining the wavefront with a full s_barrier.  A wait with
// no counter remains a true synchronization point (s_barrier).

module {
  func.func @waits(%dst: !llvm.ptr, %src: !llvm.ptr, %bytes: i64) {
    %t1 = tessera_rocm.async_copy %dst, %src, %bytes {arch = "gfx942", dst_space = "lds", src_space = "global"} : !llvm.ptr, !llvm.ptr -> !tessera_rocm.token
    tessera_rocm.wait %t1 {counter = "vmcnt"} : !tessera_rocm.token

    %t2 = tessera_rocm.async_copy %dst, %src, %bytes {arch = "gfx942", dst_space = "lds", src_space = "global"} : !llvm.ptr, !llvm.ptr -> !tessera_rocm.token
    tessera_rocm.wait %t2 {counter = "lgkmcnt"} : !tessera_rocm.token

    %t3 = tessera_rocm.async_copy %dst, %src, %bytes {arch = "gfx942", dst_space = "lds", src_space = "global"} : !llvm.ptr, !llvm.ptr -> !tessera_rocm.token
    tessera_rocm.wait %t3 : !tessera_rocm.token
    return
  }
}

// All three wait flavors lower to distinct markers; CHECK-DAG is order-robust.
// CHECK-DAG: llvm.call @llvm.amdgcn.s.waitcnt.vmcnt.contract
// CHECK-DAG: llvm.call @llvm.amdgcn.s.waitcnt.lgkmcnt.contract
// CHECK-DAG: llvm.call @llvm.amdgcn.s.barrier.contract
// CHECK-NOT: tessera_rocm.wait
