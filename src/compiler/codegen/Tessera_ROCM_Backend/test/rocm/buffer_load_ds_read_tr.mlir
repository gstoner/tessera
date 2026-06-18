// RUN: %trop %s | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: %trop --lower-tessera-target-to-rocdl %s | FileCheck %s --check-prefix=ROCDL
//
// B3 — tessera_rocm.buffer_load (AMD buffer addressing + native OOB) and
// tessera_rocm.ds_read_tr (transposing LDS read) as hardware-free Target-IR
// ops.  ROUNDTRIP proves they parse/verify/print; ROCDL proves they lower to
// the artifact marker contract (and leave no tessera_rocm.* op behind).

module {
  func.func @k(%base: !llvm.ptr, %off: i32, %lds: !llvm.ptr) -> f32 {
    %v = tessera_rocm.buffer_load %base, %off {oob} : !llvm.ptr, i32 -> f32
    %t = tessera_rocm.ds_read_tr %lds : !llvm.ptr -> f32
    return %v : f32
  }
}

// ROUNDTRIP: tessera_rocm.buffer_load
// ROUNDTRIP-SAME: oob
// ROUNDTRIP: tessera_rocm.ds_read_tr

// ROCDL-DAG: llvm.call @llvm.amdgcn.raw.buffer.load.contract
// ROCDL-DAG: llvm.call @llvm.amdgcn.ds.read.tr.contract
// ROCDL-NOT: tessera_rocm.
