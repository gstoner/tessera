// RUN: %trop %s | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: not %trop --lower-tessera-target-to-rocdl %s 2>&1 | FileCheck %s --check-prefix=STRICT
//
// B3 — tessera_rocm.buffer_load (AMD buffer addressing + native OOB) and
// tessera_rocm.ds_read_tr (transposing LDS read) as hardware-free Target-IR
// ops. ROUNDTRIP proves they parse/verify/print. They deliberately fail closed
// at executable lowering until physical buffer/LDS consumers land.

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

// STRICT: ROCm target operation has no executable ROCDL lowering
// STRICT-NOT: .contract
