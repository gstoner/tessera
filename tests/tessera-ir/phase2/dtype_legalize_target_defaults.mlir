// RUN: tessera-opt --allow-unregistered-dialect --pass-pipeline='builtin.module(tessera-lower-to-x86)' %s | FileCheck %s --check-prefix=X86
// RUN: tessera-opt --allow-unregistered-dialect --pass-pipeline='builtin.module(tessera-lower-to-gpu)' %s | FileCheck %s --check-prefix=NVIDIA
//
// CORE-COMPILER-2: target defaults are capability-driven. x86 and NVIDIA run
// compute legalization but leave terminal packing disabled until they own a
// consumer. The ROCm-owned full compute -> storage -> descriptor-consume test
// lives in the ROCm backend lit suite.

module {
  func.func @target_dtype_defaults() {
    "test.dtype_request"() {numeric_policy = {storage = "int4"}}
        : () -> ()
    return
  }
}

// X86: "test.dtype_request"
// X86-SAME: numeric_policy = {accum = "int32", storage = "int4"}
// X86-NOT: tessera.storage_pack

// NVIDIA: "test.dtype_request"
// NVIDIA-SAME: numeric_policy = {accum = "int32", storage = "int4"}
// NVIDIA-NOT: tessera.storage_pack
