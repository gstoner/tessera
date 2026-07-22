// RUN: tessera-opt --allow-unregistered-dialect --pass-pipeline='builtin.module(tessera-lower-to-x86)' %s | FileCheck %s --check-prefix=X86
// RUN: tessera-opt --allow-unregistered-dialect --pass-pipeline='builtin.module(tessera-lower-to-gpu)' %s | FileCheck %s --check-prefix=NVIDIA
// RUN: tessera-opt --allow-unregistered-dialect --pass-pipeline='builtin.module(tessera-lower-to-rocm)' %s | FileCheck %s --check-prefix=ROCM
//
// CORE-COMPILER-2: target defaults are capability-driven. x86 and NVIDIA run
// compute legalization but leave terminal packing disabled until they own a
// consumer. ROCm runs the full compute -> storage -> descriptor-consume chain.

module {
  func.func @target_dtype_defaults() {
    "test.dtype_request"() {numeric_policy = {storage = "int4"}}
        : () -> ()
    return
  }

  // ROCm's descriptor-aware WMMA generator consumes the default-produced pack.
  "tessera_rocm.wmma_gemm"() {
    name = "default_int4_wmma",
    m = 16 : i64, n = 16 : i64, k = 16 : i64,
    mt = 16 : i64, nt = 16 : i64,
    numeric_policy = {storage = "int4"}
  } : () -> ()
}

// X86: "test.dtype_request"
// X86-SAME: numeric_policy = {accum = "int32", storage = "int4"}
// X86-NOT: tessera.storage_pack

// NVIDIA: "test.dtype_request"
// NVIDIA-SAME: numeric_policy = {accum = "int32", storage = "int4"}
// NVIDIA-NOT: tessera.storage_pack

// ROCM: "test.dtype_request"
// ROCM-SAME: numeric_policy = {accum = "int32", storage = "int4"}
// ROCM-SAME: tessera.storage_container = "int8"
// ROCM-SAME: tessera.storage_pack = {container = "int8", factor = 2 : i64, logical = "int4", signedness = "signed_twos_complement"}
// ROCM-SAME: tessera.storage_packed = true
// ROCM: gpu.module @default_int4_wmma_mod
// ROCM: gpu.func @default_int4_wmma
// ROCM: vector<2xi32>
