// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(tessera-lower-to-rocm)' %s | FileCheck %s

// ROCm's descriptor-aware WMMA generator consumes the default-produced INT4
// storage pack. This fixture is backend-owned because the pipeline is absent
// from the core LLVM/Apple build by design.

module {
  func.func @target_dtype_defaults() {
    "test.dtype_request"() {numeric_policy = {storage = "int4"}}
        : () -> ()
    return
  }

  "tessera_rocm.wmma_gemm"() {
    name = "default_int4_wmma",
    m = 16 : i64, n = 16 : i64, k = 16 : i64,
    mt = 16 : i64, nt = 16 : i64,
    numeric_policy = {storage = "int4"}
  } : () -> ()
}

// CHECK: "test.dtype_request"
// CHECK-SAME: numeric_policy = {accum = "int32", storage = "int4"}
// CHECK-SAME: tessera.storage_container = "int8"
// CHECK-SAME: tessera.storage_pack = {container = "int8", factor = 2 : i64, logical = "int4", signedness = "signed_twos_complement"}
// CHECK-SAME: tessera.storage_packed = true
// CHECK: gpu.module @default_int4_wmma_mod
// CHECK: gpu.func @default_int4_wmma
// CHECK: vector<2xi32>
