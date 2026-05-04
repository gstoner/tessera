// RUN: tessera-opt %s --allow-unregistered-dialect | FileCheck %s

// Round-trip the Apple Silicon Target IR through tessera-opt to verify the
// dialect is registered and the ops parse + print cleanly. No lowering pass
// runs here — this is a pure dialect-load smoke test for Phase 8 wiring.

module attributes {tessera.ir.level = "target"} {
  // CHECK: tessera_apple.cpu.accelerate_gemm
  // CHECK-SAME: abi = "cblas_sgemm"
  // CHECK-SAME: framework = "Accelerate"
  "tessera_apple.cpu.accelerate_gemm"() {
    source = "tessera.matmul",
    result = "C",
    ordinal = 0 : i64,
    framework = "Accelerate",
    abi = "cblas_sgemm"
  } : () -> ()

  // CHECK: tessera_apple.cpu.vector_reduce
  // CHECK: abi = "vDSP"
  "tessera_apple.cpu.vector_reduce"() {
    source = "tessera.softmax",
    result = "P",
    ordinal = 1 : i64,
    framework = "Accelerate",
    abi = "vDSP"
  } : () -> ()

  // CHECK: tessera_apple.gpu.metal_kernel
  // CHECK: framework = "Metal"
  "tessera_apple.gpu.metal_kernel"() {
    source = "tessera.matmul",
    result = "C",
    ordinal = 0 : i64,
    framework = "Metal",
    threadgroup_memory = "auto"
  } : () -> ()

  // CHECK: tessera_apple.gpu.dispatch
  // CHECK: artifact = "metallib"
  "tessera_apple.gpu.dispatch"() {
    ordinal = 0 : i64,
    queue = "MTLCommandQueue",
    artifact = "metallib"
  } : () -> ()

  // CHECK: tessera_apple.diagnostic
  // CHECK: severity = "unsupported"
  "tessera_apple.diagnostic"() {
    source = "tessera.kv_cache.append",
    result = "Cache",
    ordinal = 0 : i64,
    severity = "unsupported",
    reason = "KV-cache lowering is not implemented for Apple in this phase"
  } : () -> ()
}
