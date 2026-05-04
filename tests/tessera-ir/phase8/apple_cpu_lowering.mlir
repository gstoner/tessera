// RUN: tessera-opt %s -tessera-lower-to-apple_cpu --allow-unregistered-dialect | FileCheck %s

// Exercises the tessera-lower-to-apple_cpu pipeline against synthetic Tile IR
// inputs (the same op-name spellings the Python pipeline produces in
// matmul_pipeline.py). Verifies that:
//   - matmul / gemm     -> tessera_apple.cpu.accelerate_gemm
//   - softmax           -> tessera_apple.cpu.vector_reduce
//   - flash_attn        -> tessera_apple.cpu.vector_op (CPU has no Metal kernel)
//   - kv_cache.*        -> tessera_apple.diagnostic with stable wording

module {
  // CHECK-LABEL: module
  // CHECK-NOT: "tessera.matmul"

  "tessera.matmul"() {source = "tessera.matmul",  result = "C",     ordinal = 0 : i64} : () -> ()
  "tessera.gemm"()   {source = "tessera.gemm",    result = "C2",    ordinal = 1 : i64} : () -> ()
  "tessera.softmax"() {source = "tessera.softmax", result = "P",    ordinal = 2 : i64} : () -> ()
  "tessera.flash_attn"() {source = "tessera.flash_attn", result = "O", ordinal = 3 : i64} : () -> ()
  "tessera.kv_cache.append"() {source = "tessera.kv_cache.append", result = "Cache", ordinal = 4 : i64} : () -> ()
}

// CHECK:      tessera_apple.cpu.accelerate_gemm
// CHECK:      abi = "cblas_sgemm"
// CHECK-SAME: framework = "Accelerate"
// CHECK-SAME: source = "tessera.matmul"

// CHECK:      tessera_apple.cpu.accelerate_gemm
// CHECK-SAME: source = "tessera.gemm"

// CHECK:      tessera_apple.cpu.vector_reduce
// CHECK-SAME: abi = "vDSP"
// CHECK-SAME: source = "tessera.softmax"

// CHECK:      tessera_apple.cpu.vector_op
// CHECK-SAME: abi = "vecLib"
// CHECK-SAME: source = "tessera.flash_attn"

// CHECK:      tessera_apple.diagnostic
// CHECK-SAME: KV-cache target lowering is not implemented for Apple CPU
// CHECK-SAME: severity = "unsupported"
