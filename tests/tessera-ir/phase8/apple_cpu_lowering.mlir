// RUN: tessera-opt %s --pass-pipeline='builtin.module(tessera-lower-to-apple_cpu)' --allow-unregistered-dialect | FileCheck %s

// Exercises the tessera-lower-to-apple_cpu pipeline against synthetic Tile IR
// inputs (the same op-name spellings the Python pipeline produces in
// matmul_pipeline.py). Verifies that:
//   - matmul / gemm     -> tessera_apple.cpu.accelerate_gemm
//   - softmax           -> tessera_apple.cpu.vector_reduce
//   - flash_attn        -> tessera_apple.cpu.vector_op (CPU has no Metal kernel)
//   - kv_cache.*        -> tessera_apple.cpu.kv_cache_cpu (real artifact;
//                          previously a "unsupported" diagnostic — see
//                          docs/audit/kv_cache_coverage_matrix.md, 2026-05-10)

module {
  // CHECK-LABEL: module
  // CHECK-NOT: "tessera.matmul"

  "tile.mock"() {source = "tessera.matmul",  result = "C",     ordinal = 0 : i64} : () -> ()
  "tile.mock"() {source = "tessera.gemm",    result = "C2",    ordinal = 1 : i64} : () -> ()
  "tile.mock"() {source = "tessera.softmax", result = "P",     ordinal = 2 : i64} : () -> ()
  "tile.mock"() {source = "tessera.flash_attn", result = "O",  ordinal = 3 : i64} : () -> ()
  "tile.mock"() {source = "tessera.kv_cache.append", result = "Cache", ordinal = 4 : i64} : () -> ()
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

// CHECK:      tessera_apple.cpu.kv_cache_cpu
// CHECK-SAME: abi = "kv_cache_handle"
// CHECK-SAME: kind = "tessera.kv_cache.append"
// CHECK-SAME: source = "tessera.kv_cache.append"
