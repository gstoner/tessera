// RUN: tessera-opt %s -tessera-lower-to-apple_gpu --allow-unregistered-dialect | FileCheck %s

// Exercises the tessera-lower-to-apple_gpu pipeline. Verifies that:
//   - matmul / softmax / rope -> Metal/MPS-shaped kernel artifact + dispatch
//   - flash_attn       -> tessera_apple.gpu.metal_kernel{kernel="flash_attn_contract"}
//                         + dispatch
//   - kv_cache.*       -> tessera_apple.gpu.kv_cache_gpu (real artifact;
//                         previously a "unsupported" diagnostic — see
//                         docs/audit/kv_cache_coverage_matrix.md, 2026-05-10)
//   - every kernel is paired with a dispatch (queue + metallib artifact)

module {
  // CHECK-LABEL: module
  // CHECK-NOT: "tessera.matmul"

  "tile.mock"() {source = "tessera.matmul", result = "C", ordinal = 0 : i64} : () -> ()
  "tile.mock"() {source = "tessera.softmax", result = "P", ordinal = 1 : i64} : () -> ()
  "tile.mock"() {source = "tessera.rope", result = "Qrot", ordinal = 2 : i64} : () -> ()
  "tile.mock"() {source = "tessera.flash_attn", result = "O", ordinal = 3 : i64} : () -> ()
  "tile.mock"() {source = "tessera.kv_cache.append", result = "Cache", ordinal = 4 : i64} : () -> ()
}

// CHECK:      tessera_apple.gpu.metal_kernel
// CHECK-SAME: framework = "MPSGraph"
// CHECK-SAME: kernel = "matmul_contract"
// CHECK-SAME: source = "tessera.matmul"
// CHECK-NEXT: tessera_apple.gpu.dispatch
// CHECK-SAME: artifact = "metallib"
// CHECK-SAME: execution_mode = "metal_artifact"
// CHECK-SAME: queue = "MTLCommandQueue"

// CHECK:      tessera_apple.gpu.metal_kernel
// CHECK-SAME: kernel = "softmax_contract"
// CHECK-SAME: source = "tessera.softmax"
// CHECK-SAME: temporary_memory = "row_max_sum"
// CHECK-NEXT: tessera_apple.gpu.dispatch

// CHECK:      tessera_apple.gpu.metal_kernel
// CHECK-SAME: kernel = "rope_contract"
// CHECK-SAME: source = "tessera.rope"
// CHECK-SAME: threadgroup = "128x1x1"
// CHECK-NEXT: tessera_apple.gpu.dispatch

// CHECK:      tessera_apple.gpu.metal_kernel
// CHECK-SAME: kernel = "flash_attn_contract"
// CHECK-SAME: source = "tessera.flash_attn"
// CHECK-SAME: status = "artifact_only"
// CHECK-NEXT: tessera_apple.gpu.dispatch

// CHECK:      tessera_apple.gpu.kv_cache_gpu
// CHECK-SAME: abi = "kv_cache_handle"
// CHECK-SAME: framework = "Metal"
// CHECK-SAME: kind = "tessera.kv_cache.append"
// CHECK-SAME: source = "tessera.kv_cache.append"
