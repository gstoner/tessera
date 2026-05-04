// RUN: tessera-opt %s -tessera-lower-to-apple_gpu --allow-unregistered-dialect | FileCheck %s

// Exercises the tessera-lower-to-apple_gpu pipeline. Verifies that:
//   - generic ops      -> tessera_apple.gpu.metal_kernel + dispatch pair
//   - flash_attn       -> tessera_apple.gpu.metal_kernel{kernel="flash_attn_contract"}
//                         + dispatch
//   - kv_cache.*       -> tessera_apple.diagnostic with stable wording
//   - every kernel is paired with a dispatch (queue + metallib artifact)

module {
  // CHECK-LABEL: module
  // CHECK-NOT: "tessera.matmul"

  "tessera.matmul"() {source = "tessera.matmul", result = "C", ordinal = 0 : i64} : () -> ()
  "tessera.flash_attn"() {source = "tessera.flash_attn", result = "O", ordinal = 1 : i64} : () -> ()
  "tessera.kv_cache.append"() {source = "tessera.kv_cache.append", result = "Cache", ordinal = 2 : i64} : () -> ()
}

// CHECK:      tessera_apple.gpu.metal_kernel
// CHECK-SAME: framework = "Metal"
// CHECK-SAME: source = "tessera.matmul"
// CHECK-SAME: threadgroup_memory = "auto"
// CHECK-NEXT: tessera_apple.gpu.dispatch
// CHECK-SAME: artifact = "metallib"
// CHECK-SAME: queue = "MTLCommandQueue"

// CHECK:      tessera_apple.gpu.metal_kernel
// CHECK-SAME: kernel = "flash_attn_contract"
// CHECK-SAME: source = "tessera.flash_attn"
// CHECK-SAME: status = "artifact_only"
// CHECK-NEXT: tessera_apple.gpu.dispatch

// CHECK:      tessera_apple.diagnostic
// CHECK-SAME: KV-cache target lowering is not implemented for Apple GPU
// CHECK-SAME: severity = "unsupported"
