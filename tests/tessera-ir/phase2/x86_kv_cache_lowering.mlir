// RUN: tessera-opt %s --tessera-tile-to-x86 --allow-unregistered-dialect | FileCheck %s

// kv_cache_coverage_matrix.md (2026-05-10) — x86 KV-cache lowering.
//
// Previously: ⛔ "not encountered" (no implementation; ops would
// pass through silently if a test path had pushed them).
// Now: 🔲 → ✅ artifact-only — emits real `func.call
// @tessera_x86_kv_cache_op(kind: i32)` artifacts that the Python
// runtime path consumes via the `KVCacheHandle` reference impl.

// CHECK: func.func private @tessera_x86_kv_cache_op(i32)

func.func @kv_pipeline(%c: !tessera.kv_cache,
                        %k: tensor<8x4x16xf32>,
                        %v: tensor<8x4x16xf32>) {
  // CHECK-LABEL: func.func @kv_pipeline
  // CHECK:       %[[K1:.*]] = arith.constant 1 : i32
  // CHECK:       call @tessera_x86_kv_cache_op(%[[K1]])
  // CHECK-NOT:   tessera.kv_cache.append
  "tessera.kv_cache.append"(%c, %k, %v)
      : (!tessera.kv_cache, tensor<8x4x16xf32>, tensor<8x4x16xf32>) -> !tessera.kv_cache
  return
}

func.func @kv_prune_only(%c: !tessera.kv_cache) {
  // CHECK-LABEL: func.func @kv_prune_only
  // CHECK:       %[[K2:.*]] = arith.constant 2 : i32
  // CHECK:       call @tessera_x86_kv_cache_op(%[[K2]])
  // CHECK-NOT:   tessera.kv_cache.prune
  "tessera.kv_cache.prune"(%c) {window = 64 : i64}
      : (!tessera.kv_cache) -> !tessera.kv_cache
  return
}
