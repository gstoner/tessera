// REQUIRES: tessera-metalium-backend
// RUN: tessera-opt %s --pass-pipeline='builtin.module(tessera-lower-to-metalium)' --allow-unregistered-dialect | FileCheck %s

// kv_cache_coverage_matrix.md (2026-05-10) — Metalium KV-cache lowering.
//
// Previously: 🟡 ("scaffolded but not exercised through Tile IR").
// Now: ✅ — emits a real `tessera_metalium.kv_cache_op` artifact carrying
// the DRAM/SRAM staging plan computed by `MetaliumBufferPlanner::planKVCache`.
//
// The lowering is artifact-only: ops with consumed results are left
// alive so the Python `KVCacheHandle` reference path can drive
// execution. This matches the contract used by the Apple GPU/CPU and
// x86 lowerings.

module {
  // CHECK-LABEL: module
  // CHECK:       "tessera_metalium.kv_cache_op"()
  // CHECK-SAME:    abi = "kv_cache_handle"
  // CHECK-SAME:    kind = "tessera.kv_cache.append"
  // CHECK-SAME:    plan = {
  // CHECK-NOT:   tessera.kv_cache.append

  func.func @kv_pipeline(%c: !tessera.kv_cache,
                          %k: tensor<8x4x16xf32>,
                          %v: tensor<8x4x16xf32>) {
    // Artifact-only contract — TileToMetalium erases the source op
    // after emitting the artifact, so we don't keep the result alive.
    "tessera.kv_cache.append"(%c, %k, %v)
        : (!tessera.kv_cache, tensor<8x4x16xf32>, tensor<8x4x16xf32>) -> !tessera.kv_cache
    return
  }

  // CHECK-LABEL: func.func @kv_prune_only
  // CHECK:       "tessera_metalium.kv_cache_op"()
  // CHECK-SAME:    kind = "tessera.kv_cache.prune"
  // CHECK-NOT:   tessera.kv_cache.prune
  func.func @kv_prune_only(%c: !tessera.kv_cache) {
    "tessera.kv_cache.prune"(%c) {window = 64 : i64}
        : (!tessera.kv_cache) -> !tessera.kv_cache
    return
  }
}
