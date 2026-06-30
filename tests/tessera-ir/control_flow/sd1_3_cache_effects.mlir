// SD1-3 — tessera.cache.commit / tessera.cache.rollback are the typed-effect
// (MemoryEffects<[MemWrite]>) speculative-decode cache cursor ops: commit advances
// the KV/SSM cursor to keep the accepted prefix, rollback rewinds rejected drafts.
// The cache handle is threaded value-to-value (!tessera.kv_cache → !tessera.kv_cache).
// They carry a write effect (not [Pure]) so the EffectLattice classifies a region
// using them as `state`, not pure — proven by tests/unit/test_cache_effect_ops.py.
//
// RUN: tessera-opt %s | FileCheck %s

// CHECK-LABEL: func.func @spec_commit_rollback
// CHECK:       tessera.cache.commit %{{.*}}, %{{.*}} : (!tessera.kv_cache, index) -> !tessera.kv_cache
// CHECK:       tessera.cache.rollback %{{.*}}, %{{.*}} : (!tessera.kv_cache, index) -> !tessera.kv_cache
func.func @spec_commit_rollback(%cache: !tessera.kv_cache, %accepted: index,
    %rejected: index) -> !tessera.kv_cache {
  // commit the accepted prefix (advance cursor), then roll back any rejected tail.
  %committed = "tessera.cache.commit"(%cache, %accepted)
      : (!tessera.kv_cache, index) -> !tessera.kv_cache
  %rolled = "tessera.cache.rollback"(%committed, %rejected)
      : (!tessera.kv_cache, index) -> !tessera.kv_cache
  return %rolled : !tessera.kv_cache
}
