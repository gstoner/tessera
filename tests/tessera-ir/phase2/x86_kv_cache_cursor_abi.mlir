// REQUIRES: tessera-x86-target-ir
// RUN: tessera-opt %s --split-input-file --tessera-tile-to-x86 | FileCheck %s

// ODS-WIRE-3: `tessera.cache.commit` / `tessera.cache.rollback` (SD1-3) lower
// through the x86 KV-cache HANDLE ABI (kv_cache_f32.cpp): handle + count in,
// updated handle out. Unlike the artifact-only `tessera_x86_kv_cache_op(kind)`
// bridge, the lowering threads the result -- the normal case, since the ops
// always feed their updated handle onward. ABI-only proof: the numeric check
// runs through the `x86_kv_cache_compiled` lane on a Zen 5 host.

// CHECK-DAG: func.func private @tessera_x86_kv_cache_commit_f32(!llvm.ptr, i64) -> !llvm.ptr
// CHECK-DAG: func.func private @tessera_x86_kv_cache_rollback_f32(!llvm.ptr, i64) -> !llvm.ptr
// A rejected count (dynamic accepted > current_seq, dynamic negative count,
// bad handle) makes the ABI return NULL; every call is followed by a NULL
// check that traps instead of threading NULL into later cache ops (#21).
// CHECK-LABEL: func.func @spec_commit_rollback(
// CHECK-SAME:    %[[CACHE:[^:]+]]: !tessera.kv_cache, %[[ACC:[^:]+]]: index, %[[REJ:[^:]+]]: index)
// CHECK:       %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:       %[[H0:.*]] = builtin.unrealized_conversion_cast %[[CACHE]] : !tessera.kv_cache to !llvm.ptr
// CHECK:       %[[N0:.*]] = arith.index_cast %[[ACC]] : index to i64
// CHECK:       tessera_x86.abi_call {symbol = "tessera_x86_kv_cache_commit_f32"}
// CHECK:       %[[H1:.*]] = call @tessera_x86_kv_cache_commit_f32(%[[H0]], %[[N0]])
// CHECK-SAME:    tessera.kv_cache.abi = "tessera_x86_kv_cache_f32_handle.v1"
// CHECK:       %[[OK1:.*]] = llvm.icmp "ne" %[[H1]], %[[NULL]] : !llvm.ptr
// CHECK:       cf.assert %[[OK1]], "X86_KV_CACHE_CURSOR_{{REFUSED}}: tessera.cache.commit was rejected by the x86 KV-cache handle ABI (tessera_x86_kv_cache_commit_f32 returned NULL
// CHECK:       %[[N1:.*]] = arith.index_cast %[[REJ]] : index to i64
// The committed handle feeds the rollback directly: the result is threaded.
// CHECK:       %[[H2:.*]] = call @tessera_x86_kv_cache_rollback_f32(%[[H1]], %[[N1]])
// CHECK:       %[[OK2:.*]] = llvm.icmp "ne" %[[H2]], %[[NULL]] : !llvm.ptr
// CHECK:       cf.assert %[[OK2]], "X86_KV_CACHE_CURSOR_{{REFUSED}}: tessera.cache.rollback was rejected by the x86 KV-cache handle ABI (tessera_x86_kv_cache_rollback_f32 returned NULL
// CHECK:       %[[OUT:.*]] = builtin.unrealized_conversion_cast %[[H2]] : !llvm.ptr to !tessera.kv_cache
// CHECK:       return %[[OUT]] : !tessera.kv_cache
// CHECK-NOT:   tessera.cache.commit
// CHECK-NOT:   tessera.cache.rollback
func.func @spec_commit_rollback(%cache: !tessera.kv_cache, %accepted: index,
                                %rejected: index) -> !tessera.kv_cache {
  %committed = "tessera.cache.commit"(%cache, %accepted)
      : (!tessera.kv_cache, index) -> !tessera.kv_cache
  %rolled = "tessera.cache.rollback"(%committed, %rejected)
      : (!tessera.kv_cache, index) -> !tessera.kv_cache
  return %rolled : !tessera.kv_cache
}

// -----

// The handle ABI never goes through the kind-only artifact bridge.
// CHECK-LABEL: func.func @commit_only(
// CHECK-NOT:   tessera_x86_kv_cache_op
// CHECK:       %[[H:.*]] = call @tessera_x86_kv_cache_commit_f32
// CHECK:       %[[OK:.*]] = llvm.icmp "ne" %[[H]]
// CHECK:       cf.assert %[[OK]], "X86_KV_CACHE_CURSOR_{{REFUSED}}: tessera.cache.commit
// CHECK-NOT:   cf.assert
// CHECK-NOT:   tessera_x86_kv_cache_op
func.func @commit_only(%cache: !tessera.kv_cache) -> !tessera.kv_cache {
  %n = arith.constant 3 : index
  %c = "tessera.cache.commit"(%cache, %n)
      : (!tessera.kv_cache, index) -> !tessera.kv_cache
  return %c : !tessera.kv_cache
}
