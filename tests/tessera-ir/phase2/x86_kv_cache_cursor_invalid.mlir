// REQUIRES: tessera-x86-target-ir
// RUN: tessera-opt %s --split-input-file --tessera-tile-to-x86 -verify-diagnostics

// ODS-WIRE-3 negative cases: a constant count the x86 KV-cache handle ABI
// would reject at run time is refused at compile time with a registered
// diagnostic (Decision #21), never lowered to a call that returns NULL.

func.func @negative_commit(%cache: !tessera.kv_cache) -> !tessera.kv_cache {
  %n = arith.constant -1 : index
  // expected-error @+1 {{X86_KV_CACHE_CURSOR_REFUSED}}
  %c = "tessera.cache.commit"(%cache, %n)
      : (!tessera.kv_cache, index) -> !tessera.kv_cache
  return %c : !tessera.kv_cache
}

// -----

func.func @negative_rollback(%cache: !tessera.kv_cache) -> !tessera.kv_cache {
  %n = arith.constant -4 : index
  // expected-error @+1 {{count -4 is negative}}
  %c = "tessera.cache.rollback"(%cache, %n)
      : (!tessera.kv_cache, index) -> !tessera.kv_cache
  return %c : !tessera.kv_cache
}
