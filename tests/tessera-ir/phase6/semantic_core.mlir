// RUN: FileCheck %s < %s
//
// Contract fixture for the deep-learning semantic core. This intentionally
// checks the canonical IR spelling rather than a specific lowering.

module attributes {tessera.ir.version = "1.1"} {
  // CHECK: #tessera.numeric_policy
  // CHECK: tessera.kv_cache.create
  // CHECK: schedule.prefetch
  // CHECK: tessera.flash_attn
  // CHECK: tessera.collective.reduce_scatter
  // CHECK: tessera.collective.await
  // CHECK: schedule.artifact

  func.func @decode(%q: tensor<1x128xbf16>, %k: tensor<1x128xbf16>,
                    %v: tensor<1x128xbf16>, %grad: memref<128xbf16>)
      -> tensor<1x128xbf16> {
    %cache = "tessera.kv_cache.create"() {
      max_seq = 4096 : i64,
      head_dim = 128 : i64,
      eviction = "rolling_window",
      page_size = 256 : i64,
      numeric_policy = #tessera.numeric_policy<
        storage = "bf16", accum = "f32", rounding = "stochastic",
        scale = 1.0, quant_axis = "none", deterministic = true>
    } : () -> !tessera.kv_cache
    %cache2 = "tessera.kv_cache.append"(%cache, %k, %v)
      : (!tessera.kv_cache, tensor<1x128xbf16>, tensor<1x128xbf16>) -> !tessera.kv_cache
    %staged = "schedule.prefetch"(%cache2) {
      into = "shared",
      overlap = "compute",
      stage = 0 : i32,
      vector = 16 : i32
    } : (!tessera.kv_cache) -> !tessera.kv_cache
    %out = "tessera.flash_attn"(%q, %staged) {
      causal = true,
      head_dim = 128 : i64,
      numeric_policy = #tessera.numeric_policy<
        storage = "bf16", accum = "f32", rounding = "nearest_even",
        scale = 1.0, quant_axis = "none", deterministic = true>
    } : (tensor<1x128xbf16>, !tessera.kv_cache) -> tensor<1x128xbf16>
    %future = "tessera.collective.reduce_scatter"(%grad) {
      reduce_op = "sum",
      mesh_axis = "dp",
      scatter_dim = 0 : i64,
      tessera.future_payload = memref<128xbf16>
    } : (memref<128xbf16>) -> !tessera.collective.future<memref<128xbf16>>
    %reduced = "tessera.collective.await"(%future)
      : (!tessera.collective.future<memref<128xbf16>>) -> memref<128xbf16>
    "schedule.artifact"() {
      hash = "0123456789abcdef",
      arch = "sm90",
      shape_key = "B=1;S=1;D=128;dtype=bf16",
      tile = {tile_m = 64 : i64, tile_n = 64 : i64},
      movement = {prefetch = "auto", overlap = "compute"},
      numeric_policy = "bf16@accum(f32)"
    } : () -> ()
    return %out : tensor<1x128xbf16>
  }
}
