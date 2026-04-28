# Tessera Deep Learning Semantic Core

This document defines the canonical compiler architecture for deep-learning
programs. Backend-specific passes may add target details, but they must preserve
these semantic objects through Graph IR, Schedule IR, Tile IR, and Target IR.

## 1. Numerics Policy

Numerics are part of the IR contract, not incidental dtype casts.

Canonical representation:

```mlir
#tessera.numeric_policy<
  storage = "bf16",
  accum = "f32",
  rounding = "stochastic",
  scale = 1.0,
  quant_axis = "none",
  deterministic = true>
```

Matmul, attention, safe normalization, safe softmax/logsumexp, casts, and
quantized weights should carry a `numeric_policy` when precision behavior
matters. Lowering uses this to choose MMA forms, accumulator types, rounding,
dequantization scales, and deterministic reductions.

## 2. State And Cache Objects

KV cache, page tables, and rings are Graph IR state objects before they become
backend buffers.

```mlir
%cache = tessera.kv_cache.create {
  max_seq = 4096,
  head_dim = 128,
  eviction = "rolling_window",
  page_size = 256,
  numeric_policy = #tessera.numeric_policy<...>
}
%cache2 = tessera.kv_cache.append %cache, %k, %v
%y = tessera.flash_attn %q, %cache2 { causal = true }
```

`flash_attn(q, cache)` is the preferred semantic form for inference and decode.
The older `flash_attn(q, k, v)` form remains a legal lowering input for simple
training kernels.

## 3. Movement Effects

Movement is represented in Schedule IR before Tile IR lowering.

```mlir
%staged = schedule.prefetch %cache {
  into = "shared",
  overlap = "compute",
  stage = 0,
  vector = 16
} : !tessera.kv_cache -> !tessera.kv_cache
```

Tile IR lowers schedule movement into `tile.async_copy`, `tile.wait_async`,
TMA/cp.async, barriers, and memory-space-specific layouts. This prevents data
movement from being invented late in codegen.

## 4. Typed Async Collectives

Collectives produce futures and may operate on typed shards.

```mlir
%s = tessera.collective.shard_view %x on "tp" dim 0
%f = tessera.collective.reduce_scatter %x, "sum" on "tp" dim 0
%y = tessera.collective.await %f
```

The transform library may temporarily mark generic ops with
`tessera.future_payload` when it avoids linking generated collective headers,
but dialect-aware legalization must materialize `!tessera.collective.future<T>`.

## 5. Schedule Artifacts

Autotuning output is a durable artifact:

```mlir
schedule.artifact {
  hash = "stable16hex",
  arch = "sm90",
  shape_key = "M=4096;N=4096;K=4096;dtype=bf16",
  tile = {tile_m = 128, tile_n = 128, tile_k = 32},
  movement = {prefetch = "auto", overlap = "compute"},
  numeric_policy = "bf16@accum(f32)"
}
```

Deployment bundles should store graph IR, schedule artifacts, generated
binaries, and the runtime manifest together so compilation is reproducible.

## 6. Determinism Contract

The effect lattice is:

```text
pure < random < movement < state < collective < memory < io
```

`@jit(deterministic=True)` governs:

- RNG streams and dropout masks through seed-derived streams.
- Reduction order through explicit ordered reductions.
- Collective ordering through typed future/await dependencies.
- Movement ordering through schedule movement tokens.
- Schedule choice through artifact hashes.

Host I/O and unknown external calls are not deterministic. Unseeded RNG is not
deterministic. State and collectives are allowed only when represented in IR so
the compiler/runtime can enforce ordering.
