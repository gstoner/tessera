<!-- === MERGE_START: Memory & Execution Model v1.1 (Document 1/2) === -->
# Tessera Memory & Execution Model — v1.1

## Goals
- Make **memory spaces** explicit (register, shared/LDS, global, managed, host, tmem).
- Introduce **cache management** (KV cache, page tables, ring buffers) with declarative policies.
- Enforce **determinism**: ordered reductions and seed-controlled RNG.
- Align with the canonical deep-learning semantic core:
  `tessera.numeric_policy`, Graph IR KV cache objects, Schedule IR movement
  effects, typed collective futures, and durable schedule artifacts.

## 1. Memory Spaces & Lifetimes
### 1.1 Canonical Spaces
- `register`: per-thread registers (fastest, smallest)
- `shared` / `lds`: on-chip block-local memory
- `global`: device DRAM
- `managed`: unified memory (host/device migratable)
- `host`: CPU RAM (pinned or pageable)
- `tmem`: Tensor Memory (SM_100+)

**IR surface (Schedule IR → Tile IR)**:
```mlir
%staged = schedule.prefetch %src {
  into = "shared",
  overlap = "compute",
  stage = 0,
  vector = 16
} : tensor<*xbf16> -> tensor<*xbf16>

// Allocate shared with swizzle and optional padding
%buf = tile.alloc_shared : memref<64x65xbf16, 3> { swizzle = "xor", bank_pad = 1 }

// Async copy with stage index (double buffering)
tile.async_copy %src, %buf { stage = 0, vector = 16 } : (memref<*,1>, memref<*,3>) -> ()
tile.wait_async { stage = 0 } : () -> ()
```

### 1.2 Lifetimes & Staging
- Stages are small integers forming a **DAG**: every `async_copy` must be matched by a `wait_async(stage)` in dominance.
- Verifier checks: unique stage per region; no overlapping invalidations.

## 2. Cache Management Abstractions
### 2.1 KV Cache
- Ring buffer shards with page table indirection.
- Eviction: `lru|fifo|none`, line size attr, segment size.

**IR surface (Graph IR → Schedule IR → Tile IR)**:
```mlir
%kv = tessera.kv_cache.create {
  max_seq = 4096,
  head_dim = 128,
  eviction = "rolling_window",
  page_size = 256,
  numeric_policy = #tessera.numeric_policy<storage = "bf16", accum = "f32", ...>
}
%kv2 = tessera.kv_cache.append %kv, %k, %v
%y = tessera.flash_attn %q, %kv2 { causal = true }

%kv_legacy = cache.kv.create { key_dtype = f16, value_dtype = f16, line=256, evict = "lru" }
%page = cache.page.lookup %kv, %pos : (cache.kv, i32) -> cache.page
cache.page.write %kv, %page, %k, %v
%k2, %v2 = cache.page.read %kv, %page
```

### 2.2 Page Tables & Ring Buffers
```mlir
%pt = cache.pt.create { page_size = 256, pages = 4096 }
%rb = cache.ring.create { capacity = 65536 }
cache.ring.push %rb, %item
%item = cache.ring.pop %rb
```

## 3. Deterministic Execution
- `@deterministic(seed=X)` marks a function/region:
  - Reductions use tree order with stable partitioning.
  - RNG ops derive streams from (func-id, mesh-coords, step, user-seed).

**IR attributes**:
```mlir
func.func @f(...) attributes { tessera.deterministic = { seed = 42 } }
tile.reduce %x { op = "sum", order = "tree" } : (vector<128xf32>) -> f32
rng.uniform %shape { stream = "default" } : tensor<*xf32>
```

**Verifier checks**:
- No unordered reductions in deterministic regions.
- RNG requires a bound stream; streams compose via hash(key tuple).

<!-- === MERGE_END: Memory & Execution Model v1.1 (Document 1/2) === -->
