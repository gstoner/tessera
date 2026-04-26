<!-- === MERGE_START: Parallelism Constructs v1.1 (Document 2/2) === -->
# Tessera Parallelism Constructs — v1.1

## Goals
- Integrate **mesh axes** into IR semantics.
- Expose **pipeline schedules** (1F1B, interleaved) at Schedule IR.
- Extend **MoE** with dynamic load-balancing hooks (token limiter, A2A planner).

## 1. Mesh Semantics
Introduce `schedule.mesh.region` with explicit `axes` and legality checks for collectives.

```mlir
// Define a 2x4x2 mesh and enter regions per axis
schedule.mesh.define @M dims = [2,4,2] axis_names = ["data","model","pipe"]

schedule.mesh.region @M { axis = "data" } {
  // Data-parallel work; all-reduce over "data" is legal
  schedule.collective "all_reduce" { axis = "data", op = "mean" }
}
schedule.mesh.region @M { axis = "model" } {
  // Model-parallel sharded matmul legality checked
}
```

## 2. Pipeline Schedules
Attach schedules at region granularity.

```mlir
schedule.pipeline.region { schedule = "1f1b", micro_batches = 8 } {
  // Staged subgraph
  schedule.stage @s0 { devices = ["gpu:0"] } {
    // work...
  }
  schedule.stage @s1 { devices = ["gpu:1"] } { /* ... */ }
}
```

Verifier:
- Valid schedule name (`"1f1b" | "interleaved"`), positive micro-batches.
- No backward-before-forward across stages unless schedule permits.

## 3. Expert Parallel (MoE)
Planner + dynamic token limiter.

```mlir
moe.plan { a2a_bucket = 131072, pack_cast = "bf16->fp8" }
moe.token_limiter.create @lim { max_tokens_in_flight = 4, refill = 1 }
moe.dispatch %x { router = @router, limiter = @lim } : (tensor<*xf16>) -> tensor<*xf16>
```

Runtime hooks:
- Limiter exposes acquire/release tokens for CommQ submission.
- Planner provides ranks/buckets and (optional) compression policy.

<!-- === MERGE_END: Parallelism Constructs v1.1 (Document 2/2) === -->
