# Tessera Programming Guide
## Chapter 5: Kernel Programming

Kernel programming is where Tessera meets hardware. A **kernel** in Tessera is a function decorated with `@tessera.kernel` that operates on per-rank tensor shards and executes one tile operation per launch invocation. This chapter covers how to write kernels, dispatch them across distributed arrays, and understand the Tile IR ops the compiler generates.

---

### 5.1 The `@tessera.kernel` Decorator

`@tessera.kernel` marks a function as a tile-level kernel dispatched by `tessera.index_launch`. Kernels receive per-rank tensor shards and are never called directly — they are always invoked through `index_launch`.

```python
import tessera

@tessera.kernel
def tp_gemm(
    A: tessera.f16[..., ...],
    B: tessera.f16[..., ...],
    C: tessera.mut_f32[..., ...],
):
    C[:] = tessera.ops.gemm(A, B)
```

Key properties of kernel functions:
- Parameters use **dtype annotations** (`f16`, `bf16`, `f32`, `mut_f32`) or `Region[...]` — not plain Python types.
- The body uses `tessera.ops.*` — the same ops as `@tessera.jit` functions.
- Kernels do not call `tessera.require(...)` — constraints belong in `@tessera.jit` functions.
- The decorator returns a `KernelFn` wrapper. Calling it directly raises a `TypeError`.

---

### 5.2 Dtype Annotations

Dtype annotations express both the **element type** and the **read/write privilege** of a kernel parameter. They use Python's `__class_getitem__` protocol — the `[..., ...]` subscript specifies dimensionality via `Ellipsis` (arbitrary shape).

| Annotation | Element type | Privilege |
|------------|-------------|-----------|
| `tessera.f16[..., ...]` | FP16 | Read-only |
| `tessera.bf16[..., ...]` | BF16 | Read-only |
| `tessera.f32[..., ...]` | FP32 | Read-only |
| `tessera.mut_f32[..., ...]` | FP32 | Write-privileged (mutable) |

The `mut_` prefix signals that the kernel will write to the parameter. This maps directly to the `tessera.effect = "write"` attribute on the corresponding Graph IR function argument.

```python
@tessera.kernel
def scale_add(
    X:   tessera.bf16[..., ...],    # read input
    W:   tessera.bf16[..., ...],    # read weights
    out: tessera.mut_f32[..., ...], # write output
):
    out[:] = tessera.ops.gemm(X, W)
```

---

### 5.3 `index_launch` Dispatch Pattern

`tessera.index_launch` is the primary mechanism for dispatching a kernel across the shards of a distributed array. It takes a mesh axis name, binds it to a kernel, and executes the kernel once per rank along that axis.

**Three-step call pattern:**

```python
tessera.index_launch(axis="tp")    # Step 1: create IndexLauncher for "tp" axis
    (tp_gemm)                      # Step 2: bind kernel → _ShardDispatcher
    (A.parts("tp"),                # Step 3: execute — one call per rank along "tp"
     B.parts("tp"),
     C.parts("tp"))
```

Each step is a separate callable:

```python
launcher   = tessera.index_launch(axis="tp")
dispatcher = launcher(tp_gemm)
dispatcher(A.parts("tp"), B.parts("tp"), C.parts("tp"))
```

`DistributedArray.parts(axis)` returns a list of per-rank slices. The dispatcher calls `tp_gemm` once per element in the list, passing the rank-`i` shard to each kernel invocation.

**Phase 1 behaviour:** Ranks execute sequentially (one Python thread per invocation). In Phase 3+, parallel GPU stream dispatch is used.

---

### 5.4 A Complete Kernel Example

```python
import tessera

# 1. Build distributed arrays
D    = tessera.domain.Rect((4, 128, 256))
dist = tessera.dist.Block(mesh_axes=("dp", "tp"))
X    = tessera.array.from_domain(D, dtype="bf16", distribution=dist)
W    = tessera.array.from_domain(
           tessera.domain.Rect((256, 256)), dtype="bf16",
           distribution=tessera.dist.Replicated()
       )
C    = tessera.array.from_domain(
           tessera.domain.Rect((4, 128, 256)), dtype="f32",
           distribution=dist
       )

# 2. Define the kernel
@tessera.kernel
def shard_gemm(
    x_shard: tessera.bf16[..., ...],
    w:       tessera.bf16[..., ...],
    c_shard: tessera.mut_f32[..., ...],
):
    c_shard[:] = tessera.ops.gemm(x_shard, w)

# 3. Dispatch across tp axis
tessera.index_launch(axis="tp")(shard_gemm)(
    X.parts("tp"),
    [W] * len(X.parts("tp")),  # replicated — same W for all ranks
    C.parts("tp"),
)
```

---

### 5.5 Combining `@tessera.jit` and `@tessera.kernel`

`@tessera.jit` and `@tessera.kernel` serve different levels:

| Decorator | Level | When to use |
|-----------|-------|-------------|
| `@tessera.jit` | Function / graph | Algorithmic logic, constraint checking, effect inference, Graph IR emission |
| `@tessera.kernel` | Tile / shard | Per-rank tile execution dispatched by `index_launch` |

A common pattern is to wrap a kernel dispatch inside a `@tessera.jit` function that handles the domain, distribution, and constraints:

```python
@tessera.jit(bindings={"K": 256})
def distributed_step(
    W: tessera.Region["read"],
    X: tessera.Region["read"],
    Y: tessera.Region["write"],
):
    tessera.require(tessera.constraint.Divisible("K", 64))
    # Fallback: ops.gemm works for non-distributed case
    Y[:] = tessera.ops.gemm(X, W)
```

For explicitly distributed execution with kernel-level control, use `@tessera.kernel` + `index_launch` directly.

---

### 5.6 Tile IR: What the Compiler Generates

When `@tessera.jit(target=GPUTargetProfile(isa=ISA.SM_90))` is set, the compiler lowers kernel bodies through the full Tile IR stack. Understanding Tile IR helps when debugging performance or reading compiler output.

#### Tile async copy

Async tile copies stage data from global memory into shared/L1 memory without blocking compute:

```mlir
%q_tile = tile.async_copy %Q {tile_rows = 64 : i64, tile_cols = 64 : i64}
tile.wait_async
```

In Python, `tessera.ops.flash_attn(Q, K, V)` automatically generates these from `TileIRLoweringPass`. You do not write `tile.async_copy` directly.

#### Tile MMA

The `tile.mma` op represents a tile-granularity matrix multiply accumulate, lowered to `wgmma.mma_async` PTX on SM_90+:

```mlir
%c_tile = tile.mma %a_tile, %b_tile : tensor<64x64xbf16>, tensor<64x64xbf16> -> tensor<64x64xf32>
```

On SM_80/86/89, `tile.mma` falls back to legacy WMMA automatically — no code change needed.

#### FA-4 FlashAttention ops

For `tessera.ops.flash_attn`, `TileIRLoweringPass` expands the single op into the full FA-2 online softmax sequence:

```mlir
%scores  = tessera.attn.scaled_dot_product %q_tile, %k_tile scale = 0.125 : f32
%masked  = tessera.attn.causal_mask %scores q_off = 0 kv_off = 0    // if causal=True
%new_acc, %new_m, %new_l = tessera.attn.online_softmax %masked, %m, %l, %acc
%output, %lse = tessera.attn.lse_accumulate %new_acc, %final_m, %final_l
```

This implements the FA-2 online softmax algorithm: running max + running sum with a per-tile correction factor. A plain batch softmax would OOM on long sequences — the online algorithm processes one KV tile at a time.

---

### 5.7 Warp Specialization (SM_90+)

On SM_90 (Hopper), `WarpSpecializationPass` splits the kernel into **producer** and **consumer** warp roles:

- **Producer warps** run `tile.async_copy` (TMA prefetch into shared memory).
- **Consumer warps** run `tessera.attn.*` compute and `tile.mma`.
- `tessera.queue.push/pop` at the boundary express the handoff ordering.

This is structural — the backend allocates separate register files and mbarrier slots per role. You do not express warp roles in Python; the pass infers them from the Tile IR structure automatically.

You can inspect the generated MLIR to see warp specialization:

```python
from tessera.compiler.gpu_target import GPUTargetProfile, ISA

@tessera.jit(target=GPUTargetProfile(isa=ISA.SM_90, warps_per_cta=4))
def flash_fwd(Q, K, V):
    tessera.require(tessera.constraint.Divisible("D", 64))
    return tessera.ops.flash_attn(Q, K, V, causal=True)

print(flash_fwd.graph_ir.to_mlir())  # shows tessera.flash_attn in Graph IR
```

---

### 5.8 FlashAttnLoweringConfig

When writing a kernel that calls `tessera.ops.flash_attn`, tile sizes and pipeline decisions can be controlled via `FlashAttnLoweringConfig`:

```python
from tessera.compiler.gpu_target import GPUTargetProfile, ISA
from tessera.compiler.attn_lower import FlashAttnLoweringConfig

@tessera.jit(
    target=GPUTargetProfile(isa=ISA.SM_90, warps_per_cta=4),
    attn_config=FlashAttnLoweringConfig(
        tile_q=64,
        tile_kv=64,
        pipeline_stages=2,
        causal=True,
        dropout_p=0.0,
    )
)
def causal_attn(
    Q: tessera.Tensor["B", "H", "S", "D"],
    K: tessera.Tensor["B", "H", "S", "D"],
    V: tessera.Tensor["B", "H", "S", "D"],
) -> tessera.Tensor["B", "H", "S", "D"]:
    tessera.require(tessera.constraint.Divisible("D", 64))
    return tessera.ops.flash_attn(Q, K, V, causal=True)
```

Default for SM_90: `tile_q=64, tile_kv=64, pipeline_stages=2`. The Phase 5 autotuner sweeps `tile_q` and `tile_kv` — they are stored as `tessera.tile_q` / `tessera.tile_kv` attributes on the `tessera.flash_attn` op in the Graph IR, so the autotuner can retile without re-emitting Python.

---

### 5.9 Testing Kernels with MockRankGroup

For multi-rank kernel testing without NCCL or GPU hardware, use `MockRankGroup`:

```python
from tessera.testing import MockRankGroup
import numpy as np

group = MockRankGroup(n=4, mesh_axes={"tp": 4})

def worker(rank):
    # Each rank gets its own shard
    local_A = np.ones((64, 64), dtype=np.float16) * rank.rank
    local_B = np.ones((64, 64), dtype=np.float16)
    result  = np.zeros((64, 64), dtype=np.float32)

    # Simulate kernel execution (Phase 1: numpy-backed ops.gemm)
    result[:] = tessera.ops.gemm(local_A, local_B)

    # Collective across ranks
    total = rank.all_reduce(result, op="sum")
    return total

results = group.run(worker)
# results[i] = per-rank result of the all_reduce
```

`MockRankGroup(n=4, mesh_axes={"tp": 4})` creates 4 fake ranks backed by Python threads sharing in-process numpy buffers. Supported collectives: `all_reduce`, `reduce_scatter`, `all_gather`, `barrier`.

---

### 5.10 Error Handling in Kernels

Errors in kernel programming fall into two categories:

**At decoration time** (caught before any execution):
- `TesseraPrivilegeError` — two parameters both have `mut_f32` dtype on the same output buffer.
- `TesseraAttnConfigError` — invalid `FlashAttnLoweringConfig` (e.g. `tile_q` not a power of 2).
- `TesseraTargetError` — invalid `GPUTargetProfile` (e.g. `warps_per_cta` not power of 2).

**At dispatch time** (caught when `index_launch` is called):
- `ValueError` — number of shard lists doesn't match kernel parameter count.
- `ValueError` — shard list length doesn't match rank count for the axis.
- `MockCollectiveError` — a mock rank raised an exception or timed out (testing only).

```python
# Caught at decoration time
@tessera.jit(
    attn_config=FlashAttnLoweringConfig(tile_q=48)  # TesseraAttnConfigError: 48 not power of 2
)
def bad_attn(Q, K, V): ...

# Caught at dispatch time
tessera.index_launch(axis="tp")(my_kernel)(
    X.parts("tp"),   # 4 shards
    W.parts("tp"),   # 4 shards
    # Missing C.parts("tp") — ValueError: 2 shard lists, kernel expects 3
)
```

---

### 5.11 Summary

- `@tessera.kernel` marks tile-level kernels dispatched by `index_launch`.
- Dtype annotations (`f16`, `bf16`, `f32`, `mut_f32`) express element type and read/write privilege.
- `index_launch(axis)(kernel)(shards...)` is a three-step pattern that fans the kernel across per-rank shards.
- On SM_90+ GPU targets, the compiler generates FA-4 Tile IR (online softmax, async copy, warp specialization) automatically from `tessera.ops.flash_attn`.
- `FlashAttnLoweringConfig` controls tile sizes for the GPU FA-4 lowering path.
- `MockRankGroup` enables multi-rank kernel testing without GPU hardware.
- Use `fn.graph_ir.to_mlir()` to inspect the Graph IR emitted for any `@tessera.jit` function.

---

### Chapter Navigation

- **Previous:** [Chapter 4: Execution Model](Tessera_Programming_Guide_Chapter4_Execution_Model.md)
- **Next:** [Chapter 6: Numerics Model](Tessera_Programming_Guide_Chapter6_Numerics_Model.md)
