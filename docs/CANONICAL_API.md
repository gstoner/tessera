---
status: Normative
classification: Normative
last_updated: 2026-05-04
---

# Tessera Canonical API Quick Reference
**Status:** Normative - grounded in `python/tessera/` active implementation  
**Last updated:** May 4, 2026  
**Use this document** to resolve any naming disagreement in other docs. If something here conflicts with another doc, this file wins.

---

## Core Rule: One Name per Concept

Other docs use `@tessera.function`, `@ts.kernel`, `@jit`, and other variants. **None of those are correct.** The table below is authoritative.

---

## Decorators

| Decorator | Import path | Purpose |
|-----------|-------------|---------|
| `@tessera.jit` | `tessera.jit` | Compile a Python function: run ConstraintSolver, infer effects, emit Graph IR |
| `@tessera.kernel` | `tessera.kernel` | Mark a tile-level kernel function dispatched by `index_launch` |

```python
import tessera

@tessera.jit
def step(W: tessera.Region["read"], X: tessera.Region["read"],
         Y: tessera.Region["write"]):
    Y[:] = tessera.ops.gemm(X, W)

@tessera.kernel
def tp_gemm(A: tessera.f16[..., ...], B: tessera.f16[..., ...],
            C: tessera.mut_f32[..., ...]):
    C[:] = tessera.ops.gemm(A, B)
```

### `@tessera.jit` parameters

```python
@tessera.jit(
    deterministic: bool = False,     # if True, forbid unseeded random effects
    seed: int | None = None,         # RNG seed; required when using dropout under deterministic=True
    bindings: dict[str,int] | None = None,  # concrete dim sizes for early constraint checking
    target: GPUTargetProfile | str | None = None, # CPU, GPU, or artifact target
    attn_config: FlashAttnLoweringConfig | None = None,  # FA-4 tile sizes; auto-set on SM_90+
)
```

String target aliases are normalized by `python/tessera/compiler/matmul_pipeline.py`.
Artifact targets emit inspectable Target IR; native execution is only implied for
targets marked hardware-runtime by their backend docs.

---

## Region Privileges

`Region[mode]` is a **type annotation only** — not a runtime wrapper. It lowers to `tessera.effect` attributes on Graph IR function arguments.

| Syntax | Mode | Exclusive? | Reduces? |
|--------|------|-----------|---------|
| `Region["read"]` | read-only | No | No |
| `Region["write"]` | exclusive write | Yes | No |
| `Region["reduce_sum"]` | parallel sum | No | Yes (`op="sum"`) |
| `Region["reduce_max"]` | parallel max | No | Yes (`op="max"`) |
| `Region["reduce_min"]` | parallel min | No | Yes (`op="min"`) |

```python
@tessera.jit
def grad_step(X: tessera.Region["read"], G: tessera.Region["reduce_sum"]):
    G += tessera.ops.gemm(X, X.T)
```

Invalid mode → `ValueError` at annotation time. Conflicting write regions → `TesseraPrivilegeError` at `@jit` decoration time.

---

## Domain & Distribution API

Domains and distributions are **always separate objects** — shape vs. placement. Never merge them.

```python
# 1. Define the logical shape
D = tessera.domain.Rect((4, 128, 256))      # dims: (batch, seq, hidden)

# 2. Define the placement strategy  
dist = tessera.dist.Block(mesh_axes=("dp", "tp"))

# 3. Create the distributed array
X = tessera.array.from_domain(D, dtype="bf16", distribution=dist)

assert X.shape == (4, 128, 256)              # logical (global) shape
assert X.shard_spec.mesh_axes == ("dp", "tp")
assert X.dtype == "bf16"
```

### `tessera.domain`

| Symbol | Type | Description |
|--------|------|-------------|
| `tessera.domain.Rect(dims)` | `Rect` | Dense rectangular domain. `dims` is a tuple of positive ints. |

### `tessera.dist`

| Symbol | Phase | Description |
|--------|-------|-------------|
| `tessera.dist.Block(mesh_axes)` | implemented | Contiguous block partition. `mesh_axes` is a non-empty tuple of strings. |
| `tessera.dist.Cyclic(mesh_axes)` | implemented / scaffolded runtime | Round-robin partition. `make_shard_spec` returns a cyclic `ShardSpec`; runtime behavior is still backend-dependent. |
| `tessera.dist.Replicated()` | implemented | No partition - tensor replicated on all ranks. |

### `tessera.array`

| Symbol | Description |
|--------|-------------|
| `tessera.array.from_domain(domain, dtype, distribution, fill="zeros", mesh=None)` | Create `DistributedArray` |

### `DistributedArray`

| Attribute / Method | Type | Description |
|--------------------|------|-------------|
| `.shape` | `tuple[int,...]` | Logical (global) shape |
| `.dtype` | `str` | Storage dtype string |
| `.shard_spec` | `ShardSpec` | Partition metadata |
| `.ndim` | `int` | Number of dimensions |
| `.numel` | `int` | Total element count |
| `.parts(axis)` | `list[DistributedArray]` | Per-rank slices along mesh axis |
| `.numpy()` | `np.ndarray` | Backing numpy array (Phase 1 CPU only) |

---

## `ShardSpec`

```python
ShardSpec(partition=(0, 1), mesh_axes=("dp", "tp"))
ShardSpec.replicate()   # fully replicated
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `.partition` | `tuple[int,...]` | Logical dimension indices that are partitioned |
| `.mesh_axes` | `tuple[str,...]` | Mesh axis names, one per partitioned dim |
| `.replicated` | `bool` | True if tensor is fully replicated |

---

## Constraint API

Constraints are checked at `@jit` decoration time, not at call time.

```python
@tessera.jit
def aligned_gemm(A: tessera.Tensor["M", "K"], B: tessera.Tensor["K", "N"]):
    tessera.require(tessera.constraint.Divisible("K", 64))
    tessera.require(tessera.constraint.Range("M", 1, 8192))
    tessera.require(tessera.constraint.Equal("D_in", "D_out"))
    return tessera.ops.gemm(A, B)
```

| Symbol | Args | Checks |
|--------|------|--------|
| `tessera.constraint.Divisible(dim, divisor)` | `str, int` | `dim % divisor == 0` |
| `tessera.constraint.Range(dim, lo, hi)` | `str, int, int` | `lo <= dim <= hi` |
| `tessera.constraint.Equal(dim_a, dim_b)` | `str, str` | `dim_a == dim_b` |

Violation → `TesseraConstraintError` at decoration time when concrete `bindings` are provided. Runtime first-call shape binding is planned but not currently implemented.

---

## Effect System

Effects are **inferred, not declared** (except the `deterministic` flag).

| Effect | Value | Meaning | Status |
|--------|-------|---------|--------|
| `Effect.pure` | 0 | No side effects; recompute-safe | Implemented |
| `Effect.random` | 1 | Calls RNG; result varies | Implemented |
| `Effect.movement` | 2 | Explicit data movement or async copy/wait | Implemented |
| `Effect.state` | 3 | Compiler-visible state, such as KV cache or rings | Implemented |
| `Effect.collective` | 4 | Device/rank communication | Implemented |
| `Effect.memory` | 5 | Writes mutable tensors or aliases host-visible memory | Implemented |
| `Effect.io` | 6 | Host I/O or unknown external work | Implemented |
| `Effect.top` | 7 | Unknown / unconstrained | Implemented |

Lattice join: `effect_a.join(effect_b)` → `max(a, b)`.

```python
@tessera.jit(deterministic=True, seed=42)
def stable_fwd(x):
    return tessera.ops.layer_norm(x)   # pure → OK under deterministic

@tessera.jit(deterministic=True)
def bad_fwd(x):
    return tessera.ops.dropout(x, p=0.1)  # random → TesseraEffectError
```

---

## Index Launch

```python
tessera.index_launch(axis="tp")(my_kernel)(
    A.parts("tp"),    # list of per-rank shards
    B.parts("tp"),
    C.parts("tp"),
)
```

`index_launch(axis)` returns an `IndexLauncher`. Calling it with a kernel returns a `_ShardDispatcher`. Calling that with shard lists executes the kernel once per rank (sequentially in Phase 1, parallel in Phase 3+).

---

## Target API

```python
from tessera.compiler.gpu_target import GPUTargetProfile, ISA

@tessera.jit(target=GPUTargetProfile(isa=ISA.SM_90, warps_per_cta=4))
def flash_attn_fwd(Q, K, V):
    tessera.require(tessera.constraint.Divisible("D", 64))
    return tessera.ops.flash_attn(Q, K, V, causal=True)
```

| `ISA` value | Hardware | WGMMA | TMA |
|-------------|----------|-------|-----|
| `ISA.SM_80` | A100 | ❌ | ❌ |
| `ISA.SM_86` | RTX 30xx | ❌ | ❌ |
| `ISA.SM_89` | RTX 40xx | ❌ | ❌ |
| `ISA.SM_90` | H100 / GH200 | ✅ | ✅ |
| `ISA.SM_100` | B100 / GB200 | ✅ | ✅ |
| `ISA.SM_120` | Rubin placeholder | ✅ | ✅ |

`GPUTargetProfile` key parameters: `isa`, `warps_per_cta` (default 4, must be power of 2), `shared_mem_bytes` (None = SM default), `pipeline_stages` (default 2).

Key properties: `.supports_wgmma` / `.supports_tma` / `.supports_mbarrier` → `isa >= SM_90`; `.supports_tcgen05` / `.supports_tmem` / `.supports_cta_pairs` / `.supports_block_scaled_mma` → `isa >= SM_100`; `.runtime_arch` emits CUDA architecture strings such as `sm_90a`, `sm_100a`, and `sm_120`.

### String target aliases

| Input | Normalized target | Notes |
|-------|-------------------|-------|
| `None`, `"cpu"` | `cpu` | CPU/mock-runtime path when supported by the op graph |
| `"cuda"`, `"nvidia"`, `"gpu"`, `"sm90"`, `"sm_90"`, `"sm90a"`, `"sm_90a"`, `"hopper"` | `nvidia_sm90` | NVIDIA artifact target |
| `"rocm"`, `"amd"`, `"hip"` | `rocm` | ROCm Target IR artifact |
| `"metalium"`, `"tt_metalium"`, `"tt"` | `metalium` | Metalium Target IR artifact |
| `"apple_cpu"`, `"macos_cpu"`, `"m_series_cpu"` | `apple_cpu` | Apple CPU Target IR artifact |
| `"apple_gpu"` | `apple_gpu` | Apple GPU Target IR artifact |

---

## Dtype Annotations

| Annotation | Dtype | Notes |
|------------|-------|-------|
| `tessera.f16[..., ...]` | FP16 | Read-only tensor annotation |
| `tessera.bf16[..., ...]` | BF16 | Read-only tensor annotation |
| `tessera.f32[..., ...]` | FP32 | Read-only tensor annotation |
| `tessera.mut_f32[..., ...]` | FP32 | Write-privileged tensor annotation |

---

## `tessera.ops` Namespace

The public namespace is a `SimpleNamespace` backed by a registry. Most
development-time implementations are NumPy reference paths or artifact
lowerings; runtime kernels can be registered separately.

| Op | Signature | Notes |
|----|-----------|-------|
| `tessera.ops.gemm(A, B)` | `(array, array) → array` | Matrix multiply via `np.matmul` |
| `tessera.ops.matmul(A, B)` | alias for `gemm` | |
| `tessera.ops.layer_norm(x, eps=1e-5)` | `(array) → array` | Pure effect |
| `tessera.ops.softmax(x, axis=-1)` | `(array) → array` | Pure effect |
| `tessera.ops.gelu(x)` | `(array) → array` | Pure effect |
| `tessera.ops.relu(x)` | `(array) → array` | Pure effect |
| `tessera.ops.transpose(x, axes=None)` | `(array) → array` | Pure effect |
| `tessera.ops.cast(x, dtype)` | `(array, str) → array` | Pure effect |
| `tessera.ops.dropout(x, p=0.1, training=True)` | `(array) → array` | `random` effect |
| `tessera.ops.conv2d(x, weight, bias=None, stride=1, padding=0)` | `(NHWC,HWIO) → NHWC` | NumPy reference convolution |
| `tessera.ops.flash_attn(Q, K, V, scale=None, causal=False, dropout_p=0.0, seed=None)` | `(array,array,array) → array` | NumPy reference attention; SM90+ lowering artifacts where supported |
| `tessera.ops.all_reduce(x, op="sum")` | `(array) → array` | Single-rank no-op reference path |
| `tessera.ops.reduce_scatter(x, op="sum", axis=0)` | `(array) → array` | Single-rank no-op reference path |
| `tessera.ops.all_gather(x, axis=0)` | `(array) → array` | Single-rank no-op reference path |
| `tessera.ops.fused_epilogue(x, bias=None, activation="linear")` | `(array) → array` | |
| `tessera.ops.fft/ifft/rfft/irfft(...)` | `(array) → array` | NumPy FFT helpers |
| `tessera.ops.dct(x, type=2, axis=-1)` | `(array) → array` | NumPy FFT-derived DCT reference |
| `tessera.ops.spectral_conv(x, w)` | `(array,array) → array` | NumPy FFT convolution reference |
| `tessera.ops.rmsnorm_safe(x, eps=1e-6)` | `(array) → array` | NumPy RMSNorm reference |
| `tessera.ops.kv_cache_append(cache, key, value)` | cache helper | Reference KV-cache helper |
| `tessera.ops.kv_cache_prune(cache, max_entries=None, max_seq=None)` | cache helper | Reference KV-cache helper |

Registry helpers:

| Symbol | Purpose |
|--------|---------|
| `tessera.ops.registry` | List and dispatch registered op entries |
| `tessera.ops.register_reference(name, fn, **metadata)` | Register a reference implementation |
| `tessera.ops.register_lowering(name, fn, **metadata)` | Register an artifact/lowering hook |
| `tessera.ops.register_runtime_kernel(name, fn, **metadata)` | Register a runtime kernel hook |

---

## Runtime API

Top-level runtime helpers are re-exported from `python/tessera/runtime.py`.
When a compiled runtime shared library is not available, the Python wrapper
falls back to a mock CPU runtime for development and tests.

| Symbol | Purpose |
|--------|---------|
| `tessera.RuntimeArtifact` | Serializable Graph/Schedule/Tile/Target IR artifact bundle |
| `tessera.RuntimeProfile` | Runtime/profile timing and memory metadata |
| `tessera.compile_artifact(...)` | Create a runtime artifact from IR text/metadata |
| `tessera.load_artifact(...)` | Load a serialized runtime artifact |
| `tessera.launch(...)` | Launch through the available runtime path |
| `tessera.available_backends()` | List backend capabilities |
| `tessera.backend_capabilities()` | Return backend capability records |
| `tessera.query_backend(name)` | Query one backend |
| `tessera.get_last_profile()` | Read the last runtime profile |

---

## Error Types

| Exception | Module | Raised when |
|-----------|--------|-------------|
| `TesseraConstraintError` | `tessera.compiler.constraints` | Structural constraint violated at decoration time |
| `TesseraEffectError` | `tessera.compiler.effects` | `deterministic=True` + unseeded random op |
| `TesseraJitError` | `tessera.compiler.jit` | Graph IR emission pipeline failure |
| `TesseraTargetError` | `tessera.compiler.gpu_target` | Invalid `GPUTargetProfile` parameters |
| `TesseraAttnConfigError` | `tessera.compiler.attn_lower` | Invalid `FlashAttnLoweringConfig` parameters |

---

## Phase Implementation Status

| API Symbol | Implemented | Phase |
|-----------|-------------|-------|
| `@tessera.jit` | ✅ | 1 |
| `@tessera.kernel` | ✅ | 1 |
| `tessera.Region[...]` | ✅ | 1 |
| `tessera.domain.Rect` | ✅ | 1 |
| `tessera.dist.Block` | ✅ | 1 |
| `tessera.dist.Replicated` | ✅ | 1 |
| `tessera.dist.Cyclic` | ✅ | implemented / scaffolded runtime |
| `tessera.array.from_domain` | ✅ | 1 |
| `DistributedArray.parts()` | ✅ | 1 |
| `tessera.constraint.*` | ✅ | 1 |
| `EffectLattice` | ✅ | 1 |
| `GPUTargetProfile` / `ISA` | ✅ | 3 |
| `FlashAttnLoweringConfig` | ✅ | 3 |
| `MockRankGroup` (testing) | ✅ | 1 |
| NCCL/RCCL collectives | partial | scaffolded / mock-runtime tests |
| TPU target profile and artifacts | ✅ | implemented / lit-testable |
| Runtime C ABI Python wrapper | ✅ | mock-runtime; hardware-runtime when C backend is built |
