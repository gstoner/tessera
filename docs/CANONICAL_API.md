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

## Top-level Tensor Factories

Ergonomic shortcuts that wrap `tessera.array.from_domain(...)` with `Replicated()`
distribution. Use these for single-rank examples and tests; for sharded
construction, call `tessera.array.from_domain` directly with an explicit `dist.*`.

| Factory | Returns | Equivalent to |
|---------|---------|---------------|
| `tessera.zeros(shape, dtype="fp32")` | `DistributedArray` | `array.from_domain(Rect(shape), dtype, Replicated(), fill="zeros")` |
| `tessera.ones(shape, dtype="fp32")` | `DistributedArray` | `... fill="ones"` |
| `tessera.randn(shape, dtype="fp32")` | `DistributedArray` | `... fill="randn"` |
| `tessera.empty(shape, dtype="fp32")` | `DistributedArray` | `... fill="empty"` |
| `tessera.full(shape, fill_value, dtype="fp32")` | `DistributedArray` | zeros + in-place `_data[...] = fill_value` |

```python
x = tessera.randn((4, 16, 512), dtype="fp32")   # Replicated
W = tessera.ones((512,), dtype="fp32")
assert x.shape == (4, 16, 512)
assert x.shard_spec.replicated
```

---

## `tessera.nn` — Two Surfaces (Functional + Stateful)

### Functional API (stateless)

Pass weights in, get arrays out. No `Parameter` ownership, no `Module`
subclassing. Available as `tessera.nn.<name>` and as
`tessera.nn.functional.<name>` (torch-style `from tessera.nn import functional as F`).

| Function | Signature | Composition |
|----------|-----------|-------------|
| `tessera.nn.linear` | `(x, W, bias=None) → array` | `gemm(x, W) + bias` |
| `tessera.nn.rms_norm` | `(x, weight=None, eps=1e-5) → array` | `rmsnorm(x) * weight` |
| `tessera.nn.swiglu` | `(x, W_gate, W_up, W_down) → array` | `gemm(silu(x @ W_gate) * (x @ W_up), W_down)` |
| `tessera.nn.multi_head_attention` | `(Q, K, V, num_heads, scale=None, causal=False, dropout_p=0.0, seed=None) → array` | reshape `[B,S,H*D] → [B,H,S,D]`, `flash_attn(...)`, reshape back |
| `tessera.nn.flash_attention` | alias for `tessera.ops.flash_attn` | identical signature |

```python
import tessera
from tessera.nn import functional as F

x = tessera.randn((4, 16, 512)).numpy()
W = tessera.ones((512,)).numpy()
y = F.rms_norm(x, weight=W)
```

### Stateful API (Tier 1 — owns parameters)

Subclass `Module` and override `forward`. `Parameter` and child `Module`
attributes are auto-registered via `__setattr__` (torch.nn pattern).

| Class | Constructor | Forward |
|-------|-------------|---------|
| `Module` | `Module()` (subclass + override `forward`) | `model(*args)` calls `forward` |
| `Parameter` | `Parameter(data \| shape=, dtype="fp32", requires_grad=True)` | wraps a `DistributedArray`; `.grad` slot for autodiff (Tier 2) |
| `Sequential` | `Sequential(*modules)` | chains `m₁(m₂(...))` |
| `ModuleList` | `ModuleList([...])` + `.append` / `.extend` | indexable; iterable |
| `ModuleDict` | `ModuleDict({"q": Linear(...), ...})` | keyed access; iterable |
| `Linear` | `Linear(in_features, out_features, bias=True, dtype="fp32")` | `x @ W (+ b)` via Kaiming-uniform init |
| `RMSNorm` | `RMSNorm(normalized_shape, eps=1e-5, dtype="fp32")` | learnable scale weight (init=1) |
| `LayerNorm` | `LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, bias=True, dtype="fp32")` | learnable scale + bias |
| `Embedding` | `Embedding(num_embeddings, embedding_dim, dtype="fp32")` | lookup with N(0, 0.02) init |
| `Dropout` | `Dropout(p=0.5, seed=None)` | gated by `self.training` |
| `MLP` | `MLP(dim, hidden_dim, dtype="fp32")` | SwiGLU block (W_gate / W_up / W_down) |
| `MultiHeadAttention` | `MultiHeadAttention(embed_dim, num_heads, bias=True, dropout_p=0.0, dtype="fp32")` | packed Q/K/V proj + output proj; `forward(Q, K=None, V=None, causal=False, scale=None, seed=None)` |

`Module` provides:

| Method | Purpose |
|--------|---------|
| `.parameters(recurse=True)` | iterate over `Parameter`s |
| `.named_parameters(prefix="", recurse=True)` | iterate `(name, Parameter)` pairs |
| `.children()` / `.named_children()` | direct sub-modules only |
| `.modules()` | self + all descendants (DFS) |
| `.state_dict()` | `{name: numpy.ndarray}` snapshot |
| `.load_state_dict(sd, strict=True)` | in-place copy back into existing `Parameter` handles |
| `.train(mode=True)` / `.eval()` | toggle `self.training` recursively |
| `.zero_grad()` | drop `Parameter.grad` for every parameter |

```python
import tessera
import numpy as np

class TransformerBlock(tessera.nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden):
        super().__init__()
        self.norm1 = tessera.nn.RMSNorm(dim)
        self.attn  = tessera.nn.MultiHeadAttention(dim, num_heads)
        self.norm2 = tessera.nn.RMSNorm(dim)
        self.mlp   = tessera.nn.MLP(dim, mlp_hidden)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

block = TransformerBlock(dim=32, num_heads=4, mlp_hidden=64)
y = block(np.random.randn(2, 8, 32).astype(np.float32))
sd = block.state_dict()                    # save
block2 = TransformerBlock(32, 4, 64)
block2.load_state_dict(sd)                 # restore
```

### Phase A4 additions (2026-05-09)

| Class | Notes |
|-------|-------|
| `SiLU`, `Sigmoid`, `GELU`, `ReLU`, `Tanh`, `Identity` | Stateless activation Modules — `act = nn.SiLU(); y = act(x)` torch-style. `Identity` returns input unchanged |
| `MultiHeadCrossAttention` | Subclass of `MultiHeadAttention` — `forward(query, key, value)` requires explicit K/V (no self-attention shortcut) |
| `RotaryEmbedding(head_dim, max_position, base)` | Owns precomputed `theta`; `forward(x, position=0)` calls `ops.rope` |
| `CastedLinear`, `CastedEmbedding` | Subclasses with extra `cast_dtype` arg; output is auto-cast post-forward |
| `CrossEntropyLoss(reduction='mean')` | `mean/sum/none` reductions; composes through `ops.softmax` + `ops.reduce` |
| `nn.utils.clip_grad_norm_(params, max_norm, norm_type=2.0)` | In-place `.grad` scaling; supports L1/L2/inf norms |

**Phantom (still raises `NotImplementedError` with roadmap pointer):**
`BatchNorm1d`, `Conv2d`, `LSTM`, `DynamicDepthwiseConv1d`, `KVCache`. Each
links to its phase in [`docs/audit/execution_roadmap.md`](audit/execution_roadmap.md).

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
| `tessera.ops.silu(x)` | `(array) → array` | Pure effect |
| `tessera.ops.swiglu(x, W_gate, W_up, W_down)` | `(array, array, array, array) → array` | Reference SwiGLU MLP block; lowers to fused MLP-block kernel where supported (Apple GPU `matmul→gelu`/`matmul→rmsnorm` pattern; fused `silu`-variant is **planned**, see [SwiGLU Performance Plan](#swiglu-performance-plan) below) |
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

## `tessera.graph` — Debug & Trace Namespace

Graph-level inspection helpers. Backed by `python/tessera/debug.py`. Documented
in detail in `docs/guides/Tessera_Debugging_Tools_Guide.md`.

| Symbol | Purpose |
|--------|---------|
| `tessera.graph.trace(value, ir_level="graph")` | Build a `GraphTrace` over the given JIT artifact, MLIR string, or op-descriptor list |
| `tessera.graph.debug_trace(value, ...)` | Capture a structured `DebugTrace` with bounded tensor summaries |
| `tessera.graph.debug_value(name, value, *, metadata=None)` | Tag a runtime value with a debug name |
| `tessera.graph.export_graphviz(value, ir_level="graph")` | Emit a GraphViz DOT string for the IR |
| `tessera.graph.replay_capture(value, **metadata)` | Capture environment + artifact hashes for replay |

The fuller debug surface (`check_grad`, `check_determinism`, `replay_manifest`,
tensor summaries, etc.) lives at `tessera.debug.*` — see the Debugging Tools Guide.

---

## `tessera.autotune` — Autotuner Facade

Public surface over `python/tessera/compiler/autotune_v2.py` (Bayesian
Optuna-TPE + Hyperband). The facade is a callable; methods hang off it.

| Symbol | Purpose |
|--------|---------|
| `tessera.autotune(...)` | Decorator/launcher that runs the Bayesian autotuner over a kernel |
| `tessera.autotune.load(...)` | Load a tuning run from the SQLite cache |
| `tessera.autotune.cache_key(device_class, kernel_id, config) → str` | Compute the canonical cache key |
| `tessera.autotune.schedule_artifact(...)` | Materialize a chosen schedule into a runtime artifact |
| `tessera.autotune.RooflineCostModel` | Roofline-based cost estimator used during pruning |

---

## `tessera.ops` Registry Helpers

The `tessera.ops` namespace is backed by an `_OperatorRegistry` carrying
three slots per op: `reference` (numpy fallback), `lowering` (artifact hook),
and `runtime_kernel` (compiled launch). Backends register into this surface.

| Symbol | Purpose |
|--------|---------|
| `tessera.ops.registry` | The shared `_OperatorRegistry` instance |
| `tessera.ops.register_reference(name, fn, **metadata)` | Register/override the numpy reference for an op |
| `tessera.ops.register_lowering(name, fn, **metadata)` | Register an artifact-emission hook (Graph IR / target IR) |
| `tessera.ops.register_runtime_kernel(name, fn, **metadata)` | Register a compiled-runtime kernel hook |

Dispatch precedence in `_OperatorRegistry.dispatch(name, *, prefer_runtime=True)`:
`runtime_kernel` → `lowering` → `reference`. With `prefer_runtime=False`,
`reference` wins over `lowering`. Useful for backends that want a numerical
ground-truth path during development.

---

## SwiGLU Performance Plan

`tessera.ops.swiglu` ships today as a **numpy reference path**: three matmuls
plus a `silu`-and-multiply elementwise step. The lowering plan mirrors the
existing Apple GPU MLP-block fusions (`matmul→gelu`, `matmul→rmsnorm`):

| Stage | Target lowering | Status |
|-------|------------------|--------|
| 1. Reference correctness | numpy `gemm/silu/mul/gemm` | ✅ shipped |
| 2. Schedule IR fusion pattern | `matmul → silu_mul → matmul` recognized as a 3-op chain | 🔲 planned |
| 3. Apple GPU kernel | Fused MSL kernel mirroring `matmul→softmax→matmul` (two stack arrays for gate/up, third for output); f32 first, then f16/bf16 with fp32 accumulators | 🔲 planned |
| 4. NVIDIA / ROCm kernel | Fused WGMMA / MFMA epilogue with `silu*` activation | 🔲 planned |

Tracking: file under Phase 8.4.x follow-up (Apple GPU) and Phase 3 GPU
backend execution work (NVIDIA / ROCm) — both gated on Architecture Decision
#19 hardware-free Target IR layering.

---

## `tessera.autodiff` — Tape-based Reverse-Mode Autodiff (v1)

Tape-based reverse-mode at the numpy-reference op layer. Hooks into the Tier 1
`Parameter`/`Module` surface via a buffer-id registry — `Parameter.grad` is
populated automatically when a model's forward pass flows through `tessera.ops.*`
inside a `tape()` block. See `docs/spec/AUTODIFF_SPEC.md` for the design and
explicit deferrals (Graph/Tile IR adjoints, distributed grad collectives,
rematerialization, mixed-precision master-copy).

| Symbol | Purpose |
|--------|---------|
| `tessera.autodiff.tape()` | Context manager that records every `tessera.ops.<name>` call inside its `with` block |
| `Tape.backward(target, *, cotangent=None)` | Seed cotangent at `target` (a tape-recorded numpy array) and walk the tape in reverse populating `Parameter.grad` |
| `tessera.autodiff.reverse(fn)` | Decorator: `fn(*args, **kw) -> scalar_loss` becomes `(loss, {name: ndarray})`. Loss math must flow through `ops.*` |
| `tessera.autodiff.custom_rule(name)` | Decorator: register or override a VJP for `tessera.ops.<name>` |
| `tessera.autodiff.TesseraAutodiffError` | Raised on missing VJP on the gradient path, scalar-shape errors, double-backward, or target-not-on-tape |

```python
import numpy as np
import tessera as ts

mlp = ts.nn.MLP(dim=8, hidden_dim=16)
x = np.random.randn(2, 4, 8).astype(np.float32)
target = np.random.randn(2, 4, 8).astype(np.float32)

# Pattern A — explicit cotangent (loss math in raw numpy, common case)
with ts.autodiff.tape() as t:
    y = mlp(x)
    diff = y - target
    dy = (2.0 * diff / diff.size).astype(np.float32)
    t.backward(y, cotangent=dy)

# Manual SGD step
for p in mlp.parameters():
    p._data._data[...] -= 0.05 * p.grad.numpy()
mlp.zero_grad()
```

**Built-in VJPs (v1):** `gemm`/`matmul`, `add`, `mul`, `transpose`, `cast`, `relu`, `sigmoid`, `tanh`, `silu`, `gelu`, `softmax`, `layer_norm`, `rmsnorm`/`rmsnorm_safe`, `reduce`/`sum`, `dropout`. Calling any other `ops.<name>` inside a tape is allowed; if its result actually feeds the gradient, `Tape.backward` raises `TesseraAutodiffError` with a pointer to `custom_rule`.

**`Tape.backward` semantics:** the `target` must be a tape-recorded numpy output. Pass `cotangent=` when your loss math sits outside `ops.*` (raw numpy `(y - t)**2`-style); omit it when the target is itself a scalar from `ops.reduce` / `ops.sum`.

---

## Error Types

| Exception | Module | Raised when |
|-----------|--------|-------------|
| `TesseraConstraintError` | `tessera.compiler.constraints` | Structural constraint violated at decoration time |
| `TesseraEffectError` | `tessera.compiler.effects` | `deterministic=True` + unseeded random op |
| `TesseraJitError` | `tessera.compiler.jit` | Graph IR emission pipeline failure |
| `TesseraTargetError` | `tessera.compiler.gpu_target` | Invalid `GPUTargetProfile` parameters |
| `TesseraAttnConfigError` | `tessera.compiler.attn_lower` | Invalid `FlashAttnLoweringConfig` parameters |
| `TesseraAutodiffError` | `tessera.autodiff` | Missing VJP on gradient path; non-scalar `backward()` target without cotangent; double-backward; target not on tape |

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
| `tessera.zeros / ones / randn / empty / full` | ✅ | 1 |
| `tessera.nn.{linear, rms_norm, swiglu, multi_head_attention}` (functional) | ✅ (reference) | 1; fused lowerings planned |
| `tessera.nn.{Module, Parameter, Sequential, ModuleList, ModuleDict}` | ✅ | Tier 1 |
| `tessera.nn.{Linear, RMSNorm, LayerNorm, Embedding, Dropout, MLP, MultiHeadAttention}` (stateful) | ✅ | Tier 1 |
| `tessera.ops.swiglu` | ✅ (numpy reference) | 1; fused MLP-block kernel planned |
| `tessera.graph.*` debug namespace | ✅ | 1 |
| `tessera.autotune.*` facade | ✅ | 5 |
| `tessera.ops` registry (`register_reference / register_lowering / register_runtime_kernel`) | ✅ | 1 |
| `tessera.autodiff.tape / reverse / custom_rule` (v1, numpy-reference) | ✅ | Tier 2 — see `docs/spec/AUTODIFF_SPEC.md` |
| `tessera.autodiff.rematerialize / mixed_precision / Graph-IR adjoints / distributed adjoint collectives` | 🔲 planned | Phase 5 follow-ups |
| NCCL/RCCL collectives | partial | scaffolded / mock-runtime tests |
| TPU target profile and artifacts | ✅ | implemented / lit-testable |
| Runtime C ABI Python wrapper | ✅ | mock-runtime; hardware-runtime when C backend is built |
