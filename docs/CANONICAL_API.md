---
status: Normative
classification: Normative
last_updated: 2026-07-02
---

# Tessera Canonical API Quick Reference
**Status:** Normative - grounded in `python/tessera/` active implementation  
**Last updated:** July 2, 2026  
**Use this document** to resolve any naming disagreement in other docs. If something here conflicts with another doc, this file wins.

> **Tensor attribute vocabulary lives in [`docs/reference/tessera_tensor_attributes.md`](reference/tessera_tensor_attributes.md).** That document is the canonical reference for the six tensor attributes (`shape`, `dtype`, `layout`, `device`/`target`, `distribution`, `numeric_policy`), the canonical dtype names + accepted aliases, the planned/gated dtype set (`uint*`, `complex*`, packed `int4`, `mxfp*`), and the JAX-like promotion direction. Every dtype string used in this file (`"fp32"`, `"bf16"`, `"fp16"`, `"fp8_e4m3"`, etc.) follows that vocabulary; aliases like `"f32"` are accepted at API boundaries but should normalize before storage.

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

Constraints are checked at `@jit` decoration time when concrete `bindings`
are provided, and again at call time when symbolic tensor annotations can be
resolved from actual argument shapes.

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

Violation → `TesseraConstraintError`. Decoration-time checks fire when
concrete `bindings` are provided; call-time checks fire for symbolic shapes
resolved from actual tensor arguments.

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
| `ISA.SM_100` | B100 / GB200 (Blackwell datacenter) | ✅ | ✅ |
| `ISA.SM_120` | RTX 50-series — Blackwell consumer (GB20x, CC 12.0) | ❌¹ | ✅ |

`GPUTargetProfile` key parameters: `isa`, `warps_per_cta` (default 4, must be power of 2), `shared_mem_bytes` (None = SM default), `pipeline_stages` (default 2).

Key properties (**all sourced from the per-SM CUDA 13.3 feature matrix in
`gpu_target.py`, not a coarse `isa`-ordering**): `.supports_wgmma` / `.supports_tma`
/ `.supports_mbarrier` (SM_90 / SM_100; sm_120 has `tma`+`mbarrier` but **not**
`wgmma`); `.supports_tcgen05` / `.supports_tmem` / `.supports_cta_pairs` (datacenter
SM_100 only — **not** consumer sm_120); `.supports_block_scaled_mma` (SM_100 via
`tcgen05`, sm_120 via `mma.sync.block_scale`); `.runtime_arch` emits CUDA
architecture strings such as `sm_90a`, `sm_100a`, and `sm_120`.

> **¹ Consumer Blackwell sm_120 does not implement Hopper `wgmma`** (so
> `.supports_wgmma` is `False`). It is **not** a superset of datacenter sm_100:
> no `wgmma`, no `tcgen05`/TMEM. Its low-precision matrix path is 5th-gen Tensor
> Core `mma.sync` (incl. FP4 `block_scale`, compile target `sm_120a`). The
> `.supports_*` queries read the authoritative per-arch feature matrix, so they
> return the hardware-correct answer per ISA.

### String target aliases

| Input | Normalized target | Notes |
|-------|-------------------|-------|
| `None`, `"cpu"` | `cpu` | Executed via the `tessera_jit` MLIR→LLVM JIT lane for the covered op set (f32/f16/bf16/f64); numpy reference fallback otherwise |
| `"cuda"`, `"nvidia"`, `"gpu"`, `"sm90"`, `"sm_90"`, `"sm90a"`, `"sm_90a"`, `"hopper"` | `nvidia_sm90` | NVIDIA artifact target |
| `"rocm"`, `"amd"`, `"hip"` | `rocm` | ROCm Target IR artifact |
| `"apple_cpu"`, `"macos_cpu"`, `"m_series_cpu"` | `apple_cpu` | Apple CPU Target IR artifact |
| `"apple_gpu"` | `apple_gpu` | Apple GPU Target IR artifact |

---

## Dtype Annotations

> **The full tensor attribute vocabulary lives in [`docs/reference/tessera_tensor_attributes.md`](reference/tessera_tensor_attributes.md).** That document is the canonical source for: six tensor attributes (`shape`/`dtype`/`layout`/`device`/`distribution`/`numeric_policy`), the canonical dtype name table + accepted aliases, the planned/gated dtype set, the 5-rule Promotion And Casting Policy, and the JAX-like canonicalization direction. The summary tables below are derived from that source.

### Canonical dtype strings

| Family | Canonical | Accepted aliases (normalize at API boundary) |
|---|---|---|
| FP64 | `fp64` | `f64` |
| FP32 | `fp32` | `f32` — **default user-facing floating dtype** |
| FP16 | `fp16` | `f16` |
| BF16 | `bf16` | — (preferred reduced-precision dtype where target supports) |
| FP8 | `fp8_e4m3`, `fp8_e5m2` | — (target-gated) |
| FP6 | `fp6_e2m3`, `fp6_e3m2` | — (target-gated) |
| FP4 | `fp4_e2m1`, `nvfp4` | — (`nvfp4` is the NVIDIA block-scaled policy; **do not alias** to OCP FP4 / AMD MXFP4) |
| Signed integers | `int8`, `int16`, `int32`, `int64` | `i8`/`i16`/`i32`/`i64` in MLIR spellings |
| Boolean | `bool` | (lowers to `i1` in MLIR-like text) |

### Planned / gated (not first-class today)

`uint8`/`uint16`/`uint32`/`uint64`, `complex64`/`complex128`, packed `int4`,
AMD `mxfp8`/`mxfp6`/`mxfp4`,
and TF32 (which is **not** a storage dtype — model it as `math_mode="tf32"`
on `fp32` tensors via `numeric_policy`, not as `dtype="tf32"`).

### Tensor-attribute Python annotation shortcuts

| Annotation | Dtype | Notes |
|------------|-------|-------|
| `tessera.f16[..., ...]` | `fp16` | Read-only tensor annotation |
| `tessera.bf16[..., ...]` | `bf16` | Read-only tensor annotation |
| `tessera.f32[..., ...]` | `fp32` | Read-only tensor annotation |
| `tessera.mut_f32[..., ...]` | `fp32` | Write-privileged tensor annotation |

The single-letter `f16`/`bf16`/`f32` annotation names are the *Python-side
shortcuts* and should be considered aliases — when emitted into IR
metadata, the canonical `fp16`/`bf16`/`fp32` spellings are used.

### `numeric_policy` (sixth tensor attribute)

Storage dtype on a tensor is **explicit**; accumulator dtype, rounding,
quantization scale/axis, and determinism flags belong on `numeric_policy`,
not on `dtype`. For ops where storage and accumulator differ intrinsically
(matmul/gemm/einsum/flash_attn use `storage=bf16, accum=fp32`; quant ops use
`scale + quant_axis`), this distinction is required for correctness. See
the Promotion And Casting Policy section of the source document for the
five governing rules.

#### Scale *layout* for FP8/FP4 (`numeric_policy.scale_layout`)

For low-precision quantization, the dtype name and a scalar `scale` are not
enough — the **scale layout** is a compiler-visible contract (DeepGEMM-inspired).
`numeric_policy.scale_layout` is a `compiler.grouped_layout.ScaleLayout`:

| field | meaning |
|---|---|
| `granularity` | `per_tensor` / `per_row` / `per_channel` / `block` |
| `block` | `(rows, cols)` micro-block shape (block granularity) — e.g. `(1,128)` FP8, `(1,16)` NVFP4 |
| `packing` | packed scale element format: `none` / `e4m3` / `e5m2` / `e8m0` / `ue8m0` |
| `vector_size` | elements sharing one scale (16 for NVFP4, 128 for 1×128 block) |
| `alignment` | scale-tensor alignment (TMA needs aligned scales) |
| `transposed` | TMA-ready transposed (MN-major) scale layout |

`compiler.grouped_layout.scale_layout_for(dtype)` returns the canonical layout
per low-precision dtype; the FP8/FP4/NVFP4/INT8 quantizers carry it on their
`numeric_policy` in the audit registry.

### Grouped GEMM is a first-class op family (`grouped_layout`)

Grouped GEMM (MoE) carries an explicit `grouped_layout` contract
(`compiler.grouped_layout.GroupedLayout`) on its op metadata — the family is
modelled, not just runtime shape handling:

| field | meaning |
|---|---|
| `kind` | `dense` / `contiguous` (M-grouped) / `masked` (M-grouped fixed-shape) / `k_grouped` |
| `group_axis` | implied by `kind`: `M` (contiguous/masked), `K` (k_grouped), `None` (dense) |
| `alignment` | per-group alignment along the grouped axis (power of two; default 128) |
| `compiled_dims` | dims baked into the specialized kernel (default `N`,`K`) |
| `dynamic_dims` | dims left runtime-dynamic (`M` + `num_groups` for M-grouped) |

`tessera.grouped_gemm` is the M-grouped **contiguous** family. The masked /
K-grouped variants and the distributed MegaMoE fusion (`grouped_gemm → swiglu →
grouped_gemm`, then dispatch/combine overlap) are the planned follow-on rungs.

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
| `tessera.nn.functional.block_diffusion_attention` | `(x, x_ctx, *, q_proj, k_proj, v_proj, o_proj, num_heads, num_kv_heads, head_dim, q_norm=None, k_norm=None, cache_keys=None, cache_values=None, rope_fn=None, cache_offset=0, sliding_window=None, scale=None, eps=1e-6, attention_fn=None, return_ctx_kv=False) → array` | DFlash block-diffusion attention layer (QK-norm, KV injection, GQA, sliding-window-via-`attn_bias`); folds heads → rank-3 `flash_attn` (see [`tessera.dflash`](#tesseradflash--speculative-decoding-dflash)) |
| `tessera.nn.functional.mask_token_block` | `(prev_token, block_size, mask_token_id) → int64[..., block_size]` | DFlash draft input block `[prev, MASK, …]` |
| `tessera.nn.functional.linear_general` | `(x, W, bias=None, axis=-1) → array` | Axis-flexible (Flax-style) LinearGeneral/Einsum contraction |
| `tessera.nn.functional.lora_linear` | `(x, weight, lora_a, lora_b, bias=None, alpha=1.0) → array` | LoRA-adapted linear: `x@W + (alpha)·(x@A@B)` |
| `tessera.nn.functional.spectral_norm` | `(weight, eps=1e-12) → array` | Spectral normalization (top singular value) |
| `tessera.nn.functional.conv_transpose` | `(x, weight, bias=None, stride=1, padding=0, output_padding=0, dilation=1, groups=1) → array` | NCL grouped ConvTranspose1d reference |
| `tessera.nn.functional.{avg_pool, max_pool, min_pool}` | `(x, kernel_size, stride=None, padding=0) → array` | 1-D pooling references |
| `tessera.nn.functional.adaptive_pool` | `(x, output_size, reducer=mean) → array` | Adaptive pooling to a target size |
| `tessera.nn.functional.gru_cell` | `(x, h, W_ih, W_hh, b_ih=None, b_hh=None) → array` | One GRU cell step |
| `tessera.nn.functional.simple_rnn_cell` | `(x, h, W_ih, W_hh, bias=None, activation="tanh") → array` | One vanilla-RNN cell step |
| `tessera.nn.functional.bidirectional_scan` | `(fn, init_fwd, init_bwd, xs) → (fwd, bwd)` | Forward+reverse scan (bi-RNN substrate) |

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
| `LinearGeneral` | `LinearGeneral(in_shape, out_shape, axis=-1, bias=True, dtype="fp32")` | Flax-style axis-flexible linear over `linear_general` |
| `Einsum` | `Einsum(spec, weight_shape, dtype="fp32")` | learnable-weight einsum contraction |
| `LoRALinear` | `LoRALinear(in_features, out_features, rank, alpha=1.0, bias=True, dtype="fp32")` | base linear + low-rank `A@B` adapter |
| `ConvTranspose1d` / `ConvTranspose` | `ConvTranspose1d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True, dtype="fp32")` | NCL transposed conv (`ConvTranspose` is the alias) |
| `SpectralNorm` | `SpectralNorm(eps=1e-12)` | spectral-normalization wrapper |
| `GRUCell` / `SimpleRNNCell` | `GRUCell(input_size, hidden_size, bias=True, dtype="fp32")` · `SimpleRNNCell(input_size, hidden_size, bias=True, activation="tanh", dtype="fp32")` | stateful recurrent cells over `gru_cell` / `simple_rnn_cell` |
| `NativeSparseAttention` | `NativeSparseAttention(*, embed_dim, num_heads, window_size=64, block_size=16, top_k=2, compress_weight=False, causal=True, dtype="fp32")` | DeepSeek-style native sparse attention (sliding + compressed + top-k blocks) |
| `MixtureOfRecursions` | `MixtureOfRecursions(layer, *, embed_dim, max_depth=3, dtype="fp32")` | recursion-depth router over a shared layer |
| `MinimaxSparseAttention` | `MinimaxSparseAttention(*, embed_dim, num_heads, num_kv_heads, block_size, top_k, head_dim=None, force_local_block=True, causal=True, dtype="fp32")` · `MinimaxSparseAttention.from_gqa(*, embed_dim, num_heads, num_kv_heads, seq_len, block_size=64, sparsity=0.25, dense=False, ...)` | MiniMax Sparse Attention (arXiv:2606.13392) — GQA block-sparse layer: Index Branch top-k block selection per group + exact Main Branch over `ops.msa_sparse_attention`; `top_k == num_blocks` ⇒ dense GQA. See [docs/msa.md](msa.md) |

`Module` provides:

| Method | Purpose |
|--------|---------|
| `.parameters(recurse=True)` | iterate over `Parameter`s |
| `.named_parameters(prefix="", recurse=True)` | iterate `(name, Parameter)` pairs |
| `.buffers(recurse=True)` | iterate over `Buffer`s (Phase B1) |
| `.named_buffers(prefix="", recurse=True)` | iterate `(name, Buffer)` pairs |
| `.register_buffer(name, value, persistent=True)` | register a non-trainable named tensor (Phase B1) |
| `.children()` / `.named_children()` | direct sub-modules only |
| `.modules()` | self + all descendants (DFS) |
| `.state_dict()` | `{name: numpy.ndarray}` snapshot — includes Parameters and persistent Buffers |
| `.load_state_dict(sd, strict=True)` | in-place copy back into existing `Parameter` / `Buffer` handles |
| `.train(mode=True)` / `.eval()` | toggle `self.training` recursively |
| `.zero_grad()` | drop `Parameter.grad` for every parameter (does not touch buffers) |
| `.to(dtype)` | migrate every `Parameter` and persistent `Buffer` to a new dtype in place; returns `self` (Phase B3) |

### `Buffer` (Phase B1)

Non-trainable named tensors that ride alongside `Parameter`s in a `Module`.
Used for BatchNorm running stats, RoPE precomputed tables, attention masks
— anything that's part of the module's persistent state but doesn't receive
gradients.

| Aspect | `Buffer` | `Parameter` |
|--------|----------|-------------|
| `.grad` slot | ❌ | ✅ |
| `requires_grad` flag | ❌ | ✅ |
| Yielded by `parameters()` | ❌ | ✅ |
| Yielded by `buffers()` | ✅ | ❌ |
| Persisted to `state_dict()` | ✅ if `persistent=True` (default) | always |

```python
m.register_buffer("running_mean", np.zeros(64, dtype=np.float32))
m.register_buffer("scratch", np.empty(8), persistent=False)  # not in state_dict
```

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

### Phase C + D additions (2026-05-09)

| Class | Notes |
|-------|-------|
| `BatchNorm1d` (C1) | Running-stat buffers (`running_mean`/`running_var`/`num_batches_tracked`); train/eval modes |
| `KVCache` (C2) | Module wrapper around `tessera.cache.KVCacheHandle`; `forward(k, v)` returns full `(K, V)` so far |
| `DynamicDepthwiseConv1d` (D4) | Owns kernel `Parameter` + non-persistent `_state` `Buffer` (when `streaming=True`); both single-shot and chunked inference |
| `Conv2d` (H1) | NHWC default; HWIO weight layout; `(N, H, W, C)` → `(N, H_out, W_out, C_out)` |
| `Conv2dNCHW` (H1) | Torch-port shim — wraps `Conv2d` with explicit `(N, C, H, W)` ↔ NHWC transposes |
| `LSTMCell` (H2) | Single-step LSTM. `forward(x_t, (h_prev, c_prev))` → `(h_t, c_t)`; uses `ops.lstm_cell` + state extractors so BPTT works through the v1 tape |
| `LSTM` (H2) | Multi-step LSTM. `forward(x_seq, init_state=None)` → `(output_seq, (h_n, c_n))`. Unrolls in Python so the autodiff tape sees every timestep |

**No remaining phantoms.** Every `tessera.nn.*` name ships as a real
class. The `tests/unit/test_nn_module.py::TestRemainingPhantoms`
regression test asserts this state — adding a new phantom requires
explicit registration there.

---

## `tessera.ops` Namespace

The public namespace is a `SimpleNamespace` backed by a registry. Most
development-time implementations are NumPy reference paths or artifact
lowerings; runtime kernels can be registered separately.

| Op | Signature | Notes |
|----|-----------|-------|
| `tessera.ops.gemm(A, B)` | `(array, array) → array` | Matrix multiply via `np.matmul` |
| `tessera.ops.matmul(A, B)` | alias for `gemm` | |
| `tessera.ops.bmm(A, B, epilogue=None)` | `(array, array) → array` | Batched matmul (rank-3+); broadcasts a shared `[1,K,N]` B operand. Apple GPU `tessera_apple_gpu_bmm_{f32,f16}` (`metal_runtime`) |
| `tessera.ops.fake_quantize(x, ...)` | `(array, …) → array` | QAT fake-quantize with straight-through gradient (see `tessera.quantization`) |
| `tessera.ops.layer_norm(x, eps=1e-5)` | `(array) → array` | Pure effect |
| `tessera.ops.softmax(x, axis=-1)` | `(array) → array` | Pure effect |
| `tessera.ops.gelu(x)` | `(array) → array` | Pure effect |
| `tessera.ops.silu(x)` | `(array) → array` | Pure effect |
| `tessera.ops.swiglu(x, W_gate, W_up, W_down)` | `(array, array, array, array) → array` | Reference SwiGLU MLP block; lowers to a fused MLP-block kernel on Apple GPU (Phase 8.4.8 — the fused 4-op `silu`-variant `tessera_apple_gpu_swiglu_{f32,f16,bf16}` is **landed**; see [SwiGLU Performance Plan](#swiglu-performance-plan) below) |
| `tessera.ops.relu(x)` | `(array) → array` | Pure effect |
| `tessera.ops.transpose(x, axes=None)` | `(array) → array` | Pure effect |
| `tessera.ops.cast(x, dtype)` | `(array, str) → array` | Pure effect |
| `tessera.ops.dropout(x, p=0.1, training=True)` | `(array) → array` | `random` effect |
| `tessera.ops.conv2d(x, weight, bias=None, stride=1, padding=0)` | `(NHWC,HWIO) → NHWC` | NumPy reference convolution |
| `tessera.ops.flash_attn(Q, K, V, scale=None, causal=False, dropout_p=0.0, seed=None, attn_bias=None)` | `(array,array,array[,array]) → array` | NumPy reference attention; SM90+ lowering artifacts where supported. Optional additive `attn_bias` `(B,Sq,Sk)` → `softmax(scale·Q·Kᵀ + attn_bias)·V` (Apple GPU `metal_runtime` via `flash_attn_bias_*`) |
| `tessera.ops.all_reduce(x, op="sum")` | `(array) → array` | Single-rank no-op reference path |
| `tessera.ops.reduce_scatter(x, op="sum", axis=0)` | `(array) → array` | Single-rank no-op reference path |
| `tessera.ops.all_gather(x, axis=0)` | `(array) → array` | Single-rank no-op reference path |
| `tessera.ops.fused_epilogue(x, bias=None, activation="linear")` | `(array) → array` | |
| `tessera.ops.fft/ifft/rfft/irfft(...)` | `(array) → array` | NumPy FFT helpers |
| `tessera.ops.dct(x, type=2, axis=-1)` | `(array) → array` | NumPy FFT-derived DCT reference |
| `tessera.ops.spectral_conv(x, w)` | `(array,array) → array` | NumPy FFT convolution reference |
| `tessera.ops.rmsnorm_safe(x, eps=1e-6)` | `(array) → array` | NumPy RMSNorm reference |
| `tessera.ops.kv_cache_append(cache, key, value)` | cache helper | Reference KV-cache helper; dispatches handle vs. legacy |
| `tessera.ops.kv_cache_update(cache, key, value)` | cache helper | Modern alias for `kv_cache_append` (Phase E2) |
| `tessera.ops.kv_cache_prune(cache, max_entries=None, max_seq=None)` | cache helper | Reference KV-cache helper |
| `tessera.ops.kv_cache_read(cache, start, end=None)` | cache helper | Read slice as `(K, V)` (Phase B2); dequantizes for quantized handles |
| `tessera.ops.quantize_kv(k, v, *, bits=4, symmetric=True)` | quantization | Phase E1 — block-quantize K/V to int8; returns `(k_q, v_q, scale, zero_point)` |
| `tessera.ops.dequantize_kv(k_q, v_q, scale, zero_point=None, *, symmetric=True)` | quantization | Inverse of `quantize_kv` — returns `(k, v)` |
| `tessera.ops.depthwise_conv1d(x, w, *, kernel_size, padding=0, causal=False, state=None)` | streaming conv | Phase D1 — depthwise 1-D conv with optional streaming state |
| `tessera.ops.depthwise_conv2d(x, w, *, kernel_size, stride=(1,1), padding=(0,0), causal=False)` | streaming conv 2D | Phase D3 follow-up — depthwise 2-D NHWC conv; VJP shipped |
| `tessera.ops.lstm_cell(x_t, h_prev, c_prev, W_ih, W_hh, b_ih=None, b_hh=None)` | RNN state primitive | Phase H2 — single-step LSTM. Returns packed ``concat([h_t, c_t], axis=-1)`` |
| `tessera.ops.lstm_state_h(packed)` / `lstm_state_c(packed)` | RNN state extractors | Phase H2 — autodiff-traced extractors for the packed lstm_cell output |
| `tessera.ops.online_softmax(x, *, axis=-1, state=None)` | streaming softmax | Phase D2 — single-chunk equiv. to `softmax`; pass `state` for streaming chunks |
| `tessera.ops.online_softmax_state(x, *, axis=-1, state=None)` | helper | Returns `(running_max, running_sum)` for the next streaming `online_softmax` call (non-differentiable) |
| `tessera.ops.selective_ssm(x, A, B, C, delta, *, gate=None, state=None, chunk_size=128)` | Mamba2 SSM | Phase D3 — selective state-space; chunked scan; optional output gate + initial state. **Forward + reverse-mode VJP shipped** (numerical-Jacobian verified at fp64). |

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

## `tessera.distributed.DDP` / `FSDP` — Phase I

Python wrappers that apply distributed-gradient collectives on each backward
pass. Forward is unchanged; the wrapper exposes `sync_grads(rank)` (DDP) or
`shard / gather_for_forward / reshard_after_forward / sync_grads` (FSDP)
for per-rank state management. Today's implementation runs against the
`tessera.testing.mock_collective.MockRankGroup` in-process simulator; real
NCCL/RCCL bindings land alongside Phase G's NVIDIA execution path.

| Class | Constructor | Per-rank methods |
|-------|-------------|------------------|
| `tessera.distributed.DDP` | `DDP(module, mesh_axis="dp")` | `forward(*args)` (passthrough), `sync_grads(rank)` (mean all-reduce) |
| `tessera.distributed.FSDP` | `FSDP(module, mesh_axis="dp")` | `shard(rank)`, `gather_for_forward(rank)`, `reshard_after_forward(rank)`, `sync_grads(rank)` (reduce-scatter to local shard) |

Both wrappers require the wrapped value to be a `tessera.nn.Module`. FSDP
shards along the leading dim; non-leading-dim sharding requires an explicit
transpose before wrapping. **Each rank in a real distributed run holds its
own Module instance** (matches torch FSDP's per-process model); the mock
collective tests construct one wrapper per worker.

---

## `tessera.cache` — KV-cache handle (Phase B2)

Opaque handle for paged KV-cache state. Replaces the legacy
`ReferenceKVCache` (which stays live for backward compat) with a
fixed-allocation, max-seq-bounded paged buffer that future backends will
lower as a first-class state value.

| Symbol | Purpose |
|--------|---------|
| `tessera.cache.KVCacheHandle(num_heads, head_dim, max_seq, dtype="fp32", page_size=128)` | Construct a fresh cache handle |
| `cache.append(k, v)` / `tessera.ops.kv_cache_append(cache, k, v)` | Append a chunk of `(seq, num_heads, head_dim)` (or packed `(seq, num_heads*head_dim)`) tokens |
| `cache.read(start, end=None)` / `tessera.ops.kv_cache_read(cache, start, end=None)` | Slice K/V views from the cache (single-token if `end` omitted) |
| `cache.prune(max_entries)` / `tessera.ops.kv_cache_prune(cache, max_entries=N)` | Keep only the trailing `max_entries` tokens (sliding window) |
| `.current_seq` / `.shape` / `.is_full` | Read-only metadata |

```python
import numpy as np
import tessera as ts

cache = ts.cache.KVCacheHandle(num_heads=8, head_dim=64, max_seq=2048)
k = np.random.randn(1, 8, 64).astype(np.float32)
v = np.random.randn(1, 8, 64).astype(np.float32)
cache = ts.ops.kv_cache_append(cache, k, v)
k_now, v_now = ts.ops.kv_cache_read(cache, 0, cache.current_seq)
```

`page_size` is recorded but not yet used to physically page — Phase E adds
real paging + block quantization. User code written today doesn't need to
change when that lands. See [`docs/audit/coverage/COVERAGE_AUDIT.md`](audit/coverage/COVERAGE_AUDIT.md)
for per-backend lowering status.

---

## `tessera.dflash` — Speculative Decoding (DFlash)

Block-diffusion speculative-decoding draft ([z-lab/dflash](https://github.com/z-lab/dflash))
on the `attn_bias` substrate. Python reference; attention core on Apple GPU
`metal_runtime`. Greedy spec-decode output == greedy autoregressive decode (proven
vs the MLX reference). Full overview: [`docs/dflash.md`](dflash.md);
spec: [`PYTHON_API_SPEC.md` §18](spec/PYTHON_API_SPEC.md).

Canonical names (one per concept):

| Concept | Canonical name |
|---------|----------------|
| Block-diffusion attention layer | `tessera.nn.functional.block_diffusion_attention` |
| Draft input block `[prev, MASK…]` | `tessera.nn.functional.mask_token_block` |
| Draft config / weights | `tessera.dflash.DFlashConfig` / `DFlashLayerWeights` / `DFlashWeights` |
| Stateful draft module | `tessera.dflash.DFlashDraft` (`nn.Module`) |
| Draft KV cache | `tessera.dflash.DraftKVCache` / `RotatingDraftKVCache` |
| RoPE factory | `tessera.dflash.make_rope` |
| Multi-layer target tap | `tessera.dflash.HiddenStateTap` / `capture_target_hidden` |
| Sampler | `tessera.dflash.make_sampler` (greedy/temp/top-k/top-p) |
| Greedy / rejection acceptance | `tessera.dflash.dflash_linear_verify` / `dflash_speculative_verify` |
| Efficient generation loop | `tessera.dflash.dflash_generate_cached` |
| Training loss | `tessera.dflash.dflash_block_loss` (+ `dflash_block_loss_grad`) |
| Apple GPU attention seam | `tessera.dflash.apple_gpu_attention_fn` |
| Reference target model | `tessera.dflash_reference.ReferenceDecoderLM` (stateful KV cache + `rollback`) |
| Checkpoint I/O | `tessera.dflash_io.load_dflash_weights` / `save_dflash_weights` |
| Serving | `tessera.dflash_serve.dflash_generate_text` / `DFlashScheduler` |

```python
from tessera import dflash as D
from tessera import dflash_reference as R
from tessera.dflash_serve import DFlashScheduler

target = R.random_decoder_lm(lm_cfg, rng)              # or a real target LM
sched = DFlashScheduler(draft_weights, cfg, target)    # draft_weights: DFlashWeights
ids = sched.generate(prompt_ids, max_new_tokens=64)    # greedy == autoregressive
```

---

## `tessera.models` — Production Model Graphs (experimental)

Compiler-visible, dimension-checked model graphs built from Tessera primitives.
Today: **DiffusionGemma**, a Gemma-4-calibrated block-diffusion MoE text model as
a *shape-only* graph + config-aware verifier (the contract layer the
runtime/kernel lowering builds on) plus numpy-reference routing/sampling/decode.
Full spec: [`PYTHON_API_SPEC.md` §19](spec/PYTHON_API_SPEC.md).

| Concept | Canonical name |
|---------|----------------|
| Model config (Gemma-4-calibrated) | `tessera.models.DiffusionGemmaConfig` |
| One text layer: shape-only graph | `tessera.models.build_text_block` → `TextBlockGraph` (`GraphNode` list) |
| Config / graph / LM-head / budget verifiers | `tessera.models.verify_config` / `verify_text_block` / `verify_lm_head` / `verify_param_budget` |
| Param-budget estimate | `tessera.models.estimated_param_counts` |
| MoE top-k routing + pack/combine (reference) | `tessera.models.route_top_k` / `plan_packing` / `pack_tokens` / `unpack_combine` / `moe_forward` |
| Entropy-bound sampler | `tessera.models.entropy_bound_sample` (`SamplerConfig` / `temperature_schedule`) |
| Block-diffusion step graph + decode loop | `tessera.models.build_block_diffusion_step` / `run_block_diffusion_step` / `BlockDiffusionDecoder` |
| Quantization / vision staging manifests | `tessera.models.plan_quantization` / `default_vision_metadata` / `import_model_metadata` (`ModelManifest`) |
| Gemma logit soft-cap op | `tessera.ops.softcap(x, *, cap)` — `cap·tanh(x/cap)`, VJP+JVP |

```python
from tessera.models import DiffusionGemmaConfig, build_text_block, verify_text_block
cfg = DiffusionGemmaConfig()                    # Gemma 4 26B A4B card defaults
graph = build_text_block(cfg)                   # shape-only GraphNode list
verify_text_block(graph, cfg)                   # rejects dim mismatches before any runtime
```

## `tessera.diffusion_guidance` — Guided Diffusion (CGG v1)

Forward-only guided diffusion utilities. CGG combines denoiser scores at test
time; it does **not** require a reward model or autodiff. Objective-gradient
guidance is intentionally a later contract because it needs gradients through
`x0_pred`.

| Concept | Canonical name |
|---------|----------------|
| Schedule metadata | `tessera.diffusion_guidance.DiffusionSchedule` |
| Denoiser result | `tessera.diffusion_guidance.DenoiseOutput` |
| One favored/unfavored pair | `tessera.diffusion_guidance.ContrastivePair` |
| CGG score composition | `tessera.diffusion_guidance.ContrastiveScoreGuidance` |
| Compiler-visible combine primitive | `tessera.ops.score_combine(base, delta, gamma=...)` |
| Compiler-visible orchestration marker | `tessera.guided_denoise_region` Graph IR op |
| Deterministic guided sampler | `tessera.diffusion_guidance.GuidedDiffusionSampler` |
| Deferred look-ahead contract | `tessera.diffusion_guidance.ObjectiveGradientGuidance` |

CGG composition:

```python
s_guided = s_ref + sum(gamma_i * (s_favored_i - s_unfavored_i))
# library implementation composes each pair through:
s_next = tessera.ops.score_combine(s_prev, s_favored_i - s_unfavored_i, gamma=gamma_i)
```

For PO-vs-reference guidance, set `unfavored="base"` to get
`(1 - gamma) * s_ref + gamma * s_po`.

Examples:

```bash
PYTHONPATH=python python3 examples/diffusion_guidance/cgg_diffusion_gemma.py --out /tmp/cgg_report.json
PYTHONPATH=python python3 examples/diffusion_guidance/cgg_benchmark.py --smoke --out /tmp/cgg_benchmark.json
```

Compiler boundary:

- `tessera.score_combine` is the executable primitive; it verifies equal ranked
  floating score tensor shapes and lowers to `base + gamma * delta` through
  `tessera-to-linalg`.
- `tessera.guided_denoise_region` is a metadata/audit marker for the enclosing
  guided-denoise orchestration. It records timestep, schedule, gamma, and
  optional preference label, but it is not a fused model kernel.

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

**Built-in VJPs (v1 + Phase D + F3):** `gemm`/`matmul`, `add`, `mul`, `transpose`, `cast`, `relu`, `sigmoid`, `tanh`, `silu`, `gelu`, `softmax`, `layer_norm`, `rmsnorm`/`rmsnorm_safe`, `reduce`/`sum`, `dropout`, `depthwise_conv1d` (Phase D1), `online_softmax` (D2), **`flash_attn`** (F3), **`fft`** / **`ifft`** / **`rfft`** / **`irfft`** (F3). Calling any other `ops.<name>` inside a tape is allowed; if its result actually feeds the gradient, `Tape.backward` raises `TesseraAutodiffError` with a pointer to `custom_rule`. Notable still-unsupported: `moe` (depends on routing — register your own).

### Phase F1 — `tessera.autodiff.autocast` + `GradScaler`

Mixed-precision training. `with autocast("fp16"):` casts op-input arrays to
fp16 before the underlying op runs; reductions and norms (`softmax`,
`layer_norm`, `rmsnorm`, `reduce`, `sum`, `online_softmax`) are *up-cast* to
fp32 for numerical stability — matches torch's default policy.

| Symbol | Purpose |
|--------|---------|
| `tessera.autodiff.autocast(dtype="fp16")` | Context manager — casts/promotes op inputs |
| `tessera.autodiff.autocast_dtype()` | Inspect the active autocast dtype (or `None`) |
| `tessera.autodiff.GradScaler(init_scale=2**16, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000)` | Loss-scaling helper |
| `scaler.scale_loss(loss)` / `scaler.scale_grad(dy)` | Multiply loss / cotangent by scale |
| `scaler.step(optimizer_fn, *, params)` | Unscale, check inf/nan, run optimizer (returns `True` if step taken, `False` on overflow) |

### Phase F2 — `tessera.autodiff.rematerialize` / `checkpoint`

Activation checkpointing. `@rematerialize` on a function (or `with` block) drops its intermediate ops from the outer tape; on backward, the function is re-run inside a nested tape and gradients are extracted from there. Trade compute for memory.

```python
@tessera.autodiff.rematerialize
def expensive_block(x):
    return mlp(x)

with tessera.autodiff.tape() as t:
    y = expensive_block(input)
    t.backward(y, cotangent=dy)
```

### Phase F4 — Graph IR autodiff (ODS + pass body + per-op `buildAdjoint` impls landed; lit verified on MLIR 22)

`Tessera_AdjointInterface` ODS at `src/compiler/ir/include/Tessera/AdjointInterface.td` with tablegen target `TesseraAdjointInterfaceTableGen`. Per-op `buildAdjoint` C++ impls in `src/compiler/ir/AdjointInterface.cpp` for `MatmulOp`, `LayerNormOp`, `SoftmaxOp`, and pointwise activations (GELU/ReLU/Sigmoid/Sin) — pointwise ops route via the Python registry through the new `tessera.custom_adjoint_call` placeholder op. The full reverse-walk body is in `src/transforms/lib/AutodiffPass.cpp` and registered as `--tessera-autodiff` (also packaged as the `tessera-autodiff-pipeline` together with F5).

### Phase F5 — Effect-aware adjoint collective insertion

`AdjointCollectiveInsertionPass` (`--tessera-adjoint-collective-insertion`) runs after AutodiffPass. For each function arg with a recorded cotangent + sharding declaration, plans the appropriate distributed-gradient collective (`reduce_scatter` / `all_gather` / `all_reduce`) and records the choice as a per-arg `tessera.adjoint_collective_plan` attribute. Real op insertion follows once F4's multi-output rewrite step is in. Options: `--dp-axis=` (default `dp`) and `--tp-axis=` (default `tp`).

The Python tape (above) remains the production path until the MLIR build runs `cmake -DTESSERA_BUILD_TESTS=ON`; both surfaces will share the `tessera.autodiff.custom_rule` registry. See `docs/spec/AUTODIFF_SPEC.md` §Phase F4.

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
| Runtime C ABI Python wrapper | ✅ | CPU reference/mock path; native CPU C ABI when built; Apple CPU/GPU runtime-backed via Accelerate / MPS / MPSGraph / MSL |
