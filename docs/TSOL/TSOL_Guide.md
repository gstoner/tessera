
# Tessera Standard Operator Library (TSOL)

This document catalogs core Tessera operators and includes **Python type stubs** for IDEs.

1. Introduction

TSOL is Tessera’s curated set of high-performance, portable primitives with stable semantics and back-end–aware implementations. It covers:

- Linear algebra (BLAS-like + factorized ops)
- NN primitives (convs, attention, normalization, MoE)
- Spectral/FFT operators
- Sparse & segment/graph ops
- Randomness & dropout
- Collectives hooks (NCCL/NVSHMEM/DeepEP aware)
- Layout transforms & packing

Design goals

- Portable & deterministic: well-defined accuracy and reduction order (when requested).
- Fusion-friendly: canonical epilogues (+bias, activation, residual).
- Autotuned: persistent cache per (arch, dtype, shape, layout) with on-device measurements.
- IR-native lowering: Graph→Schedule→Tile→Target mapping with clear shape/layout contracts.

⸻

2. Namespaces & Imports

Python DSL
```python
from tessera import op, dist, layout, rng
```
C++/C API (example header below)
```cpp
#include <tessera/ops.hpp>
```
---
3. Tensor, Dtype, Layout (Normative)

- Tensor: rank-N array with dtype, layout, optional distribution (ShardSpec) and alignment.
- Dtype: fp32, bf16, fp16, fp8_e4m3, fp8_e5m2, int8, int32.
- Accumulator: unless specified, matmul/conv/attention accumulate in fp32, with epilogue cast to output dtype.
- Layouts:
- Dense: row_major, col_major
- Packed/tiled: tile(BM, BN, BK), channels_last (NHWC)
- Block-sparse: bsr(block_m, block_n, mask)
- Sequence: NLD or BLH (batch, length, hidden)

Determinism knobs
- deterministic=True enforces fixed reduction tree + ordered send/recv and disables numerically-unstable fast paths.

4.0 Operators 

4.1 Linear Algebra
| Operator | Python Signature | Notes |
|---|---|---|
| GEMM | `op.matmul(A, B, *, epilogue=None)` | Fused epilogues: `bias`, `activation`, `residual` |
| Batched GEMM | `op.batched_gemm(A, B)` | Strided or pointer arrays |
| Einsum | `op.einsum(spec, *tensors)` | Lowers to contractions + permutes |
| Factorized Matmul | `op.factorized_matmul(A, B, rank=k)` | Low-rank trade-offs |
| Triangular Solve | `op.tri_solve(A, b, lower=True)` |  |
| Decompositions | `op.cholesky/qr/svd` | CPU fallback when needed |

Best practices
- Prefer epilogue= fusion ("bias_relu", "bias_silu_residual") to save bandwidth.
- For tall/skinny GEMM, tune Tile IR BM/BN/BK and warp count; let autotuner cache.

⸻

4.2 NN Primitives
| Op  |Signature| Notes |
|------------------|----------------------------------------------|------------------------|
|conv2d/3d| op.conv2d(x, w, stride, padding, layout="nhwc") | Fused +bias, act|
|layernorm/rmsnorm| op.layernorm(x, eps) / op.rmsnorm(x, eps)| Deterministic reductions|
|dropout| op.dropout(x, p, rng=...) |Philox counter-based |
|Attention |       |
| qkv_projection| op.qkv_projection(x, W_qkv) |TP-friendly|
|flash_attention| op.flash_attention(q,k,v,*, causal=False, block_q, block_k, block_d)| Double-buffering + softmax-stability|
|rope| op.rope(x, theta, axes="qk")| |
|MoE | |
|moe |op.moe(x, experts, router="topk", k=2, transport=..., deterministic=...) |DeepEP/NVSHMEM fast path|
|moe_dispatch/combine | exposed for manual routing | Pack/route/combine procs |

4.3 Spectral / FFT


| Operator | Python Signature | Notes |
|---|---|---|
| FFT/IFFT | `op.fft(x, axes=...)` / `op.ifft(xf, axes=...)` | Batched, multi-dim |
| RFFT/IRFFT | `op.rfft(x)` / `op.irfft(xf)` | Real transforms |
| STFT/ISTFT | `op.stft(x, win, hop)` | Windowing built-in |
| Spectral Filter | `op.spectral_filter(Xf, Hf)` | Complex dtype aware |

Best practices
- Prefer batched FFT with contiguous per-batch memory; let tuner choose pencil decomposition for multi-GPU.

4.4 Sparse & Segment/Graph

| Operator | Python Signature | Notes |
|---|---|---|
| SPMM (COO/CSR) | `op.spmm_coo(A_coo, B)` | Coalesced loads |
| SDDMM | `op.sddmm(A, B, mask)` | For attention sparsity |
| Block-Sparse GEMM | `op.bsmm(X, W_bsr)` | BSR: (block_m, block_n, mask) |
| Segment Reduce | `op.segment_reduce(x, seg_ids, op="sum|max|mean")` | |

Best practices
- For block-sparse, align block size to TC tile to keep MMA utilization.

⸻

4.5 Randomness & Init

| Operator | Python Signature | Notes |
|---|---|---|
| RNG Uniform | `rng.uniform(shape, dtype="fp32", seed=..., counter=...)` | Stateless Philox |
| RNG Normal | `rng.normal(shape, dtype="fp32", seed=..., counter=...)` | |
| Dropout | `op.dropout(x, p, rng=...)` | Deterministic with fixed seed |

Best practices
- Use counter-based RNG with (seed, subsequence) per stream to avoid overlap.

⸻

4.6 Collectives (hooks)
| Operator | Python Signature | Default Backend |
|---|---|---|
| All-Reduce | `dist.all_reduce(x, axis="dp", deterministic=True)` | NCCL |
| Reduce-Scatter | `dist.reduce_scatter(x, axis="dp", deterministic=True)` | NCCL |
| All-Gather | `dist.all_gather(x, axis="dp", deterministic=True)` | NCCL |
| MoE Dispatch/Combine | `op.moe_dispatch / op.moe_combine` | NVSHMEM + DeepEP |

**See also**: `Tessera_Collectives_Distributed.md` for determinism, transports (NCCL/NVSHMEM/DeepEP), and tuning.


4.7 Layout & Packing

| Operator | Python Signature | Notes |
|---|---|---|
| Transpose | `op.transpose(x, perm)` | |
| Rearrange | `op.rearrange(x, layout)` | Dense or tiled |
| Pack/Unpack | `op.pack(x, layout)` / `op.unpack(x)` | For tile/block views |
| Tile View | `op.tile_view(x, BM, BN, BK)` | Feeds Tile IR autotuner |

# 5. Operator Catalog (Concise Reference)

## Linear Algebra
- `op.matmul(A, B, *, epilogue=None)` → `(M×K)·(K×N)`
- `op.batched_gemm(A, B)`
- `op.einsum(spec, *tensors)`
- `op.factorized_matmul(A, B, rank)`
- `op.tri_solve(A, b)`
- `op.cholesky/qr/svd(A)`

## NN Primitives
- `op.conv2d/3d(x, w, stride, padding, layout)`
- `op.layernorm/rmsnorm(x, eps)`
- `op.dropout(x, p)`
- `op.qkv_projection(x, W_qkv)`
- `op.flash_attention(q, k, v, causal=False, block_q, block_k, block_d)`
- `op.moe(x, experts, router, k, transport, deterministic)`

## Spectral
- `op.fft/ifft/rfft/irfft(x, axes)`
- `op.spectral_filter(Xf, Hf)`
- `op.stft/istft(x, win, hop)`

## Sparse/Graph
- `op.spmm_coo(A_coo, B)`
- `op.sddmm(A, B, mask)`
- `op.bsmm(X, W_bsr)`
- `op.segment_reduce(x, seg_ids, op)`

## RNG
- `rng.uniform(shape, dtype, seed, counter)`
- `rng.normal(shape, dtype, seed, counter)`

## Collectives
- `dist.all_reduce(x, axis, deterministic=True)`
- `dist.reduce_scatter(x, axis, deterministic=True)`
- `dist.all_gather(x, axis, deterministic=True)`

6. Error Handling (Normative)

- All ops can raise tessera.error with code and what():
- TS_ERR_INVALID_ARG, TS_ERR_SHAPE_MISMATCH, TS_ERR_UNSUPPORTED_DTYPE,
- TS_ERR_BACKEND_FAILURE (wraps NCCL/NVSHMEM/etc.), TS_ERR_OOM.
- Deterministic-mode violations raise TS_ERR_NONDETERMINISM.

⸻

7. Performance Notes (Informative)

- Let the autotuner warm up; caches per (arch, dtype, shape, layout).
- Prefer fused epilogues and intra-op layout matches to avoid transposes.
- Use streams to overlap H2D/D2H with compute; pin host buffers.

⸻

8. Usage Snippets

Fused GEMM epilogue
```python 
y = op.matmul(x, w, epilogue="bias_silu_residual", bias=b, residual=skip)
```
FlashAttention
```python 
y = op.flash_attention(q, k, v, causal=True, block_q=128, block_k=128, block_d=128)
```
MoE with NVSHMEM/DeepEP
```python 
y = op.moe(x, experts, router="topk", k=2,
           transport={"type":"nvshmem","multi_qp":True,"pack_dtype":"fp8_e4m3"},
           deterministic=True)
```
Block-sparse matmul
```python 
y = op.bsmm(x, w_bsr)  # bsr: (block_m, block_n, mask, values)
```
## Python Type Stubs (`tessera/ops.pyi`)

> Save as `tessera/ops.pyi` in your source tree or site-packages to enable autocompletion and static checking.

```python
from typing import Literal, Sequence, Tuple, Optional, Iterable, Any, overload, TypedDict

DType = Literal["fp32","bf16","fp16","fp8_e4m3","fp8_e5m2","int8","int32"]
Layout = Literal["row_major","col_major","nhwc","nchw","tile","bsr"]
Act = Literal["none","relu","silu","gelu"]

class TileParams(TypedDict, total=False):
    BM: int
    BN: int
    BK: int

class BsrMeta(TypedDict, total=False):
    block_m: int
    block_n: int
    mask: "memoryview|bytes"
    mask_shape: Tuple[int, int]

class Tensor: ...
# Minimal protocol-like hints for IDEs
def tensor(shape: Tuple[int, ...], *, dtype: DType="bf16", layout: Layout="row_major",
           meta: Optional[dict]=..., device: Optional[str]=..., requires_grad: bool=False) -> Tensor: ...

class Epilogue(TypedDict, total=False):
    add_bias: bool
    bias: Tensor
    activation: Act
    add_residual: bool
    residual: Tensor

class Determinism(TypedDict, total=False):
    deterministic: bool
    reduce_tree: Literal["binary","flat","ring"]
    send_order: Literal["lexicographic","rank","auto"]

class Transport(TypedDict, total=False):
    type: Literal["auto","nccl","nvshmem"]
    multi_qp: bool
    pack_dtype: DType

# Linear Algebra
def matmul(A: Tensor, B: Tensor, *, epilogue: Optional[Epilogue]=...) -> Tensor: ...
def batched_gemm(A: Tensor, B: Tensor) -> Tensor: ...
def einsum(spec: str, *tensors: Tensor) -> Tensor: ...
def factorized_matmul(A: Tensor, B: Tensor, *, rank: int) -> Tensor: ...
def tri_solve(A: Tensor, b: Tensor, *, lower: bool=True) -> Tensor: ...
def cholesky(A: Tensor) -> Tensor: ...
def qr(A: Tensor) -> Tuple[Tensor, Tensor]: ...
def svd(A: Tensor) -> Tuple[Tensor, Tensor, Tensor]: ...

# NN Primitives
def conv2d(x: Tensor, w: Tensor, *, stride: Tuple[int,int]=(1,1), padding: Tuple[int,int]=(0,0),
           layout: Literal["nhwc","nchw"]="nhwc", epilogue: Optional[Epilogue]=...) -> Tensor: ...
def layernorm(x: Tensor, *, eps: float=1e-5) -> Tensor: ...
def rmsnorm(x: Tensor, *, eps: float=1e-5) -> Tensor: ...
def dropout(x: Tensor, p: float, *, rng: Any=...) -> Tensor: ...
def qkv_projection(x: Tensor, W_qkv: Tensor) -> Tuple[Tensor, Tensor, Tensor]: ...

class FlashParams(TypedDict, total=False):
    causal: bool
    block_q: int
    block_k: int
    block_d: int

def flash_attention(q: Tensor, k: Tensor, v: Tensor, *, params: Optional[FlashParams]=..., 
                    deterministic: Optional[Determinism]=...) -> Tensor: ...

# MoE
def moe(x: Tensor, experts: Iterable[Any], *, router: Literal["topk","hash"]="topk", k: int=2,
        transport: Optional[Transport]=..., deterministic: Optional[Determinism]=...) -> Tensor: ...
def moe_dispatch(x: Tensor, route: Tensor, *, transport: Optional[Transport]=...) -> Tensor: ...
def moe_combine(partials: Tensor, inverse_route: Tensor, *, reduce: Literal["sum","mean"]="sum") -> Tensor: ...

# Spectral / FFT
def fft(x: Tensor, *, axes: Optional[Sequence[int]]=...) -> Tensor: ...
def ifft(xf: Tensor, *, axes: Optional[Sequence[int]]=...) -> Tensor: ...
def rfft(x: Tensor, *, axes: Optional[Sequence[int]]=...) -> Tensor: ...
def irfft(xf: Tensor, *, axes: Optional[Sequence[int]]=...) -> Tensor: ...
def spectral_filter(Xf: Tensor, Hf: Tensor) -> Tensor: ...

# Sparse & Segment
def spmm_coo(A_coo: Tensor, B: Tensor) -> Tensor: ...
def spmm_csr(A_csr: Tensor, B: Tensor) -> Tensor: ...
def bsmm(X: Tensor, W_bsr: Tensor) -> Tensor: ...
def segment_reduce(x: Tensor, seg_ids: Tensor, *, op: Literal["sum","max","mean"]="sum") -> Tensor: ...

# RNG
class PhiloxSeed(TypedDict, total=False):
    key_hi: int
    key_lo: int
    counter: int

def rng_uniform(shape: Tuple[int,...], *, dtype: DType="fp32", seed: PhiloxSeed=..., 
                lo: float=0.0, hi: float=1.0) -> Tensor: ...
def rng_normal(shape: Tuple[int,...], *, dtype: DType="fp32", seed: PhiloxSeed=..., 
               mean: float=0.0, std: float=1.0) -> Tensor: ...

# Collectives (surface for IDEs; backend chosen by runtime)
def all_reduce(x: Tensor, *, axis: str|int="dp", deterministic: Optional[Determinism]=...) -> Tensor: ...
def reduce_scatter(x: Tensor, *, axis: str|int="dp", deterministic: Optional[Determinism]=...) -> Tensor: ...
def all_gather(x: Tensor, *, axis: str|int="dp", deterministic: Optional[Determinism]=...) -> Tensor: ...
```

---

## C++ Header (Example Surface)

> Minimal header preview. Full API is in the runtime bindings.

```cpp
#pragma once
#include <cstdint>
#include <cstddef>
#include <stdexcept>
namespace tessera {
enum class DType:int32_t{FP32,BF16,FP16,FP8_E4M3,FP8_E5M2,INT8,INT32};
enum class Layout:int32_t{ROW_MAJOR,COL_MAJOR,NHWC,NCHW,TILE,BSR};
struct Tensor{void*data;DType dtype;Layout layout;int64_t*shape;int32_t rank;void*meta;};
struct Epilogue{bool add_bias=false;Tensor bias{};enum Act{NONE,RELU,SILU,GELU} activation=NONE;bool add_residual=false;Tensor residual{};};
struct Determinism{bool deterministic=false;};
struct FlashParams{bool causal=false;int block_q=128,block_k=128,block_d=128;};
void matmul(const Tensor&,const Tensor&,Tensor&,const Epilogue* =nullptr) noexcept(false);
void conv2d(const Tensor&,const Tensor&,Tensor&,int,int,int,int,const Epilogue* =nullptr) noexcept(false);
void layernorm(const Tensor&,Tensor&,float) noexcept(false);
void qkv_projection(const Tensor&,const Tensor&,Tensor&,Tensor&,Tensor&) noexcept(false);
void flash_attention(const Tensor&,const Tensor&,const Tensor&,Tensor&,const FlashParams& ={}, const Determinism* =nullptr) noexcept(false);
} // namespace tessera
```
