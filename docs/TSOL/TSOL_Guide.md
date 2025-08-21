
# Tessera Standard Operator Library (TSOL)

This document catalogs core Tessera operators and includes **Python type stubs** for IDEs.

---

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
