from typing import Any, Iterable, Literal, Optional, Sequence, Tuple, TypedDict, Union

DType = Literal[
    "fp32",
    "tf32",
    "bf16",
    "fp16",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp6_e2m3",
    "fp6_e3m2",
    "fp4_e2m1",
    "nvfp4",
    "int8",
    "int16",
    "int32",
    "int64",
    "bool",
]
Layout = Literal[
    "row_major",
    "col_major",
    "nhwc",
    "nchw",
    "tile",
    "bsr",
    "nld",
    "blh",
]
Activation = Literal["linear", "none", "relu", "silu", "gelu"]
ReduceOp = Literal["sum", "max", "mean", "min", "prod"]


class Tensor: ...


class NumericPolicy(TypedDict, total=False):
    storage_dtype: DType
    accumulator_dtype: DType
    output_dtype: DType
    rounding: Literal["nearest_even", "stochastic", "toward_zero"]
    scale: Union[float, str]
    quantization_axis: int
    deterministic: bool


class Epilogue(TypedDict, total=False):
    add_bias: bool
    bias: Tensor
    activation: Activation
    add_residual: bool
    residual: Tensor
    dropout_p: float
    cast_dtype: DType
    numeric_policy: NumericPolicy


class Determinism(TypedDict, total=False):
    deterministic: bool
    reduce_tree: Literal["binary", "flat", "ring"]
    send_order: Literal["lexicographic", "rank", "auto"]
    schedule: Literal["reuse", "stable_search", "auto"]


class Transport(TypedDict, total=False):
    type: Literal["auto", "nccl", "rccl", "nvshmem", "deepep"]
    multi_qp: bool
    pack_dtype: DType


class FlashParams(TypedDict, total=False):
    causal: bool
    block_q: int
    block_k: int
    block_d: int
    dropout_p: float
    deterministic: bool


class BsrMeta(TypedDict, total=False):
    block_m: int
    block_n: int
    mask: Union[memoryview, bytes]
    mask_shape: Tuple[int, int]


class PhiloxSeed(TypedDict, total=False):
    key_hi: int
    key_lo: int
    counter: int
    subsequence: int


class ScheduleArtifact(TypedDict, total=False):
    op: str
    shape: Tuple[int, ...]
    layout: Layout
    arch: str
    numeric_policy: NumericPolicy
    movement_plan: dict[str, Any]
    tile_knobs: dict[str, Any]
    hash: str


# Linear algebra
def gemm(A: Tensor, B: Tensor, *, epilogue: Optional[Epilogue] = ...) -> Tensor: ...
def matmul(A: Tensor, B: Tensor, *, epilogue: Optional[Epilogue] = ...) -> Tensor: ...
def batched_gemm(A: Tensor, B: Tensor) -> Tensor: ...
def einsum(spec: str, *tensors: Tensor) -> Tensor: ...
def factorized_matmul(A: Tensor, B: Tensor, *, rank: int) -> Tensor: ...
def tri_solve(A: Tensor, b: Tensor, *, lower: bool = ...) -> Tensor: ...
def cholesky(A: Tensor) -> Tensor: ...
def qr(A: Tensor) -> Tuple[Tensor, Tensor]: ...
def svd(A: Tensor) -> Tuple[Tensor, Tensor, Tensor]: ...


# Neural-network primitives
def conv2d(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = ...,
    *,
    stride: Union[int, Tuple[int, int]] = ...,
    padding: Union[int, Tuple[int, int]] = ...,
    layout: Literal["nhwc", "nchw"] = ...,
    epilogue: Optional[Epilogue] = ...,
) -> Tensor: ...
def conv3d(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = ...,
    *,
    stride: Union[int, Tuple[int, int, int]] = ...,
    padding: Union[int, Tuple[int, int, int]] = ...,
    layout: Literal["ndhwc", "ncdhw"] = ...,
    epilogue: Optional[Epilogue] = ...,
) -> Tensor: ...
def layer_norm(x: Tensor, *, eps: float = ...) -> Tensor: ...
def rmsnorm(x: Tensor, *, eps: float = ...) -> Tensor: ...
def softmax(x: Tensor, *, axis: int = ...) -> Tensor: ...
def gelu(x: Tensor) -> Tensor: ...
def relu(x: Tensor) -> Tensor: ...
def silu(x: Tensor) -> Tensor: ...
def dropout(x: Tensor, p: float = ..., *, rng: Any = ..., training: bool = ...) -> Tensor: ...
def fused_epilogue(
    x: Tensor,
    bias: Optional[Tensor] = ...,
    *,
    activation: Activation = ...,
    residual: Optional[Tensor] = ...,
    dropout_p: float = ...,
    cast_dtype: Optional[DType] = ...,
) -> Tensor: ...
def qkv_projection(x: Tensor, W_qkv: Tensor) -> Tuple[Tensor, Tensor, Tensor]: ...
def flash_attn(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    *,
    scale: Optional[float] = ...,
    causal: bool = ...,
    cache: Optional[Any] = ...,
    dropout_p: float = ...,
    params: Optional[FlashParams] = ...,
    deterministic: Optional[Determinism] = ...,
) -> Tensor: ...
def rope(x: Tensor, theta: Tensor, *, axes: Literal["q", "k", "qk"] = ...) -> Tensor: ...


# MoE
def moe(
    x: Tensor,
    experts: Iterable[Any],
    *,
    router: Literal["topk", "hash"] = ...,
    k: int = ...,
    transport: Optional[Transport] = ...,
    deterministic: Optional[Determinism] = ...,
) -> Tensor: ...
def moe_dispatch(x: Tensor, route: Tensor, *, transport: Optional[Transport] = ...) -> Tensor: ...
def moe_combine(
    partials: Tensor,
    inverse_route: Tensor,
    *,
    reduce: Literal["sum", "mean"] = ...,
) -> Tensor: ...


# Spectral
def fft(x: Tensor, *, axes: Optional[Sequence[int]] = ...) -> Tensor: ...
def ifft(xf: Tensor, *, axes: Optional[Sequence[int]] = ...) -> Tensor: ...
def rfft(x: Tensor, *, axes: Optional[Sequence[int]] = ...) -> Tensor: ...
def irfft(xf: Tensor, *, axes: Optional[Sequence[int]] = ...) -> Tensor: ...
def stft(x: Tensor, win: Tensor, *, hop: int) -> Tensor: ...
def istft(xf: Tensor, win: Tensor, *, hop: int) -> Tensor: ...
def spectral_filter(Xf: Tensor, Hf: Tensor) -> Tensor: ...


# Sparse, segment, and graph
def spmm_coo(A_coo: Tensor, B: Tensor) -> Tensor: ...
def spmm_csr(A_csr: Tensor, B: Tensor) -> Tensor: ...
def sddmm(A: Tensor, B: Tensor, mask: Tensor) -> Tensor: ...
def bsmm(X: Tensor, W_bsr: Tensor, *, meta: Optional[BsrMeta] = ...) -> Tensor: ...
def segment_reduce(x: Tensor, seg_ids: Tensor, *, op: ReduceOp = ...) -> Tensor: ...


# RNG
def rng_uniform(
    shape: Tuple[int, ...],
    *,
    dtype: DType = ...,
    seed: Optional[PhiloxSeed] = ...,
    lo: float = ...,
    hi: float = ...,
) -> Tensor: ...
def rng_normal(
    shape: Tuple[int, ...],
    *,
    dtype: DType = ...,
    seed: Optional[PhiloxSeed] = ...,
    mean: float = ...,
    std: float = ...,
) -> Tensor: ...


# Collectives
def all_reduce(
    x: Tensor,
    *,
    axis: Union[str, int] = ...,
    op: ReduceOp = ...,
    deterministic: Optional[Determinism] = ...,
) -> Tensor: ...
def reduce_scatter(
    x: Tensor,
    *,
    axis: Union[str, int] = ...,
    op: ReduceOp = ...,
    deterministic: Optional[Determinism] = ...,
) -> Tensor: ...
def all_gather(
    x: Tensor,
    *,
    axis: Union[str, int] = ...,
    deterministic: Optional[Determinism] = ...,
) -> Tensor: ...
def all_to_all(
    x: Tensor,
    *,
    axis: Union[str, int],
    deterministic: Optional[Determinism] = ...,
) -> Tensor: ...


# Layout and packing
def transpose(x: Tensor, perm: Optional[Sequence[int]] = ...) -> Tensor: ...
def cast(x: Tensor, dtype: DType) -> Tensor: ...
def rearrange(x: Tensor, layout: Layout) -> Tensor: ...
def pack(x: Tensor, layout: Layout) -> Tensor: ...
def unpack(x: Tensor) -> Tensor: ...
def tile_view(x: Tensor, BM: int, BN: int, BK: Optional[int] = ...) -> Tensor: ...
