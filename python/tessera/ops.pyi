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


#: Tessera ops are polymorphic across array carriers — numpy arrays,
#: ``DistributedArray`` instances, and the future native tensor
#: handle.  The stub aliases ``Tensor`` to ``Any`` so callers in
#: ``nn/layers.py`` and elsewhere that work in numpy throughout the
#: reference path type-check cleanly.  A future tightening pass can
#: replace this with a ``Protocol`` once every carrier exposes the
#: same minimal interface.
Tensor = Any


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
def layer_norm(
    x: Tensor,
    gamma: Optional[Tensor] = ...,
    beta: Optional[Tensor] = ...,
    *,
    eps: float = ...,
) -> Tensor: ...
def rmsnorm(x: Tensor, gamma: Optional[Tensor] = ..., *, eps: float = ...) -> Tensor: ...
def softmax(x: Tensor, *, axis: int = ...) -> Tensor: ...
def sigmoid(x: Tensor) -> Tensor: ...
def gelu(x: Tensor) -> Tensor: ...
def softcap(x: Tensor, *, cap: float) -> Tensor: ...
def score_combine(base: Tensor, delta: Tensor, *, gamma: float = ...) -> Tensor: ...
def relu(x: Tensor) -> Tensor: ...
def sin(x: Tensor) -> Tensor: ...
def silu(x: Tensor) -> Tensor: ...
def adam(
    param: Tensor,
    grad: Tensor,
    moment1: Tensor,
    moment2: Tensor,
    *,
    lr: float = ...,
    beta1: float = ...,
    beta2: float = ...,
    eps: float = ...,
    step: int = ...,
    compute_dtype: DType = ...,
    state_dtype: DType = ...,
    master_dtype: Optional[DType] = ...,
    cast_updates_to_param_dtype: bool = ...,
) -> Tuple[Tensor, Tensor, Tensor]: ...
def adamw(params: Any, grads: Any, state: Optional[dict[str, Any]] = ..., **kwargs: Any) -> Tuple[Any, dict[str, Any]]: ...
def momentum(params: Any, grads: Any, state: Optional[dict[str, Any]] = ..., **kwargs: Any) -> Tuple[Any, dict[str, Any]]: ...
def adafactor(params: Any, grads: Any, state: Optional[dict[str, Any]] = ..., **kwargs: Any) -> Tuple[Any, dict[str, Any]]: ...
def lion(params: Any, grads: Any, state: Optional[dict[str, Any]] = ..., **kwargs: Any) -> Tuple[Any, dict[str, Any]]: ...
def dropout(x: Tensor, p: float = ..., *, rng: Any = ..., training: bool = ..., seed: Optional[int] = ...) -> Tensor: ...
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
    seed: Optional[int] = ...,
    params: Optional[FlashParams] = ...,
    deterministic: Optional[Determinism] = ...,
    attn_bias: Optional[Tensor] = ...,
) -> Tensor: ...
def varlen_sdpa(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    *,
    cu_seqlens_q: Any,
    cu_seqlens_k: Any,
    causal: bool = ...,
    scale: Optional[float] = ...,
) -> Tensor: ...
def rope(x: Tensor, theta: Tensor, *, axes: Literal["q", "k", "qk"] = ...) -> Tensor: ...
def ntk_rope(x: Tensor, theta: Tensor, *, scale: float = ...) -> Tensor: ...
def rope_split(x: Tensor, *, rope_dim: int) -> Tuple[Tensor, Tensor]: ...
def rope_merge(rope_part: Tensor, no_rope_part: Tensor) -> Tensor: ...
def alibi(num_heads: int, seq_len: int, slopes: Optional[Tensor] = ...) -> Tensor: ...
def multi_head_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    *,
    num_heads: int,
    scale: Optional[float] = ...,
    causal: bool = ...,
    dropout_p: float = ...,
    seed: Optional[int] = ...,
) -> Tensor: ...
def gqa_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    *,
    num_query_heads: int,
    num_kv_heads: int,
    scale: Optional[float] = ...,
    causal: bool = ...,
) -> Tensor: ...
def mqa_attention(Q: Tensor, K: Tensor, V: Tensor, **kwargs: Any) -> Tensor: ...
def mla_decode(Q: Tensor, K_latent: Tensor, V_latent: Tensor, W_k: Optional[Tensor] = ..., W_v: Optional[Tensor] = ..., **kwargs: Any) -> Tensor: ...
def mla_decode_fused(
    x: Tensor,
    w_dkv: Tensor,
    w_uk: Tensor,
    w_uv: Tensor,
    q: Tensor,
    *,
    scale: Optional[float] = ...,
    causal: bool = ...,
) -> Tensor: ...
def latent_kv_compress(x: Tensor, w_dkv: Tensor) -> Tensor: ...
def latent_kv_expand_k(c: Tensor, w_uk: Tensor) -> Tensor: ...
def latent_kv_expand_v(c: Tensor, w_uv: Tensor) -> Tensor: ...
def attn_sliding_window(Q: Tensor, K: Tensor, V: Tensor, *, window_size: int, causal: bool = ...) -> Tensor: ...
def attn_compressed_blocks(Q: Tensor, K_c: Tensor, V_c: Tensor) -> Tensor: ...
def attn_top_k_blocks(Q: Tensor, K: Tensor, V: Tensor, *, scores: Tensor, top_k: int, block_size: int, causal: bool = ...) -> Tensor: ...
def deepseek_sparse_attention(Q: Tensor, K: Tensor, V: Tensor, gate_logits: Optional[Tensor] = ..., *, window_size: int, block_size: int, top_k: int, causal: bool = ...) -> Tensor: ...
def msa_index_scores(Q: Tensor, K: Tensor, *, block_size: int, scale: Optional[float] = ...) -> Tensor: ...
def msa_select_blocks(scores: Tensor, *, top_k: int, block_size: int, force_local_block: bool = ..., causal: bool = ...) -> Tensor: ...
def msa_sparse_attention(Q: Tensor, K: Tensor, V: Tensor, *, block_size: int, top_k: int, force_local_block: bool = ..., causal: bool = ..., scale: Optional[float] = ..., return_debug: bool = ...) -> Union[Tensor, Tuple[Tensor, dict]]: ...
def lightning_attention(Q: Tensor, K: Tensor, V: Tensor, *, state: Optional[Tensor] = ..., chunk_size: Optional[int] = ..., decay: Optional[Tensor] = ..., causal: bool = ..., return_state: bool = ..., state_dtype: DType = ...) -> Union[Tensor, Tuple[Tensor, Tensor]]: ...
def gated_attention(Q: Tensor, K: Tensor, V: Tensor, gate: Tensor, *, scale: Optional[float] = ..., causal: bool = ..., gate_activation: Literal["sigmoid", "identity", "none"] = ...) -> Tensor: ...
def gated_deltanet(Q: Tensor, K: Tensor, V: Tensor, gate: Optional[Tensor] = ..., beta: Optional[Tensor] = ..., decay: Optional[Tensor] = ..., *, state: Optional[Tensor] = ..., causal: bool = ..., return_state: bool = ..., state_dtype: DType = ...) -> Union[Tensor, Tuple[Tensor, Tensor]]: ...
def kimi_delta_attention(Q: Tensor, K: Tensor, V: Tensor, gate: Optional[Tensor] = ..., beta: Optional[Tensor] = ..., decay: Optional[Tensor] = ..., *, state: Optional[Tensor] = ..., causal: bool = ..., return_state: bool = ..., state_dtype: DType = ...) -> Union[Tensor, Tuple[Tensor, Tensor]]: ...
def modified_delta_attention(Q: Tensor, K: Tensor, V: Tensor, gate: Optional[Tensor] = ..., beta: Optional[Tensor] = ..., decay: Optional[Tensor] = ..., *, state: Optional[Tensor] = ..., causal: bool = ..., return_state: bool = ..., state_dtype: DType = ...) -> Union[Tensor, Tuple[Tensor, Tensor]]: ...
def hybrid_attention(Q: Tensor, K: Tensor, V: Tensor, *, pattern: str = ..., layer_index: int = ..., gate: Optional[Tensor] = ..., beta: Optional[Tensor] = ..., decay: Optional[Tensor] = ..., state: Optional[Tensor] = ..., w_dkv: Optional[Tensor] = ..., w_uk: Optional[Tensor] = ..., w_uv: Optional[Tensor] = ..., q_mla: Optional[Tensor] = ..., causal: bool = ..., return_state: bool = ..., state_dtype: DType = ...) -> Union[Tensor, Tuple[Tensor, Tensor]]: ...
def quantize_fp8(x: Tensor, *, format: Literal["e4m3", "e5m2"] = ..., scale: Optional[float] = ...) -> Tuple[Tensor, float]: ...
def dequantize_fp8(x_q: Tensor, scale: float, *, format: Literal["e4m3", "e5m2"] = ...) -> Tensor: ...
def quantize_fp4(x: Tensor, *, format: Literal["e2m1"] = ..., scale: Optional[float] = ...) -> Tuple[Tensor, float]: ...
def dequantize_fp4(x_q: Tensor, scale: float, *, format: Literal["e2m1"] = ...) -> Tensor: ...
def fake_quantize(x: Tensor, num_bits: int = ..., scale: Optional[float] = ..., zero_point: int = ..., *, symmetric: bool = ...) -> Tensor: ...


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


# Fallback for the S-series / RL / spectral / recurrent / GA / EBM ops
# that ship at runtime but aren't yet declared in this stub.  A
# focused stub-completeness sprint can replace this catch-all with
# explicit signatures op by op.  Until then ``__getattr__`` returning
# ``Any`` keeps mypy from emitting false-positive "Module has no
# attribute" errors against ops that demonstrably exist at runtime.
def __getattr__(name: str) -> Any: ...
