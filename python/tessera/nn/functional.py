"""Functional `tessera.nn` surface — stateless layer wrappers over `tessera.ops`.

Use these when you don't want to allocate `Parameter` storage; pass weights in
explicitly as numpy arrays / `DistributedArray`s. Stateful versions live in
`tessera.nn` (`Linear`, `RMSNorm`, etc.) and just compose these calls.

Both `tessera.nn.linear` and `tessera.nn.functional.linear` resolve here.
"""

from __future__ import annotations

import numpy as np

from .. import ops


def _asarray(x):
    if hasattr(x, "_data"):
        x = x._data
    if hasattr(x, "_data"):
        x = x._data
    return np.asarray(x)


def _pair(v):
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(f"expected scalar or length-2 tuple; got {v}")
        return int(v[0]), int(v[1])
    return int(v), int(v)


def linear(x, W, bias=None):
    """y = x @ W (+ bias). Bias is optional and broadcast over leading dims.

    Composed through `ops.gemm` (+ `ops.add` when bias is non-None) so an
    active autodiff tape sees every primitive.
    """
    y = ops.gemm(x, W)
    if bias is not None:
        if hasattr(bias, "_data"):
            bias = bias._data
        y = ops.add(y, bias)
    return y


def linear_general(x, W, bias=None, axis=-1):
    """Axis-flexible linear projection.

    Contracts `x` over `axis` with the leading axes of `W`; trailing axes of
    `W` become output features. This covers Flax-style LinearGeneral/Einsum
    layers without introducing an external framework dependency.
    """
    x_arr = _asarray(x)
    w_arr = _asarray(W)
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    axes = tuple(ax if ax >= 0 else x_arr.ndim + ax for ax in axes)
    if len(set(axes)) != len(axes):
        raise ValueError(f"linear_general axes must be unique; got {axis}")
    if len(axes) > w_arr.ndim:
        raise ValueError("weight rank must be at least the number of contracted axes")
    for x_ax, w_dim in zip(axes, w_arr.shape[: len(axes)]):
        if x_arr.shape[x_ax] != w_dim:
            raise ValueError(
                f"contracted dim mismatch: x axis {x_ax} has {x_arr.shape[x_ax]}, "
                f"weight dim has {w_dim}"
            )
    # Route the contraction through ops.einsum (not np.tensordot) so an active
    # autodiff tape sees the projection — functional.* must decompose through
    # ops.* (mirrors linear()). einsum has a registered VJP/JVP.
    pool = [chr(c) for c in range(ord("a"), ord("z") + 1)]
    n_out_feat = w_arr.ndim - len(axes)
    if x_arr.ndim + n_out_feat > len(pool):
        # Degenerate very-high-rank case: fall back to the numpy reference
        # (autodiff unavailable, correctness preserved).
        y = np.tensordot(x_arr, w_arr, axes=(axes, tuple(range(len(axes)))))
    else:
        x_letters = pool[: x_arr.ndim]
        contract = set(axes)
        w_trailing = pool[x_arr.ndim : x_arr.ndim + n_out_feat]
        w_letters = [x_letters[ax] for ax in axes] + w_trailing
        out_letters = [l for i, l in enumerate(x_letters) if i not in contract] + w_trailing
        spec = "".join(x_letters) + "," + "".join(w_letters) + "->" + "".join(out_letters)
        y = ops.einsum(spec, x_arr, w_arr)
    if bias is not None:
        y = ops.add(y, _asarray(bias))
    return y


def einsum(spec: str, *tensors):
    """Numpy-backed Einsum reference for axis-flexible model layers."""
    return ops.einsum(spec, *tensors)


def rms_norm(x, weight=None, eps: float = 1e-5):
    """RMSNorm: x / sqrt(mean(x*x) + eps), optionally scaled by `weight`.

    Composed through `ops.rmsnorm` and `ops.mul` so the autodiff tape captures
    both the normalization and the affine.
    """
    y = ops.rmsnorm(x, eps=eps)
    if weight is not None:
        if hasattr(weight, "_data"):
            weight = weight._data
        y = ops.mul(y, weight)
    return y


def group_norm(x, num_groups: int, weight=None, bias=None, eps: float = 1e-5):
    """GroupNorm over N,C,* tensors."""
    x_arr = _asarray(x).astype(np.float32, copy=False)
    if x_arr.ndim < 2:
        raise ValueError(f"group_norm expects rank >= 2; got {x_arr.shape}")
    n, c = x_arr.shape[:2]
    if c % num_groups != 0:
        raise ValueError(f"channels {c} must be divisible by num_groups {num_groups}")
    grouped = x_arr.reshape(n, num_groups, c // num_groups, *x_arr.shape[2:])
    reduce_axes = tuple(range(2, grouped.ndim))
    mean = grouped.mean(axis=reduce_axes, keepdims=True)
    var = grouped.var(axis=reduce_axes, keepdims=True)
    y = ((grouped - mean) / np.sqrt(var + eps)).reshape(x_arr.shape)
    if weight is not None:
        y = y * _asarray(weight).reshape(1, c, *([1] * (x_arr.ndim - 2)))
    if bias is not None:
        y = y + _asarray(bias).reshape(1, c, *([1] * (x_arr.ndim - 2)))
    return y


def instance_norm(x, weight=None, bias=None, eps: float = 1e-5):
    """InstanceNorm over N,C,* tensors."""
    x_arr = _asarray(x).astype(np.float32, copy=False)
    if x_arr.ndim < 3:
        raise ValueError(f"instance_norm expects rank >= 3; got {x_arr.shape}")
    reduce_axes = tuple(range(2, x_arr.ndim))
    mean = x_arr.mean(axis=reduce_axes, keepdims=True)
    var = x_arr.var(axis=reduce_axes, keepdims=True)
    y = (x_arr - mean) / np.sqrt(var + eps)
    c = x_arr.shape[1]
    if weight is not None:
        y = y * _asarray(weight).reshape(1, c, *([1] * (x_arr.ndim - 2)))
    if bias is not None:
        y = y + _asarray(bias).reshape(1, c, *([1] * (x_arr.ndim - 2)))
    return y


def weight_norm(weight, axis: int = 0, eps: float = 1e-12):
    """Normalize a weight tensor along all axes except `axis`."""
    w = _asarray(weight).astype(np.float32, copy=False)
    axis = axis if axis >= 0 else w.ndim + axis
    reduce_axes = tuple(i for i in range(w.ndim) if i != axis)
    norm = np.sqrt(np.sum(w * w, axis=reduce_axes, keepdims=True) + eps)
    return w / norm


def spectral_norm(weight, eps: float = 1e-12):
    """Reference spectral normalization using exact SVD of the flattened matrix."""
    w = _asarray(weight).astype(np.float32, copy=False)
    mat = w.reshape(w.shape[0], -1)
    sigma = np.linalg.svd(mat, compute_uv=False)[0]
    return w / max(float(sigma), eps)


def conv1d(x, weight, bias=None, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1):
    """NCL grouped Conv1d reference.

    Input shape `[N, C_in, L]`; weight shape `[C_out, C_in/groups, K]`.
    """
    x_arr = _asarray(x).astype(np.float32, copy=False)
    w_arr = _asarray(weight).astype(np.float32, copy=False)
    if x_arr.ndim != 3 or w_arr.ndim != 3:
        raise ValueError("conv1d expects x [N,C,L] and weight [O,I,K]")
    n, c_in, length = x_arr.shape
    c_out, c_per_group, kernel = w_arr.shape
    if groups <= 0 or c_in % groups != 0 or c_out % groups != 0:
        raise ValueError("groups must divide input and output channels")
    if c_per_group != c_in // groups:
        raise ValueError("weight input channels must equal C_in/groups")
    padded = np.pad(x_arr, ((0, 0), (0, 0), (padding, padding)))
    out_len = (length + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
    if out_len <= 0:
        raise ValueError("conv1d output length must be positive")
    out = np.zeros((n, c_out, out_len), dtype=np.float32)
    out_per_group = c_out // groups
    in_per_group = c_in // groups
    for b in range(n):
        for g in range(groups):
            in_base = g * in_per_group
            out_base = g * out_per_group
            for oc in range(out_per_group):
                for pos in range(out_len):
                    acc = 0.0
                    start = pos * stride
                    for ic in range(in_per_group):
                        for k in range(kernel):
                            acc += padded[b, in_base + ic, start + k * dilation] * w_arr[out_base + oc, ic, k]
                    out[b, out_base + oc, pos] = acc
    if bias is not None:
        out += _asarray(bias).reshape(1, c_out, 1)
    return out


def conv_transpose(x, weight, bias=None, stride: int = 1, padding: int = 0, output_padding: int = 0, dilation: int = 1, groups: int = 1):
    """NCL grouped ConvTranspose1d reference.

    Input shape `[N, C_in, L]`; weight shape `[C_in, C_out/groups, K]`.
    """
    x_arr = _asarray(x).astype(np.float32, copy=False)
    w_arr = _asarray(weight).astype(np.float32, copy=False)
    if x_arr.ndim != 3 or w_arr.ndim != 3:
        raise ValueError("conv_transpose expects x [N,C,L] and weight [I,O,K]")
    n, c_in, length = x_arr.shape
    w_in, c_out_per_group, kernel = w_arr.shape
    if w_in != c_in or groups <= 0 or c_in % groups != 0:
        raise ValueError("weight input channels and groups are inconsistent")
    c_out = c_out_per_group * groups
    out_len = (length - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
    out = np.zeros((n, c_out, out_len), dtype=np.float32)
    in_per_group = c_in // groups
    for b in range(n):
        for g in range(groups):
            in_base = g * in_per_group
            out_base = g * c_out_per_group
            for ic in range(in_per_group):
                for pos in range(length):
                    for k in range(kernel):
                        out_pos = pos * stride - padding + k * dilation
                        if 0 <= out_pos < out_len:
                            out[b, out_base:out_base + c_out_per_group, out_pos] += (
                                x_arr[b, in_base + ic, pos] * w_arr[in_base + ic, :, k]
                            )
    if bias is not None:
        out += _asarray(bias).reshape(1, c_out, 1)
    return out


def _pool2d(x, kernel_size, stride=None, padding=0, reducer=np.max):
    x_arr = _asarray(x).astype(np.float32, copy=False)
    if x_arr.ndim != 4:
        raise ValueError(f"pool expects NCHW rank-4 input; got {x_arr.shape}")
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride if stride is not None else kernel_size)
    ph, pw = _pair(padding)
    pad_value = -np.inf if reducer is np.max else (np.inf if reducer is np.min else 0.0)
    padded = np.pad(x_arr, ((0, 0), (0, 0), (ph, ph), (pw, pw)), constant_values=pad_value)
    out_h = (x_arr.shape[2] + 2 * ph - kh) // sh + 1
    out_w = (x_arr.shape[3] + 2 * pw - kw) // sw + 1
    out = np.zeros((x_arr.shape[0], x_arr.shape[1], out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            window = padded[:, :, i * sh:i * sh + kh, j * sw:j * sw + kw]
            out[:, :, i, j] = reducer(window, axis=(2, 3))
    return out


def max_pool(x, kernel_size, stride=None, padding=0):
    return _pool2d(x, kernel_size, stride=stride, padding=padding, reducer=np.max)


def avg_pool(x, kernel_size, stride=None, padding=0):
    return _pool2d(x, kernel_size, stride=stride, padding=padding, reducer=np.mean)


def min_pool(x, kernel_size, stride=None, padding=0):
    return _pool2d(x, kernel_size, stride=stride, padding=padding, reducer=np.min)


def adaptive_pool(x, output_size, reducer=np.mean):
    """Adaptive 2D pooling over NCHW input."""
    x_arr = _asarray(x).astype(np.float32, copy=False)
    if x_arr.ndim != 4:
        raise ValueError(f"adaptive_pool expects NCHW rank-4 input; got {x_arr.shape}")
    out_h, out_w = _pair(output_size)
    out = np.zeros((x_arr.shape[0], x_arr.shape[1], out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        h0 = int(np.floor(i * x_arr.shape[2] / out_h))
        h1 = int(np.ceil((i + 1) * x_arr.shape[2] / out_h))
        for j in range(out_w):
            w0 = int(np.floor(j * x_arr.shape[3] / out_w))
            w1 = int(np.ceil((j + 1) * x_arr.shape[3] / out_w))
            out[:, :, i, j] = reducer(x_arr[:, :, h0:h1, w0:w1], axis=(2, 3))
    return out


def simple_rnn_cell(x, h, W_ih, W_hh, bias=None, activation: str = "tanh"):
    pre = _asarray(x) @ _asarray(W_ih) + _asarray(h) @ _asarray(W_hh)
    if bias is not None:
        pre = pre + _asarray(bias)
    if activation == "relu":
        return np.maximum(pre, 0.0)
    if activation == "tanh":
        return np.tanh(pre)
    raise ValueError(f"unsupported activation {activation!r}")


def gru_cell(x, h, W_ih, W_hh, b_ih=None, b_hh=None):
    """GRU cell with gate order z, r, n."""
    x_arr, h_arr = _asarray(x), _asarray(h)
    gates_x = x_arr @ _asarray(W_ih)
    gates_h = h_arr @ _asarray(W_hh)
    if b_ih is not None:
        gates_x = gates_x + _asarray(b_ih)
    if b_hh is not None:
        gates_h = gates_h + _asarray(b_hh)
    x_z, x_r, x_n = np.split(gates_x, 3, axis=-1)
    h_z, h_r, h_n = np.split(gates_h, 3, axis=-1)
    z = 1.0 / (1.0 + np.exp(-(x_z + h_z)))
    r = 1.0 / (1.0 + np.exp(-(x_r + h_r)))
    n = np.tanh(x_n + r * h_n)
    return (1.0 - z) * n + z * h_arr


def bidirectional_scan(fn, init_fwd, init_bwd, xs):
    xs_arr = _asarray(xs)
    fwd_states = []
    carry = init_fwd
    for x_t in xs_arr:
        carry = fn(carry, x_t)
        fwd_states.append(carry)
    bwd_states = []
    carry = init_bwd
    for x_t in xs_arr[::-1]:
        carry = fn(carry, x_t)
        bwd_states.append(carry)
    return np.stack(fwd_states), np.stack(bwd_states[::-1])


def lora_linear(x, weight, lora_a, lora_b, bias=None, alpha: float = 1.0):
    rank = _asarray(lora_a).shape[-1]
    y = _asarray(x) @ _asarray(weight)
    y = y + ((_asarray(x) @ _asarray(lora_a)) @ _asarray(lora_b)) * (float(alpha) / max(1, rank))
    if bias is not None:
        y = y + _asarray(bias)
    return y


def alibi(num_heads: int, seq_len: int, slopes=None):
    if slopes is None:
        slopes = 2.0 ** (-8.0 * np.arange(1, num_heads + 1, dtype=np.float32) / num_heads)
    slopes = np.asarray(slopes, dtype=np.float32).reshape(num_heads, 1, 1)
    positions = np.arange(seq_len, dtype=np.float32)
    distance = positions.reshape(1, seq_len) - positions.reshape(seq_len, 1)
    return slopes * distance.reshape(1, seq_len, seq_len)


def ntk_rope(x, theta, scale: float = 1.0):
    return ops.rope(_asarray(x), _asarray(theta) / float(scale))


def _repeat_kv(t, repeats: int):
    return np.repeat(_asarray(t), repeats, axis=1)


def gqa_attention(Q, K, V, num_query_heads: int, num_kv_heads: int, **kwargs):
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    return ops.flash_attn(_asarray(Q), _repeat_kv(K, num_query_heads // num_kv_heads), _repeat_kv(V, num_query_heads // num_kv_heads), **kwargs)


def mqa_attention(Q, K, V, **kwargs):
    return gqa_attention(Q, K, V, num_query_heads=_asarray(Q).shape[1], num_kv_heads=1, **kwargs)


def minimax_sparse_attention(Q, K, V, *, block_size: int, top_k: int,
                             force_local_block: bool = True, causal: bool = True,
                             scale=None, return_debug: bool = False):
    """MiniMax Sparse Attention (arXiv:2606.13392).

    Grouped-query block-sparse attention: a lightweight Index Branch scores KV
    blocks and selects ``top_k`` of them per GQA group, then the Main Branch
    runs exact attention over only the selected blocks. ``Q`` ``(B, Hq, Sq, D)``,
    ``K``/``V`` ``(B, Hkv, Sk, D)`` with ``Hq % Hkv == 0``. Delegates to the
    reference :func:`tessera.ops.msa_sparse_attention`; when ``top_k`` equals the
    number of KV blocks this reduces to dense GQA attention.
    """
    return ops.msa_sparse_attention(
        _asarray(Q), _asarray(K), _asarray(V),
        block_size=block_size, top_k=top_k,
        force_local_block=force_local_block, causal=causal,
        scale=scale, return_debug=return_debug,
    )


def mla_decode(Q, K_latent, V_latent, W_k=None, W_v=None, **kwargs):
    K = _asarray(K_latent) if W_k is None else linear_general(K_latent, W_k)
    V = _asarray(V_latent) if W_v is None else linear_general(V_latent, W_v)
    return ops.flash_attn(_asarray(Q), K, V, **kwargs)


def swiglu(x, W_gate, W_up, W_down):
    """SwiGLU MLP block: (silu(x @ W_gate) * (x @ W_up)) @ W_down.

    Composed through primitive ops (`gemm` / `silu_mul` / `gemm`) so the
    autodiff tape sees every step AND the Schedule IR fusion recognizer
    matches the `matmul → silu_mul → matmul` chain. Functionally equivalent
    to `ops.swiglu(...)`; either spelling works.
    """
    gate = ops.gemm(x, W_gate)
    up = ops.gemm(x, W_up)
    hidden = ops.silu_mul(gate, up)
    return ops.gemm(hidden, W_down)


def moe_layer(
    x,
    W_router,
    W_gate,
    W_up,
    W_down,
    *,
    top_k: int = 1,
    normalize_weights: bool = True,
    capacity_factor: float = 0.0,
    kind: str = "contiguous",
    alignment=None,
    quant=None,
):
    """Single-device top-k SwiGLU MoE layer: router → permute → expert FFN → combine.

    The full local MoE forward, one step short of distributed MegaMoE (no
    token all-to-all, no comm/compute overlap):

        scores   = x @ W_router                    # (T, E) gate logits
        route    = top-k(softmax(scores))          # per-token expert + weight
        x_perm   = gather tokens into expert order # contiguous grouped layout
        y_perm   = moe_swiglu_block(x_perm, ...)   # GPU expert FFN (grouped GEMM)
        out      = scatter-add weighted y_perm      # combine k expert outputs

    The heavy expert compute flows through ``ops.moe_swiglu_block`` (which routes
    to Apple-GPU ``metal_runtime`` by composing the grouped-GEMM + silu_mul lanes);
    the data-dependent routing / permute / combine are host-side index math — the
    same host-falls-back-for-data-dependent-indexing policy ``argmax`` already uses.

    Shapes
    ------
    x        : (T, K)        input tokens
    W_router : (K, E)        router / gate projection
    W_gate   : (E, K, F)     per-expert SwiGLU gate weight
    W_up     : (E, K, F)     per-expert SwiGLU up weight
    W_down   : (E, F, N)     per-expert down projection
    returns  : (T, N)        combined expert outputs

    Capacity drops (``route_tokens`` overflow → slot weight 0) contribute nothing
    to the combine, so over-capacity tokens are simply not routed.
    """
    from ..distributed.moe import MoEConfig, route_tokens

    xa = _asarray(x)
    Wr = _asarray(W_router)
    T, _K = xa.shape
    E = Wr.shape[1]
    N = _asarray(W_down).shape[2]

    # 1. Router — gate logits on GPU, top-k selection on host (data-dependent).
    scores = np.asarray(ops.gemm(xa, Wr))
    cfg = MoEConfig(
        num_experts=E,
        top_k=top_k,
        normalize_weights=normalize_weights,
        capacity_factor=capacity_factor if capacity_factor > 0 else 1e9,
    )
    route = route_tokens(scores.astype(np.float32), cfg)

    # 2. Expand each token to its top_k slots and sort slots into expert order.
    slot_token = np.repeat(np.arange(T, dtype=np.int64), top_k)
    slot_expert = route.assignment.reshape(-1)
    slot_weight = route.weights.reshape(-1).astype(np.float32)
    keep = slot_expert >= 0                       # drop capacity-overflow slots
    slot_token, slot_expert, slot_weight = (
        slot_token[keep],
        slot_expert[keep],
        slot_weight[keep],
    )
    order = np.argsort(slot_expert, kind="stable")
    perm_token = slot_token[order]
    perm_weight = slot_weight[order]
    group_sizes = np.bincount(slot_expert[order], minlength=E).astype(np.int64)
    x_perm = xa[perm_token]                        # (S, K) tokens in expert order

    # 3. Expert FFN over the contiguous grouped layout (GPU grouped-GEMM lanes).
    y_perm = np.asarray(
        ops.moe_swiglu_block(
            x_perm, W_gate, W_up, W_down, group_sizes,
            kind=kind, alignment=alignment, quant=quant,
        )
    )

    # 4. Combine — weight each slot by its gate, scatter-add back per token.
    out = np.zeros((T, N), dtype=y_perm.dtype)
    np.add.at(out, perm_token, y_perm * perm_weight[:, None])
    return out


def multi_head_attention(
    Q,
    K,
    V,
    num_heads: int,
    scale: float | None = None,
    causal: bool = False,
    dropout_p: float = 0.0,
    seed: int | None = None,
):
    """Multi-head attention via `ops.flash_attn`.

    Inputs are `[B, S, H*D]` tensors. They are reshaped to `[B, num_heads, S, D]`,
    passed through `ops.flash_attn`, then reshaped back to `[B, S, H*D]`.

    For raw `[B, H, S, D]` inputs, call `ops.flash_attn` directly.
    """
    if hasattr(Q, "_data"):
        Q = Q._data
    if hasattr(K, "_data"):
        K = K._data
    if hasattr(V, "_data"):
        V = V._data
    Q, K, V = np.asarray(Q), np.asarray(K), np.asarray(V)
    if Q.ndim != 3 or K.ndim != 3 or V.ndim != 3:
        raise ValueError(
            "multi_head_attention expects [B, S, H*D] inputs; "
            "for raw [B, H, S, D] tensors, call tessera.ops.flash_attn directly"
        )
    B, Sq, hd = Q.shape
    Sk = K.shape[1]
    Sv = V.shape[1]
    if K.shape[0] != B or V.shape[0] != B:
        raise ValueError("Q/K/V must share batch dim")
    if K.shape[2] != hd or V.shape[2] != hd:
        raise ValueError("Q/K/V must share hidden dim")
    if Sk != Sv:
        raise ValueError(f"K and V must share sequence length; got {Sk} vs {Sv}")
    if hd % num_heads != 0:
        raise ValueError(f"hidden dim {hd} not divisible by num_heads {num_heads}")
    D = hd // num_heads

    def split(t):
        S_t = t.shape[1]
        return t.reshape(B, S_t, num_heads, D).transpose(0, 2, 1, 3)

    out = ops.flash_attn(
        split(Q),
        split(K),
        split(V),
        scale=scale,
        causal=causal,
        dropout_p=dropout_p,
        seed=seed,
    )
    return out.transpose(0, 2, 1, 3).reshape(B, Sq, hd)


def _rms_over_last(t, weight=None, eps: float = 1e-6):
    """RMSNorm over the last (head_dim) axis — the QK-norm used by DFlash/Qwen3."""
    t = _asarray(t).astype(np.float32, copy=False)
    y = t / np.sqrt(np.mean(t * t, axis=-1, keepdims=True) + eps)
    if weight is not None:
        y = y * _asarray(weight)
    return y


def mask_token_block(prev_token, block_size: int, mask_token_id: int):
    """Build the DFlash draft input block ``[prev_token, MASK, MASK, ...]``.

    The block-diffusion draft predicts ``block_size - 1`` masked positions in a
    single parallel forward; position 0 carries the real previous token and the
    rest are the ``mask_token_id`` placeholder (see DFlash ``model_mlx`` line 497).
    Returns an int64 array of shape ``(..., block_size)``.
    """
    prev = np.asarray(prev_token, dtype=np.int64)
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    tail = np.full(prev.shape + (block_size - 1,), int(mask_token_id), dtype=np.int64)
    return np.concatenate([prev[..., None], tail], axis=-1)


def block_diffusion_attention(
    x,
    x_ctx,
    *,
    q_proj,
    k_proj,
    v_proj,
    o_proj,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_norm=None,
    k_norm=None,
    cache_keys=None,
    cache_values=None,
    rope_fn=None,
    cache_offset: int = 0,
    sliding_window: int | None = None,
    scale: float | None = None,
    eps: float = 1e-6,
    attention_fn=None,
    return_ctx_kv: bool = False,
):
    """One DFlash block-diffusion attention layer (numpy reference).

    Faithful to DFlash ``DFlashAttention.__call__`` (``model_mlx`` lines 82-116):
    queries come from the draft block hidden ``x`` ``(B, L, D_model)``; keys and
    values are ``concat([context_KV, proposal_KV])`` where the context KV is
    projected from the injected target features ``x_ctx`` ``(B, S, D_model)`` (the
    same tensor fed to every layer) and the proposal KV is projected from ``x``.
    Within-block attention is bidirectional (``mask=None``) for full-attention
    layers; sliding-attention layers fold a causal window into an additive bias
    routed through the ``attn_bias`` substrate of ``ops.flash_attn``.

    QK-norm (RMSNorm over ``head_dim``) is applied to queries / context keys /
    proposal keys, matching Qwen3-style draft models. GQA is handled by repeating
    the ``num_kv_heads`` K/V heads up to ``num_heads`` query heads. ``rope_fn`` —
    when given — is ``rope_fn(tensor_BHSD, offset)``; context keys use
    ``cache_offset`` and queries / proposal keys use ``cache_offset + S`` so block
    positions sit after the context (``model_mlx`` lines 102-104).

    ``cache_keys`` / ``cache_values`` (shape ``(B, S_cached, num_kv_heads,
    head_dim)``) are prior accumulated context KV prepended ahead of this step's
    context — this is where a ``KVCacheHandle`` plugs in across drafting steps.

    Heads are folded into the batch axis so the attention core rides the rank-3
    ``ops.flash_attn`` path (Apple GPU ``metal_runtime`` lane).
    """
    x = _asarray(x)
    x_ctx = _asarray(x_ctx)
    if x.ndim != 3 or x_ctx.ndim != 3:
        raise ValueError("block_diffusion_attention expects rank-3 (B, L, D) x and x_ctx")
    B, L, _ = x.shape
    S = x_ctx.shape[1]
    Hq, Hkv, Dh = int(num_heads), int(num_kv_heads), int(head_dim)
    if Hq % Hkv != 0:
        raise ValueError(f"num_heads {Hq} must be a multiple of num_kv_heads {Hkv}")
    if scale is None:
        scale = Dh ** -0.5

    def proj_heads(t, W, heads):
        # (B, T, D_model) @ (D_model, heads*Dh) -> (B, heads, T, Dh)
        y = linear_general(t, W)
        T = y.shape[1]
        return _asarray(y).reshape(B, T, heads, Dh).transpose(0, 2, 1, 3)

    queries = _rms_over_last(proj_heads(x, q_proj, Hq), q_norm, eps)
    ctx_k = _rms_over_last(proj_heads(x_ctx, k_proj, Hkv), k_norm, eps)
    ctx_v = proj_heads(x_ctx, v_proj, Hkv)
    prop_k = _rms_over_last(proj_heads(x, k_proj, Hkv), k_norm, eps)
    prop_v = proj_heads(x, v_proj, Hkv)

    if rope_fn is not None:
        queries = _asarray(rope_fn(queries, cache_offset + S))
        ctx_k = _asarray(rope_fn(ctx_k, cache_offset))
        prop_k = _asarray(rope_fn(prop_k, cache_offset + S))

    # This step's (roped) context KV, before prepending the prior cache. These
    # are what a stateful loop appends to the per-layer cache for the next step
    # (re-projecting accumulated x_ctx would give the same result, but caching
    # the projected+roped KV is what makes DFlash drafting cheap).
    ctx_k_this = ctx_k
    ctx_v_this = ctx_v

    # Prepend prior cached context KV (accumulated across drafting steps).
    if cache_keys is not None and cache_values is not None:
        ck = _asarray(cache_keys).transpose(0, 2, 1, 3)  # (B, Hkv, Sc, Dh)
        cv = _asarray(cache_values).transpose(0, 2, 1, 3)
        ctx_k = np.concatenate([ck, ctx_k], axis=2)
        ctx_v = np.concatenate([cv, ctx_v], axis=2)
    ctx_len = ctx_k.shape[2]

    # KV injection layout: K/V = [context_KV, proposal_KV] along the seq axis.
    keys = np.concatenate([ctx_k, prop_k], axis=2)    # (B, Hkv, ctx_len+L, Dh)
    values = np.concatenate([ctx_v, prop_v], axis=2)
    Sk = keys.shape[2]

    # GQA: repeat KV heads up to query heads. This is numerically exact; the
    # native non-repeated path (the runtime `flash_attn_gqa` kernel) reads the
    # KV group directly to save bandwidth, but it does not support DFlash's
    # concatenated-context+proposal KV with an additive bias, so the reference
    # (and the rank-3 flash_attn lane) materialize the repeat.
    if Hkv != Hq:
        rep = Hq // Hkv
        keys = np.repeat(keys, rep, axis=1)
        values = np.repeat(values, rep, axis=1)

    # Optional additive bias — sliding-window causal mask folded into attn_bias.
    bias = None
    if sliding_window is not None:
        # Block query i sits at absolute position ctx_len + i; it may attend to
        # keys within the trailing ``sliding_window`` positions (causal).
        qpos = ctx_len + np.arange(L)[:, None]          # (L, 1)
        kpos = np.arange(Sk)[None, :]                    # (1, Sk)
        allow = (kpos <= qpos) & (kpos > qpos - int(sliding_window))
        bias = np.where(allow, 0.0, -1e30).astype(np.float32)  # (L, Sk)
        bias = np.broadcast_to(bias, (B * Hq, L, Sk)).astype(np.float32)

    # Fold heads into batch for the rank-3 flash_attn lane.
    q3 = queries.reshape(B * Hq, L, Dh)
    k3 = keys.reshape(B * Hq, Sk, Dh)
    v3 = values.reshape(B * Hq, Sk, Dh)
    # The attention core is the rank-3 flash_attn(+attn_bias) workload. It
    # defaults to ops.flash_attn (eager / tape / @jit-traced) but a caller may
    # inject the Apple GPU metal_runtime dispatcher (or any equivalent) here.
    attn_core = attention_fn if attention_fn is not None else ops.flash_attn
    out = attn_core(q3, k3, v3, scale=scale, causal=False, attn_bias=bias)
    out = _asarray(out).reshape(B, Hq, L, Dh).transpose(0, 2, 1, 3).reshape(B, L, Hq * Dh)
    out = linear_general(out, o_proj)
    if return_ctx_kv:
        # (B, Hkv, S, Dh) -> (B, S, Hkv, Dh) to match the cache_keys/values layout.
        return out, ctx_k_this.transpose(0, 2, 1, 3), ctx_v_this.transpose(0, 2, 1, 3)
    return out


# Alias for torch-style `nn.flash_attention` callsites (same signature as ops.flash_attn).
flash_attention = ops.flash_attn


__all__ = [
    "adaptive_pool",
    "alibi",
    "avg_pool",
    "bidirectional_scan",
    "block_diffusion_attention",
    "conv1d",
    "conv_transpose",
    "einsum",
    "linear",
    "linear_general",
    "lora_linear",
    "group_norm",
    "gqa_attention",
    "gru_cell",
    "instance_norm",
    "mask_token_block",
    "max_pool",
    "min_pool",
    "mla_decode",
    "mqa_attention",
    "ntk_rope",
    "rms_norm",
    "moe_layer",
    "simple_rnn_cell",
    "spectral_norm",
    "swiglu",
    "weight_norm",
    "multi_head_attention",
    "flash_attention",
]
