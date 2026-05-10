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
    y = np.tensordot(x_arr, w_arr, axes=(axes, tuple(range(len(axes)))))
    if bias is not None:
        y = y + _asarray(bias)
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


# Alias for torch-style `nn.flash_attention` callsites (same signature as ops.flash_attn).
flash_attention = ops.flash_attn


__all__ = [
    "adaptive_pool",
    "alibi",
    "avg_pool",
    "bidirectional_scan",
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
    "max_pool",
    "min_pool",
    "mla_decode",
    "mqa_attention",
    "ntk_rope",
    "rms_norm",
    "simple_rnn_cell",
    "spectral_norm",
    "swiglu",
    "weight_norm",
    "multi_head_attention",
    "flash_attention",
]
