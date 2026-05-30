"""MLA paged-cache decoder — production-serving wiring (2026-05-30).

Promotes the "growing numpy arrays" MLA decode integration into Tessera's real
paged-cache ABI. An :class:`MLAPagedDecoder` bundles **two**
:class:`LatentKVCacheHandle` instances — one for the compressed latent ``c_kv``
``[max_seq, latent_dim]`` and a tiny one for the shared decoupled-RoPE key slice
``k_rope`` ``[max_seq, rope_dim]`` — together with the absorbed up-projection
weights and a RoPE table generator, and drives the on-GPU weight-absorbed MLA
decode kernel (``tessera_apple_gpu_mla_absorb_decode_f32``).

The cache stores only the latent + rope slices (shared across all heads); per
head K/V are never materialized — the ~8.9× KV-cache reduction DeepSeek reports.

Serving loop::

    dec = MLAPagedDecoder(num_heads=H, nope_dim=dn, rope_dim=dr, v_dim=dv,
                          latent_dim=Dl, Wuk_t=Wuk_t, Wuv=Wuv, max_seq=4096)
    dec.append(c_kv_prefill, k_rope_prefill)   # prefill the prompt
    for step in decode_loop:
        dec.append(c_kv_t, k_rope_t)           # cache the new token
        out = dec.decode(q_nope_t, q_rope_t)   # [H, v_dim]

On non-Apple hosts (or when the GPU symbol is unavailable) the decoder
transparently falls back to a numpy reference, so the serving loop is portable.
Absolute token positions are tracked across eviction so sliding-window decode
stays RoPE-correct.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from .latent import LatentKVCacheHandle


def _rope_apply(x: np.ndarray, cos: np.ndarray, sin: np.ndarray,
                style: str) -> np.ndarray:
    """x [..., dr]; cos/sin broadcast to [..., dr/2]."""
    dr = x.shape[-1]
    half = dr // 2
    out = np.empty_like(x)
    if style == "interleaved":
        a, b = x[..., 0::2], x[..., 1::2]
        out[..., 0::2] = a * cos - b * sin
        out[..., 1::2] = a * sin + b * cos
    else:
        a, b = x[..., :half], x[..., half:]
        out[..., :half] = a * cos - b * sin
        out[..., half:] = b * cos + a * sin
    return out


def _reference_absorb(q_nope, q_rope, c_kv, k_rope, Wuk_t, Wuv,
                      cosQ, sinQ, cosK, sinK, style):
    """Numpy reference of the absorbed decode. Shapes match the GPU kernel:
    q_nope [H,dn], q_rope [H,dr], c_kv [S,Dl], k_rope [S,dr],
    Wuk_t [H,dn,Dl], Wuv [H,Dl,dv], cosQ/sinQ [1,dr/2], cosK/sinK [S,dr/2].
    Returns [H, dv]."""
    q_nope = q_nope.astype(np.float64)
    q_rope = q_rope.astype(np.float64)
    c_kv = c_kv.astype(np.float64)
    k_rope = k_rope.astype(np.float64)
    Wuk_t, Wuv = Wuk_t.astype(np.float64), Wuv.astype(np.float64)
    H, dn = q_nope.shape
    dr = q_rope.shape[-1]
    S, Dl = c_kv.shape
    dv = Wuv.shape[-1]
    scale = 1.0 / math.sqrt(dn + dr)
    krR = _rope_apply(k_rope, cosK.astype(np.float64), sinK.astype(np.float64),
                      style)                                        # [S,dr]
    O = np.empty((H, dv))
    for h in range(H):
        qabs = q_nope[h] @ Wuk_t[h]                                 # [Dl]
        qrR = _rope_apply(q_rope[h], cosQ[0].astype(np.float64),
                          sinQ[0].astype(np.float64), style)        # [dr]
        s = qabs @ c_kv.T + qrR @ krR.T                            # [S]
        s = s * scale
        s = s - s.max()
        e = np.exp(s)
        attn = e / e.sum()
        ctx = attn @ c_kv                                          # [Dl]
        O[h] = ctx @ Wuv[h]                                        # [dv]
    return O


class MLAPagedDecoder:
    """Single-sequence MLA decoder over a paged latent + rope cache.

    Parameters
    ----------
    num_heads, nope_dim, rope_dim, v_dim, latent_dim
        MLA head geometry. ``rope_dim`` must be even.
    Wuk_t
        Absorbed K up-projection, shape ``[num_heads, nope_dim, latent_dim]``
        (i.e. ``Wuk`` transposed on its last two axes).
    Wuv
        V up-projection, shape ``[num_heads, latent_dim, v_dim]``.
    max_seq
        Cache capacity in tokens.
    rope_base
        RoPE frequency base (default 10000.0).
    rotation_style
        ``"interleaved"`` (NeoX even/odd) or ``"half"`` (GPT-J split-halves).
    dtype, page_size, auto_evict
        Forwarded to the underlying :class:`LatentKVCacheHandle` instances.
        ``auto_evict=True`` enables sliding-window decode.
    """

    def __init__(self, *, num_heads: int, nope_dim: int, rope_dim: int,
                 v_dim: int, latent_dim: int, Wuk_t: Any, Wuv: Any,
                 max_seq: int, rope_base: float = 10000.0,
                 rotation_style: str = "interleaved", dtype: str = "fp32",
                 page_size: int = 128, auto_evict: bool = False) -> None:
        if rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be even, got {rope_dim}")
        if rotation_style not in ("interleaved", "half"):
            raise ValueError(f"rotation_style must be interleaved|half, "
                             f"got {rotation_style!r}")
        self.num_heads = int(num_heads)
        self.nope_dim = int(nope_dim)
        self.rope_dim = int(rope_dim)
        self.v_dim = int(v_dim)
        self.latent_dim = int(latent_dim)
        self.rope_base = float(rope_base)
        self.rotation_style = rotation_style

        self.Wuk_t = np.ascontiguousarray(Wuk_t, np.float32)
        self.Wuv = np.ascontiguousarray(Wuv, np.float32)
        if self.Wuk_t.shape != (self.num_heads, self.nope_dim, self.latent_dim):
            raise ValueError(
                f"Wuk_t must be [H,nope_dim,latent_dim] = "
                f"{(self.num_heads, self.nope_dim, self.latent_dim)}; "
                f"got {self.Wuk_t.shape}")
        if self.Wuv.shape != (self.num_heads, self.latent_dim, self.v_dim):
            raise ValueError(
                f"Wuv must be [H,latent_dim,v_dim] = "
                f"{(self.num_heads, self.latent_dim, self.v_dim)}; "
                f"got {self.Wuv.shape}")

        self.latent_cache = LatentKVCacheHandle(
            latent_dim=self.latent_dim, max_seq=max_seq, dtype=dtype,
            page_size=page_size, auto_evict=auto_evict)
        self.rope_cache = LatentKVCacheHandle(
            latent_dim=self.rope_dim, max_seq=max_seq, dtype=dtype,
            page_size=page_size, auto_evict=auto_evict)
        # Absolute position of cache index 0 — advances as tokens are evicted
        # so RoPE stays correct under sliding-window decode.
        self._abs_base = 0

    # ------------------------------------------------------------------
    @property
    def current_seq(self) -> int:
        return self.latent_cache.current_seq

    @property
    def num_pages(self) -> int:
        return self.latent_cache.num_pages

    def cache_bytes_per_token(self) -> int:
        """Latent + rope bytes per token (shared across heads) — the footprint
        that beats per-head K/V."""
        elem = self.latent_cache.latents.itemsize
        return (self.latent_dim + self.rope_dim) * elem

    # ------------------------------------------------------------------
    def append(self, c_kv: Any, k_rope: Any) -> "MLAPagedDecoder":
        """Append ``n_new`` tokens' compressed latent + shared rope key.

        c_kv ``[n_new, latent_dim]``, k_rope ``[n_new, rope_dim]``. Tracks the
        absolute base so RoPE positions stay consistent across eviction."""
        c_arr = np.asarray(c_kv)
        r_arr = np.asarray(k_rope)
        if c_arr.ndim == 1:
            c_arr = c_arr.reshape(1, -1)
        if r_arr.ndim == 1:
            r_arr = r_arr.reshape(1, -1)
        if c_arr.shape[0] != r_arr.shape[0]:
            raise ValueError(
                f"latent and rope chunks must append the same number of tokens; "
                f"got {c_arr.shape[0]} and {r_arr.shape[0]}")
        before = self.latent_cache.current_seq
        n_new = c_arr.shape[0]
        self.latent_cache.append(c_arr)
        self.rope_cache.append(r_arr)
        evicted = before + n_new - self.latent_cache.current_seq
        if evicted > 0:
            self._abs_base += evicted
        return self

    def evict_oldest(self, n: int) -> "MLAPagedDecoder":
        """Drop the oldest ``n`` tokens from both caches (rolling window)."""
        n = min(int(n), self.current_seq)
        if n <= 0:
            return self
        self.latent_cache.evict_oldest(n)
        self.rope_cache.evict_oldest(n)
        self._abs_base += n
        return self

    # ------------------------------------------------------------------
    def _rope_tables(self, positions: np.ndarray):
        half = self.rope_dim // 2
        inv = self.rope_base ** (
            -(np.arange(half, dtype=np.float64) * 2.0 / self.rope_dim))
        ang = positions.astype(np.float64)[:, None] * inv[None, :]
        return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)

    def decode(self, q_nope: Any, q_rope: Any,
               query_pos: Optional[int] = None) -> np.ndarray:
        """Decode a single query against the full cached window.

        q_nope ``[num_heads, nope_dim]``, q_rope ``[num_heads, rope_dim]``.
        ``query_pos`` defaults to the most recent token's absolute position
        (``current_seq - 1`` + base). Returns ``[num_heads, v_dim]``."""
        S = self.current_seq
        if S == 0:
            raise ValueError("decode called on an empty cache; append first")
        qn = np.ascontiguousarray(q_nope, np.float32)
        qr = np.ascontiguousarray(q_rope, np.float32)
        if qn.shape != (self.num_heads, self.nope_dim):
            raise ValueError(
                f"q_nope must be [num_heads,nope_dim]={(self.num_heads, self.nope_dim)}; "
                f"got {qn.shape}")
        if qr.shape != (self.num_heads, self.rope_dim):
            raise ValueError(
                f"q_rope must be [num_heads,rope_dim]={(self.num_heads, self.rope_dim)}; "
                f"got {qr.shape}")

        c_kv = np.ascontiguousarray(self.latent_cache.read(0, S), np.float32)
        k_rope = np.ascontiguousarray(self.rope_cache.read(0, S), np.float32)
        key_pos = self._abs_base + np.arange(S)
        if query_pos is None:
            query_pos = self._abs_base + S - 1
        cosK, sinK = self._rope_tables(key_pos)
        cosQ, sinQ = self._rope_tables(np.asarray([query_pos]))

        out = self._gpu_decode(qn, qr, c_kv, k_rope, cosQ, sinQ, cosK, sinK)
        if out is None:
            out = _reference_absorb(qn, qr, c_kv, k_rope, self.Wuk_t, self.Wuv,
                                    cosQ, sinQ, cosK, sinK, self.rotation_style)
        return out

    def _gpu_decode(self, qn, qr, c_kv, k_rope, cosQ, sinQ, cosK, sinK):
        """Try the on-GPU absorbed kernel; return None to fall back."""
        try:
            from .. import runtime as R
        except Exception:  # pragma: no cover
            return None
        dispatch = getattr(R, "_apple_gpu_mla_absorb_decode", None)
        if dispatch is None:
            return None
        # kernel wants [B,H,Sq,*] with B=Sq=1
        qn4 = qn[None, :, None, :]
        qr4 = qr[None, :, None, :]
        ckv3 = c_kv[None]
        kr3 = k_rope[None]
        res = dispatch(qn4, qr4, ckv3, kr3, self.Wuk_t, self.Wuv, cosQ, sinQ,
                       cosK, sinK, np, rotation_style=self.rotation_style)
        if res is None:
            return None
        return np.ascontiguousarray(res[0, :, 0, :])

    def __repr__(self) -> str:
        return (f"MLAPagedDecoder(num_heads={self.num_heads}, "
                f"latent_dim={self.latent_dim}, rope_dim={self.rope_dim}, "
                f"v_dim={self.v_dim}, current_seq={self.current_seq}, "
                f"abs_base={self._abs_base}, style={self.rotation_style!r})")
