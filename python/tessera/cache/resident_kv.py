"""Device-resident latent KV cache — R4.

The final phase of the GPU-resident architecture: the compressed latent
``c_kv`` and the shared decoupled-RoPE key slice ``k_rope`` live in **resident
device buffers** that persist across decode steps. So:

  * **append writes only the new token** straight into the resident buffer (a
    host write through the unified-memory `.numpy()` view — no upload, the GPU
    sees it immediately), instead of re-sending the whole window every step; and
  * the decode reads the populated window as a **zero-copy prefix view** of the
    resident buffer (no per-step cache copy).

Contrast with the host caches (`MLAPagedDecoder` / `MLABlockPagedCache`), which
hold numpy state and re-wrap/re-upload the window each step. `ResidentLatentKVCache`
is the device-resident backing those would adopt for serving.

This covers the **contiguous** latent cache. A device-resident *block-paged*
cache (non-contiguous block-table gather on-device) is the documented follow-on.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import runtime as R


class ResidentLatentKVCache:
    """Growing, device-resident ``c_kv`` + ``k_rope`` cache.

    Parameters
    ----------
    latent_dim, rope_dim
        Widths of the compressed latent and the shared rope key slice.
    max_seq
        Capacity in tokens. ``append`` past this raises (eviction is a follow-on).
    """

    def __init__(self, *, latent_dim: int, rope_dim: int, max_seq: int) -> None:
        if latent_dim <= 0 or rope_dim <= 0 or max_seq <= 0:
            raise ValueError("latent_dim / rope_dim / max_seq must be positive")
        self.latent_dim = int(latent_dim)
        self.rope_dim = int(rope_dim)
        self.max_seq = int(max_seq)
        self.current_seq = 0
        dt = R.DeviceTensor
        self._latent = dt.empty((self.max_seq, self.latent_dim), np.float32)
        self._rope = dt.empty((self.max_seq, self.rope_dim), np.float32)
        self._resident = self._latent is not None and self._rope is not None
        if self._latent is not None and self._rope is not None:
            # views over the resident storage — host writes go straight to the
            # buffer the GPU reads (unified memory), no upload.
            self._latent_view = self._latent.numpy()
            self._rope_view = self._rope.numpy()
        else:  # portable fallback: plain numpy backing
            self._latent_view = np.zeros((self.max_seq, self.latent_dim), np.float32)
            self._rope_view = np.zeros((self.max_seq, self.rope_dim), np.float32)

    # ------------------------------------------------------------------
    @property
    def resident(self) -> bool:
        return self._resident

    @property
    def shape(self) -> tuple[int, int]:
        return (self.current_seq, self.latent_dim)

    def cache_bytes_per_token(self) -> int:
        return (self.latent_dim + self.rope_dim) * 4

    # ------------------------------------------------------------------
    def append(self, c_kv: Any, k_rope: Any) -> "ResidentLatentKVCache":
        """Append ``n`` tokens' latent + rope key. The data is written **in
        place** into the resident device buffer — no upload, no reallocation."""
        c = np.asarray(c_kv, np.float32)
        r = np.asarray(k_rope, np.float32)
        if c.ndim == 1:
            c = c.reshape(1, -1)
        if r.ndim == 1:
            r = r.reshape(1, -1)
        if c.shape[1] != self.latent_dim or r.shape[1] != self.rope_dim:
            raise ValueError(
                f"expected c_kv [*,{self.latent_dim}] / k_rope [*,{self.rope_dim}]; "
                f"got {c.shape} / {r.shape}")
        if c.shape[0] != r.shape[0]:
            raise ValueError("c_kv and k_rope must append the same token count")
        n = c.shape[0]
        if self.current_seq + n > self.max_seq:
            raise ValueError(
                f"append would exceed max_seq={self.max_seq}: "
                f"current={self.current_seq}, appending={n}")
        s, e = self.current_seq, self.current_seq + n
        self._latent_view[s:e] = c
        self._rope_view[s:e] = r
        self.current_seq = e
        return self

    # ------------------------------------------------------------------
    def latent_window(self) -> Any:
        """Resident ``DeviceTensor`` view of the populated latent
        ``[current_seq, latent_dim]`` (zero-copy prefix). Falls back to a numpy
        slice when not resident."""
        if self._resident and self._latent is not None:
            return self._latent.prefix_view(self.current_seq)
        return self._latent_view[:self.current_seq]

    def rope_window(self) -> Any:
        if self._resident and self._rope is not None:
            return self._rope.prefix_view(self.current_seq)
        return self._rope_view[:self.current_seq]

    def latent_numpy(self) -> Any:
        """Zero-copy numpy view of the populated latent window (no download)."""
        return self._latent_view[:self.current_seq]

    def rope_numpy(self) -> Any:
        return self._rope_view[:self.current_seq]

    def free(self) -> None:
        for t in (self._latent, self._rope):
            if t is not None:
                t.free()
        self._latent = self._rope = None
        self._resident = False

    def __del__(self) -> None:
        try:
            self.free()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (f"ResidentLatentKVCache(latent_dim={self.latent_dim}, "
                f"rope_dim={self.rope_dim}, current_seq={self.current_seq}/"
                f"{self.max_seq}, resident={self._resident})")
