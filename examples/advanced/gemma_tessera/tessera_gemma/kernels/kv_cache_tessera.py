
import torch
from torch import Tensor
from typing import List, Tuple, Iterator

class PagedKVCache:
    """
    Minimal paged KV cache for (B, T, Hkv, Dh).
    Stores a list of pages with length <= page_size. New tokens are appended to pages.
    """
    def __init__(self, batch: int, kv_heads: int, head_dim: int, *, page_size: int = 128,
                 device=None, dtype=None):
        self.B = batch
        self.Hkv = kv_heads
        self.Dh = head_dim
        self.page_size = int(page_size)
        self.device = device
        self.dtype = dtype or torch.float16
        self._k_pages: List[Tensor] = []
        self._v_pages: List[Tensor] = []
        self._len = 0

    def __len__(self) -> int:
        return self._len

    def _ensure_page(self) -> None:
        if not self._k_pages or self._k_pages[-1].shape[1] >= self.page_size:
            # start a new empty page
            empty_k = torch.empty(self.B, 0, self.Hkv, self.Dh, device=self.device, dtype=self.dtype)
            empty_v = torch.empty_like(empty_k)
            self._k_pages.append(empty_k)
            self._v_pages.append(empty_v)

    def append(self, k: Tensor, v: Tensor) -> None:
        """
        Append tokens to cache. k,v shapes: (B, T, Hkv, Dh)
        """
        assert k.shape == v.shape, "K/V shape mismatch"
        B, T, Hkv, Dh = k.shape
        assert B == self.B and Hkv == self.Hkv and Dh == self.Dh, "shape mismatch to cache"
        if T == 0:
            return
        t = 0
        while t < T:
            self._ensure_page()
            room = self.page_size - self._k_pages[-1].shape[1]
            take = min(room, T - t)
            k_slice = k[:, t:t+take]
            v_slice = v[:, t:t+take]
            # append by concatenation along time dimension
            self._k_pages[-1] = torch.cat([self._k_pages[-1], k_slice], dim=1)
            self._v_pages[-1] = torch.cat([self._v_pages[-1], v_slice], dim=1)
            self._len += take
            t += take

    def pages(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for kp, vp in zip(self._k_pages, self._v_pages):
            if kp.shape[1] > 0:
                yield kp, vp
