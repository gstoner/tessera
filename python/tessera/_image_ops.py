"""Shared numpy helpers for the VLM image-preprocessing primitives.

Factored into a module (rather than closures inside ``_make_ops_namespace``)
so the forward references in ``tessera/__init__.py`` and the VJP/JVP rules in
``tessera/autodiff/{vjp,jvp}.py`` share one implementation of the bilinear
resample linear operator and the layout canonicalizer.

Layout strings: ``nchw`` (default), ``nhwc``, ``chw``, ``hwc``. Everything
operates on a canonical ``(N, C, H, W)`` view and restores the caller's layout.
The resample is a separable linear map (a 1-D weight matrix per spatial axis),
so its VJP is just the transpose of those matrices — exact, not finite-diff.
"""

from __future__ import annotations

import numpy as np


def img_unwrap(x):
    """Unwrap a Tessera Tensor wrapper to its backing array."""
    if hasattr(x, "_data"):
        x = x._data
    return np.asarray(x)


def img_canon(x, layout: str):
    """Return ``(x_nchw, restore)`` where ``restore`` maps an NCHW result back
    to ``layout``. Supports 3-D (chw/hwc) and 4-D (nchw/nhwc) tensors."""
    x = np.asarray(x)
    lay = layout.lower()
    if lay == "nchw":
        return x, (lambda o: o)
    if lay == "nhwc":
        return np.transpose(x, (0, 3, 1, 2)), (lambda o: np.transpose(o, (0, 2, 3, 1)))
    if lay == "chw":
        return x[None], (lambda o: o[0])
    if lay == "hwc":
        return np.transpose(x, (2, 0, 1))[None], (lambda o: np.transpose(o[0], (1, 2, 0)))
    raise ValueError(
        f"image op: unsupported layout {layout!r}; use one of nchw/nhwc/chw/hwc."
    )


def resize_matrix(in_size: int, out_size: int, align_corners: bool, mode: str) -> np.ndarray:
    """1-D resample weight matrix ``W`` of shape ``(out_size, in_size)`` such
    that ``out = W @ in`` along one spatial axis. ``mode`` ∈ {bilinear, nearest}.
    Coordinate convention matches ``torch.nn.functional.interpolate``.
    """
    if in_size <= 0 or out_size <= 0:
        raise ValueError("resize_matrix: sizes must be positive.")
    if mode not in ("bilinear", "nearest"):
        raise ValueError(
            f"image resample: unsupported mode {mode!r}; use 'bilinear' or 'nearest'."
        )
    w = np.zeros((out_size, in_size), dtype=np.float64)
    for o in range(out_size):
        if align_corners and out_size > 1:
            src = o * (in_size - 1) / (out_size - 1)
        else:
            src = (o + 0.5) * in_size / out_size - 0.5
        src = min(max(src, 0.0), in_size - 1.0)
        if mode == "nearest":
            j = int(min(max(round(src), 0), in_size - 1))
            w[o, j] = 1.0
        else:  # bilinear
            lo = int(np.floor(src))
            hi = min(lo + 1, in_size - 1)
            frac = src - lo
            w[o, lo] += 1.0 - frac
            w[o, hi] += frac
    return w


def resample_nchw(x_nchw: np.ndarray, out_hw, mode: str, align_corners: bool) -> np.ndarray:
    """Forward separable resample of an NCHW tensor to ``out_hw = (oh, ow)``."""
    x = np.asarray(x_nchw)
    _, _, h, w = x.shape
    oh, ow = int(out_hw[0]), int(out_hw[1])
    wh = resize_matrix(h, oh, align_corners, mode)   # (oh, h)
    ww = resize_matrix(w, ow, align_corners, mode)   # (ow, w)
    t = np.einsum("ph,nchw->ncpw", wh, x.astype(np.float64))  # resample H
    o = np.einsum("qw,ncpw->ncpq", ww, t)                     # resample W
    return o


def resample_nchw_vjp(dout_nchw: np.ndarray, in_hw, mode: str, align_corners: bool) -> np.ndarray:
    """Transpose of :func:`resample_nchw`: maps an NCHW cotangent at the output
    resolution back to ``in_hw = (h, w)`` at the input resolution."""
    dout = np.asarray(dout_nchw, dtype=np.float64)
    _, _, oh, ow = dout.shape
    h, w = int(in_hw[0]), int(in_hw[1])
    wh = resize_matrix(h, oh, align_corners, mode)   # (oh, h)
    ww = resize_matrix(w, ow, align_corners, mode)   # (ow, w)
    dt = np.einsum("qw,ncpq->ncpw", ww, dout)        # transpose of W resample
    dx = np.einsum("ph,ncpw->nchw", wh, dt)          # transpose of H resample
    return dx


def center_crop_bounds(h: int, w: int, ch: int, cw: int):
    """Top/left offsets for a centered ``(ch, cw)`` crop of an ``(h, w)`` image."""
    if ch > h or cw > w:
        raise ValueError(
            f"center_crop: crop {(ch, cw)} exceeds image {(h, w)}."
        )
    return (h - ch) // 2, (w - cw) // 2


def pixel_unshuffle_nchw(x: np.ndarray, r: int) -> np.ndarray:
    """Space-to-depth: ``(B, C, H, W) → (B, C*r*r, H/r, W/r)`` (torch
    ``pixel_unshuffle`` ordering). Pure permute/reshape."""
    x = np.asarray(x)
    b, c, h, w = x.shape
    if r < 1 or h % r or w % r:
        raise ValueError(
            f"pixel_unshuffle: H={h}, W={w} must be divisible by factor r={r}."
        )
    t = x.reshape(b, c, h // r, r, w // r, r)
    t = np.transpose(t, (0, 1, 3, 5, 2, 4))   # (B, C, r, r, H/r, W/r)
    return t.reshape(b, c * r * r, h // r, w // r)


def pixel_shuffle_nchw(x: np.ndarray, r: int) -> np.ndarray:
    """Depth-to-space: ``(B, C*r*r, H, W) → (B, C, H*r, W*r)`` (torch
    ``pixel_shuffle`` ordering). Inverse of :func:`pixel_unshuffle_nchw`."""
    x = np.asarray(x)
    b, cr, h, w = x.shape
    if r < 1 or cr % (r * r):
        raise ValueError(
            f"pixel_shuffle: channel dim {cr} must be divisible by r*r={r * r}."
        )
    c = cr // (r * r)
    t = x.reshape(b, c, r, r, h, w)
    t = np.transpose(t, (0, 1, 4, 2, 5, 3))   # (B, C, H, r, W, r)
    return t.reshape(b, c, h * r, w * r)
