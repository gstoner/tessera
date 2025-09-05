# SPDX-License-Identifier: MIT
# Utilities for FFT/DCT and spectral convolution in NumPy and optional PyTorch.

import math
import time
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False


def _as_numpy(x):
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _sync_device(device: str):
    if _HAS_TORCH and device in ("cuda", "rocm"):
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def estimate_fft_flops(shape, complex_op=True, real_op=False):
    """Return estimated real FLOPs for FFT over given shape (supports 1D/2D).
    For a complex FFT: 5*N*log2(N). For 2D HxW: 5*H*W*(log2(H)+log2(W)).
    For real transforms (R2C/C2R), use ~0.5x the complex estimate.
    """
    if isinstance(shape, int):
        n = int(shape)
        flops = 5.0 * n * math.log2(max(2, n))
    else:
        # assume tuple/list
        if len(shape) == 1:
            n = int(shape[0])
            flops = 5.0 * n * math.log2(max(2, n))
        elif len(shape) == 2:
            h, w = int(shape[0]), int(shape[1])
            flops = 5.0 * h * w * (math.log2(max(2, h)) + math.log2(max(2, w)))
        else:
            n = int(np.prod(shape))
            flops = 5.0 * n * math.log2(max(2, n))
    if real_op:
        flops *= 0.5
    return flops


def estimate_bytes(n_elems, dtype=np.float32, factor=2.0):
    """Approximate bytes moved; factor~=2 for read+write in-place."""
    itemsize = np.dtype(dtype).itemsize
    return float(n_elems * itemsize * factor)


def dct2_numpy(x, type=2, axis=-1):
    """1D DCT-II/III via FFT identities in NumPy only (CPU).
    type=2 or type=3 supported. axis can be any valid axis.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[axis]
    if type == 2:
        # DCT-II: even extension & FFT
        y = np.concatenate([x, np.flip(x, axis=axis)], axis=axis)
        Y = np.fft.fft(y, axis=axis)
        k = np.arange(n)
        # phase factor: exp(-i*pi*k/(2N))
        phase = np.exp(-1j * np.pi * k / (2.0 * n))
        slicer = [slice(None)] * Y.ndim
        slicer[axis] = slice(0, n)
        X = (Y[tuple(slicer)] * phase).real
        return X
    elif type == 3:
        # DCT-III is the inverse of DCT-II up to scale; we implement via DCT-II by transpose trick
        # Simple (but not the fastest): use the definition O(N^2) for reference-sized vectors
        k = np.arange(n)
        const = np.pi / n
        # alpha_0 = 0.5, alpha_k = 1 for k>0
        alpha = np.ones(n)
        alpha[0] = 0.5
        # X[m] = sum_{k} alpha_k x[k] cos(const*(m+0.5)*k)
        m = np.arange(n)[:, None]
        X = np.sum(alpha[None, :] * x * np.cos(const * (m + 0.5) * k[None, :]), axis=1)
        # Make axis placement consistent with input axis
        return np.moveaxis(X, 0, axis)
    else:
        raise ValueError("Only DCT types 2 and 3 supported.")


def dct2_numpy_2d(x, type=2):
    X = dct2_numpy(dct2_numpy(x, type=type, axis=0), type=type, axis=1)
    return X


def spectral_conv1d_fft(x, w, device="cpu", backend="numpy"):
    """Valid-size 1D convolution using FFT (circular or zero-padded to next pow2).
    Returns y and a dict with perf hooks on request (only timing done by caller).
    """
    if backend == "torch" and _HAS_TORCH:
        # Use next pow2 for speed
        n = x.shape[-1] + w.shape[-1] - 1
        nfft = 1 << (int(math.ceil(math.log2(n))))
        X = torch.fft.rfft(x, n=nfft)
        W = torch.fft.rfft(w, n=nfft)
        Y = X * W
        y = torch.fft.irfft(Y, n=nfft)
        # Crop to valid size
        valid = x.shape[-1] + w.shape[-1] - 1
        return y[..., :valid]
    else:
        n = x.shape[-1] + w.shape[-1] - 1
        nfft = 1 << (int(math.ceil(math.log2(n))))
        X = np.fft.rfft(x, n=nfft)
        W = np.fft.rfft(w, n=nfft)
        Y = X * W
        y = np.fft.irfft(Y, n=nfft)
        valid = x.shape[-1] + w.shape[-1] - 1
        return y[..., :valid]


def spectral_conv2d_fft(x, w, device="cpu", backend="numpy"):
    """2D convolution via FFT (valid-size), zero-padded to next pow2 per dim."""
    if backend == "torch" and _HAS_TORCH:
        H = x.shape[-2] + w.shape[-2] - 1
        W = x.shape[-1] + w.shape[-1] - 1
        Hfft = 1 << int(math.ceil(math.log2(H)))
        Wfft = 1 << int(math.ceil(math.log2(W)))
        X = torch.fft.rfft2(x, s=(Hfft, Wfft))
        Wf = torch.fft.rfft2(w, s=(Hfft, Wfft))
        Y = X * Wf
        y = torch.fft.irfft2(Y, s=(Hfft, Wfft))
        return y[..., :H, :W]
    else:
        H = x.shape[-2] + w.shape[-2] - 1
        W = x.shape[-1] + w.shape[-1] - 1
        Hfft = 1 << int(math.ceil(math.log2(H)))
        Wfft = 1 << int(math.ceil(math.log2(W)))
        X = np.fft.rfft2(x, s=(Hfft, Wfft))
        Wf = np.fft.rfft2(w, s=(Hfft, Wfft))
        Y = X * Wf
        y = np.fft.irfft2(Y, s=(Hfft, Wfft))
        return y[..., :H, :W]
