import os
import numpy as np

_BACKEND = os.environ.get("TESSERA_NUMERICS_BACKEND", "numpy").lower()

def backend_name():
    return _BACKEND

# Optional torch import
_torch = None
if _BACKEND in ("torch", "tessera"):
    try:
        import torch
        _torch = torch
    except Exception:
        _torch = None

# --- Utilities ---
def to_numpy(x):
    if _torch is not None and isinstance(x, _torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def rng(seed=1234):
    if _BACKEND in ("torch", "tessera") and _torch is not None:
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)

# --- Ops ---
def matmul(a, b, dtype="float32"):
    if _BACKEND == "numpy":
        return (a.astype(dtype) @ b.astype(dtype)).astype(dtype)
    if _BACKEND == "torch" and _torch is not None:
        device = "cuda" if _torch.cuda.is_available() else "cpu"
        ta = _torch.tensor(a, dtype=getattr(_torch, dtype), device=device)
        tb = _torch.tensor(b, dtype=getattr(_torch, dtype), device=device)
        return (ta @ tb).cpu().numpy()
    if _BACKEND == "tessera":
        # TODO: replace with real Tessera binding call
        # For now fall through to torch if available else numpy
        if _torch is not None:
            return matmul(a, b, dtype=dtype).copy()
        return (a.astype(dtype) @ b.astype(dtype)).astype(dtype)
    raise RuntimeError("Unknown backend")

def softmax(x, axis=-1, dtype="float32"):
    # numerically-stable softmax
    if _BACKEND == "numpy" or _torch is None:
        x = x.astype(dtype)
        x = x - np.max(x, axis=axis, keepdims=True)
        ex = np.exp(x, dtype=np.float64)
        return (ex / np.sum(ex, axis=axis, keepdims=True)).astype(dtype)
    if _BACKEND in ("torch", "tessera"):
        t = _torch.tensor(x, dtype=getattr(_torch, dtype), device="cuda" if _torch.cuda.is_available() else "cpu")
        y = _torch.nn.functional.softmax(t, dim=axis)
        return y.detach().cpu().numpy()
    raise RuntimeError("Unknown backend")

def conv2d_nhwc(x, w, stride=(1,1), padding=(0,0), dtype="float32"):
    # Reference via torch if available; otherwise simple im2col
    if _BACKEND in ("torch", "tessera") and _torch is not None:
        import torch.nn.functional as F
        device = "cuda" if _torch.cuda.is_available() else "cpu"
        xt = _torch.tensor(x, dtype=getattr(_torch, dtype), device=device).permute(0,3,1,2)  # NHWC->NCHW
        wt = _torch.tensor(w, dtype=getattr(_torch, dtype), device=device).permute(3,2,0,1)  # HWIO->OIH
        y = F.conv2d(xt, wt, stride=stride, padding=padding)
        return y.permute(0,2,3,1).detach().cpu().numpy()
    # Fallback im2col (slow, but fine for tests)
    N,H,W,C = x.shape
    KH,KW,CI,CO = w.shape
    SH,SW = stride
    PH,PW = padding
    H2 = (H + 2*PH - KH)//SH + 1
    W2 = (W + 2*PW - KW)//SW + 1
    xpad = np.pad(x, ((0,0),(PH,PH),(PW,PW),(0,0)))
    cols = []
    for i in range(H2):
        for j in range(W2):
            patch = xpad[:, i*SH:i*SH+KH, j*SW:j*SW+KW, :].reshape(N, -1)
            cols.append(patch)
    cols = np.stack(cols, axis=1)  # N, (H2*W2), (KH*KW*CI)
    wmat = w.reshape(-1, CO)       # (KH*KW*CI), CO
    out = cols @ wmat              # N,(H2*W2),CO
    out = out.reshape(N, H2, W2, CO).astype(dtype)
    return out
