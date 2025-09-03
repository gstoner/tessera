# Copyright (c) 2025
# Minimal KAN layer built for Tessera graphs with a NumPy fallback.
from dataclasses import dataclass
from typing import Optional, Tuple
import math

try:
    import numpy as np
except Exception:
    np = None

try:
    import tessera as tsr  # type: ignore
    _HAS_TESSERA = True
except Exception:
    _HAS_TESSERA = False

@dataclass
def bspline_params(k:int=3, grid_min:float=-1.0, grid_max:float=1.0, grid_size:int=16):
    return {"degree": int(k), "grid_min": float(grid_min), "grid_max": float(grid_max), "grid_size": int(grid_size)}

class KANLinear:
    """KANLinear for Tessera.

    Args:
        in_features: I
        out_features: O
        k: spline degree (1=linear,2=quadratic,3=cubic)
        grid_min, grid_max: support of the basis
        grid_size: number of interior cells (M ≈ grid_size + k - 1 basis funcs)
        enable_standalone_scale_spline: add per-(I,M) multiplicative scale (as in Efficient‑KAN option)
    """
    def __init__(self,
                 in_features:int,
                 out_features:int,
                 k:int=3,
                 grid_min:float=-1.0,
                 grid_max:float=1.0,
                 grid_size:int=16,
                 enable_standalone_scale_spline:bool=True,
                 dtype:str="f32"):
        self.I = in_features
        self.O = out_features
        self.k = int(k)
        self.gmin = float(grid_min)
        self.gmax = float(grid_max)
        self.gsz = int(grid_size)
        self.enable_standalone_scale_spline = bool(enable_standalone_scale_spline)
        self.dtype = dtype

        # Parameter shapes
        M = self.gsz + self.k - 1
        # Base linear weight and spline mixing weights
        rng = np.random.default_rng(42) if np else None
        self.W_base = (rng.standard_normal((self.I, self.O)) * (1.0 / math.sqrt(self.I))).astype(np.float32) if np else None
        self.W_spline = (rng.standard_normal((self.I, M, self.O)) * (1.0 / math.sqrt(self.I))).astype(np.float32) if np else None
        self.spline_scale = (rng.standard_normal((self.I, M))).astype(np.float32) if (np and enable_standalone_scale_spline) else None

    # ---------- public API ----------
    def __call__(self, x):
        """x: [B, I] → y: [B, O]"""
        if _HAS_TESSERA:
            return self._forward_tessera(x)
        else:
            return self._forward_numpy(x)

    # ---------- numpy reference path ----------
    def _forward_numpy(self, x):
        assert np is not None, "NumPy not available for fallback."
        x = np.asarray(x, dtype=np.float32)
        B, I = x.shape
        assert I == self.I, f"Expected last dim {self.I}, got {I}"
        phi = _bspline_eval_numpy(x, self.k, self.gmin, self.gmax, self.gsz)  # [B, I, M]
        if self.spline_scale is not None:
            phi = phi * self.spline_scale[None, :, :]  # broadcast [B, I, M]
        # Contract over (I,M) with weights (I,M,O): reshape to 2D then matmul
        B, I, M = phi.shape
        phi2 = phi.reshape(B, I*M)                     # [B, I*M]
        Wmix = self.W_spline.reshape(I*M, self.O)      # [I*M, O]
        y = phi2 @ Wmix                                # [B, O]
        if self.W_base is not None:
            y = y + (x @ self.W_base)                  # residual linear term
        return y

    # ---------- tessera graph path (minimal) ----------
    def _forward_tessera(self, x):
        # Expect x to be a tsr.Tensor or array-like convertible via tsr.tensor
        X = x if hasattr(x, 'shape') and getattr(x, 'is_tessera', False) else tsr.tensor(x, dtype=self.dtype)
        phi = tsr.ops.kan_bspline_eval(X,
                                       degree=self.k,
                                       grid_min=self.gmin,
                                       grid_max=self.gmax,
                                       grid_size=self.gsz)              # [B, I, M]
        if self.enable_standalone_scale_spline and self.spline_scale is not None:
            S = tsr.tensor(self.spline_scale, dtype=self.dtype)          # [I, M]
            phi = tsr.ops.mul(phi, tsr.ops.broadcast(S, phi.shape))
        # Fold (I,M) and do GEMM with W_spline
        B, I, M = phi.shape
        phi2 = tsr.ops.reshape(phi, (B, I*M))                            # [B, I*M]
        Wmix = tsr.tensor(self.W_spline.reshape(I*M, self.O), dtype=self.dtype)
        Y = tsr.ops.matmul(phi2, Wmix)                                   # [B, O]
        if self.W_base is not None:
            Wb = tsr.tensor(self.W_base, dtype=self.dtype)
            Y = tsr.ops.add(Y, tsr.ops.matmul(X, Wb))
        return Y

# ---------------- helper: NumPy B-spline evaluator ----------------
def _make_uniform_knots(gmin:float, gmax:float, grid_size:int, degree:int):
    # Open uniform knot vector with repeats at ends
    # number of basis M = grid_size + degree - 1
    n_cells = grid_size
    k = degree
    # knots length = n_cells + 2*k
    t0, t1 = float(gmin), float(gmax)
    interior = np.linspace(t0, t1, n_cells+1)[1:-1] if n_cells > 1 else np.array([], dtype=np.float32)
    left = np.full(k, t0, dtype=np.float32)
    right = np.full(k, t1, dtype=np.float32)
    knots = np.concatenate([left, interior.astype(np.float32), right])
    return knots

def _bspline_eval_numpy(x: "np.ndarray", degree:int, gmin:float, gmax:float, grid_size:int):
    assert np is not None
    B, I = x.shape
    k = degree
    knots = _make_uniform_knots(gmin, gmax, grid_size, k)  # length: grid_size + 2*k - 1
    # number of basis functions M equals len(knots)-k-1 for open uniform
    M = len(knots) - k - 1
    # Evaluate per feature independently, vectorized over batch
    # Cox–de Boor iterative DP
    # Start with B-spline of degree 0 (indicator on intervals)
    # For numerical stability, clamp x into [gmin, gmax]
    xv = np.clip(x, gmin, gmax)
    # Map x to knot spans indices
    # Build phi for degree 0: N_{i,0}(x)
    # Using DP arrays of shape [B, I, M]
    phi = np.zeros((B, I, M), dtype=np.float32)
    # Degree 0 basis
    for m in range(M):
        t_m = knots[m]
        t_m1 = knots[m+1]
        # Indicator for [t_m, t_{m+1})
        mask = (xv >= t_m) & (xv < t_m1) if m < M-1 else (xv >= t_m) & (xv <= t_m1 if m+1 < len(knots) else xv<=gmax)
        phi[:, :, m] = mask.astype(np.float32)
    # Elevate degree iteratively
    for d in range(1, k+1):
        phi_next = np.zeros_like(phi)
        for m in range(M):
            t_m = knots[m]
            t_m_d = knots[m+d]
            t_m1 = knots[m+1]
            t_m1_d1 = knots[m+1+d]
            left_num = (xv - t_m)
            left_den = (t_m_d - t_m) if (t_m_d - t_m) != 0 else 1.0
            right_num = (knots[min(m+1, len(knots)-1)] + 0.0)  # placeholder to keep shape
            # Standard recurrence:
            # N_{m,d}(x) = ((x - t_m)/(t_{m+d} - t_m)) * N_{m,d-1}(x) + ((t_{m+d+1} - x)/(t_{m+d+1} - t_{m+1})) * N_{m+1,d-1}(x)
            term1 = 0.0
            if (t_m_d - t_m) != 0:
                term1 = ((xv - t_m)/ (t_m_d - t_m)) * phi[:, :, m]
            term2 = 0.0
            if m+1 < M and (knots[m+d+1] - knots[m+1]) != 0:
                term2 = ((knots[m+d+1] - xv) / (knots[m+d+1] - knots[m+1])) * phi[:, :, m+1]
            phi_next[:, :, m] = term1 + term2
        phi = phi_next
    return phi