"""Execute-compare fixture for the shipped ROCm WMMA flash-attention symbol.

This is the numerical proof behind the `backend_manifest` `hardware_verified`
row for `tessera.flash_attn` on `rocm` (gfx1151 / RDNA 3.5): it dlopens the
**shipped** `libtessera_rocm_flash_attn.so`, calls the C-ABI symbols
`tessera_rocm_wmma_flash_attn_{f16,bf16}` (which HIPRTC-compile the RDNA WMMA
flash-attention kernel for the device arch and launch it), and compares the GPU
output to a numpy attention reference.

flash_attn is the **second** op after matmul to execute natively on a non-Apple
backend. Both the QK^T scores and the P@V output run on 16x16x16 WMMA with an
online (FA-2) softmax; causal masking and ragged Sq/Sk are exercised here.

Skip-clean: lib not built / no usable GPU (the symbol returns rc=2).
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[2]
ATTN_LIB = (REPO_ROOT / "build" / "src" / "compiler" / "codegen"
            / "Tessera_ROCM_Backend" / "runtime" / "hip"
            / "libtessera_rocm_flash_attn.so")
ROCM_LIB_DIR = os.environ.get("ROCM_PATH", "/opt/rocm") + "/lib"


def _bf16():
    ml = pytest.importorskip("ml_dtypes")
    return ml.bfloat16


def _load_lib():
    if not ATTN_LIB.is_file():
        pytest.skip(f"build the shipped flash-attn lib: ninja -C build "
                    f"tessera_rocm_flash_attn ({ATTN_LIB} missing)")
    for dep in ("libamdhip64.so", "libhiprtc.so"):
        p = os.path.join(ROCM_LIB_DIR, dep)
        if os.path.isfile(p):
            try:
                ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
    return ctypes.CDLL(str(ATTN_LIB), mode=ctypes.RTLD_GLOBAL)


def _bind(lib, name):
    fn = getattr(lib, name)
    # (Q, K, V, O, B, H, Sq, Sk, D, scale, causal)
    fn.argtypes = ([ctypes.c_void_p] * 4 + [ctypes.c_int] * 4
                   + [ctypes.c_int, ctypes.c_float, ctypes.c_int])
    fn.restype = ctypes.c_int
    return fn


def _ref_attention(Q, K, V, scale, causal):
    """Plain numpy attention reference over [B, H, S, D] float32 tensors."""
    Sq, Sk = Q.shape[2], K.shape[2]
    scores = np.einsum("bhqd,bhkd->bhqk", Q, K) * scale
    if causal:
        q = np.arange(Sq)[:, None]
        k = np.arange(Sk)[None, :]
        scores = np.where(q >= k, scores, -1e30)
    scores = scores - scores.max(-1, keepdims=True)
    p = np.exp(scores)
    p = p / p.sum(-1, keepdims=True)
    return np.einsum("bhqk,bhkd->bhqd", p, V)


def _run(fn, store, B, H, Sq, Sk, D, causal):
    rng = np.random.default_rng(0)
    Q = (rng.standard_normal((B, H, Sq, D)) * 0.5).astype(store)
    K = (rng.standard_normal((B, H, Sk, D)) * 0.5).astype(store)
    V = (rng.standard_normal((B, H, Sk, D)) * 0.5).astype(store)
    O = np.zeros((B, H, Sq, D), dtype=np.float32)
    scale = 1.0 / float(np.sqrt(D))
    rc = fn(Q.ctypes.data_as(ctypes.c_void_p),
            K.ctypes.data_as(ctypes.c_void_p),
            V.ctypes.data_as(ctypes.c_void_p),
            O.ctypes.data_as(ctypes.c_void_p),
            B, H, Sq, Sk, D, ctypes.c_float(scale), 1 if causal else 0)
    return rc, Q, K, V, O, scale


# Forward attention over both storage dtypes — head_dim 16/32/64/128, multi
# batch/head, ragged Sq/Sk (non-multiple-of-16: zero-pad load + -inf score mask
# + bounds-checked store), and causal masking.
@pytest.mark.parametrize("shape,causal", [
    ((1, 1, 16, 16, 16), False),
    ((1, 1, 16, 16, 64), False),
    ((1, 2, 32, 32, 64), False),
    ((2, 3, 48, 80, 128), False),     # multi b/h, Sk>Sq, D=128
    ((1, 1, 16, 16, 16), True),
    ((1, 1, 64, 64, 64), True),
    ((1, 2, 33, 33, 64), True),       # ragged + causal
    ((2, 2, 40, 72, 32), True),       # ragged Sq/Sk + causal
])
def test_shipped_rocm_flash_attn_f16_matches_numpy(shape, causal):
    fn = _bind(_load_lib(), "tessera_rocm_wmma_flash_attn_f16")
    rc, Q, K, V, O, scale = _run(fn, np.float16, *shape, causal)
    if rc == 2:
        pytest.skip("no usable AMD GPU / HIPRTC (shipped symbol returned rc=2)")
    assert rc == 0, f"tessera_rocm_wmma_flash_attn_f16{shape} returned {rc}"
    ref = _ref_attention(Q.astype(np.float32), K.astype(np.float32),
                         V.astype(np.float32), scale, causal)
    maxerr = float(np.max(np.abs(O - ref)))
    assert maxerr < 2e-2, f"f16 flash_attn{shape} causal={causal} maxerr={maxerr}"


@pytest.mark.parametrize("shape,causal", [
    ((1, 1, 16, 16, 64), False),
    ((2, 2, 48, 48, 64), False),
    ((1, 2, 40, 72, 32), True),       # ragged + causal
])
def test_shipped_rocm_flash_attn_bf16_matches_numpy(shape, causal):
    bf16 = _bf16()
    fn = _bind(_load_lib(), "tessera_rocm_wmma_flash_attn_bf16")
    rc, Q, K, V, O, scale = _run(fn, bf16, *shape, causal)
    if rc == 2:
        pytest.skip("no usable AMD GPU / HIPRTC (shipped symbol returned rc=2)")
    assert rc == 0, f"tessera_rocm_wmma_flash_attn_bf16{shape} returned {rc}"
    ref = _ref_attention(Q.astype(np.float32), K.astype(np.float32),
                         V.astype(np.float32), scale, causal)
    maxerr = float(np.max(np.abs(O - ref)))
    # bf16 has ~8 mantissa bits — looser tolerance than f16.
    assert maxerr < 6e-2, f"bf16 flash_attn{shape} causal={causal} maxerr={maxerr}"


def test_shipped_rocm_flash_attn_rejects_bad_shape():
    fn = _bind(_load_lib(), "tessera_rocm_wmma_flash_attn_f16")
    q = np.zeros((1, 1, 16, 17), dtype=np.float16)   # head_dim 17 not mult of 16
    o = np.zeros((1, 1, 16, 17), dtype=np.float32)
    # head_dim not a multiple of 16 -> rc=1, no device needed (validated first).
    rc = fn(q.ctypes.data_as(ctypes.c_void_p), q.ctypes.data_as(ctypes.c_void_p),
            q.ctypes.data_as(ctypes.c_void_p), o.ctypes.data_as(ctypes.c_void_p),
            1, 1, 16, 16, 17, ctypes.c_float(0.25), 0)
    assert rc == 1, f"expected rc=1 (bad head_dim), got {rc}"
