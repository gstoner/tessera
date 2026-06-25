"""Compiler-generated grouped-query / multi-query attention (GQA/MQA) on gfx1151.

GQA is flash_attn where H query heads share G<H key/value heads: query head h
reads KV head h/(H/G). The `generate-wmma-flash-attn-kernel` pass, with the
directive's `gqa = true`, emits the same FA-2 forward WMMA kernel plus two
runtime args (heads H, kv_ratio = H/G) and reads K/V from the grouped head
(b*G + h/kv_ratio). kv_ratio=1 is plain MHA (equivalence checked); kv_ratio=H is
MQA. Validated vs a numpy GQA reference (f16 storage, f32 softmax + accumulate)
across MQA / GQA / MHA-equivalence and causal/non-causal. Q is [B,H,Sq,D];
K/V are [B,G,Sk,D]; O is [B,H,Sq,D] f32.

Skip-clean: tessera-opt not built, or no usable AMD GPU.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[2]
TESSERA_OPT = REPO / "build" / "tools" / "tessera-opt" / "tessera-opt"
CHIP = os.environ.get("TESSERA_ROCM_CHIP", "gfx1151")


def _pipeline():
    return (
        "builtin.module(generate-wmma-flash-attn-kernel,"
        "lower-tessera-target-to-rocdl,"
        "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
        f"reconcile-unrealized-casts),rocdl-attach-target{{chip={CHIP}}},"
        "gpu-module-to-binary)")


def _extract_hsaco(text):
    i = text.index('bin = "') + len('bin = "')
    out = bytearray(); j = i
    hexd = "0123456789abcdefABCDEF"
    simple = {"\\": 0x5C, '"': 0x22, "n": 0x0A, "t": 0x09, "r": 0x0D}
    while j < len(text):
        c = text[j]
        if c == '"':
            break
        if c == "\\":
            nx = text[j + 1:j + 3]
            if len(nx) == 2 and nx[0] in hexd and nx[1] in hexd:
                out.append(int(nx, 16)); j += 3; continue
            if text[j + 1] in simple:
                out.append(simple[text[j + 1]]); j += 2; continue
        out.append(ord(c)); j += 1
    return bytes(out)


def _build(head_dim):
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    directive = ('module {\n  "tessera_rocm.flash_attn"() {name = "fa", '
                 f'head_dim = {head_dim} : i64, dtype = "f16", gqa = true}} '
                 ': () -> ()\n}\n')
    r = subprocess.run([str(TESSERA_OPT), "-", f"--pass-pipeline={_pipeline()}"],
                       input=directive, capture_output=True, text=True)
    if r.returncode != 0 or "gpu.binary" not in r.stdout:
        pytest.skip(f"gqa serialize unavailable (rc={r.returncode}): "
                    f"{r.stderr[:300]}")
    hsaco = _extract_hsaco(r.stdout)
    assert hsaco[:4] == b"\x7fELF"
    return hsaco


def _hip():
    rocm_lib = os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "lib")
    for dep in ("libamdhip64.so", "libhiprtc.so"):
        p = os.path.join(rocm_lib, dep)
        if os.path.isfile(p):
            try:
                ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
    try:
        return ctypes.CDLL("libamdhip64.so", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return None


def _gqa_ref(Q, K, V, scale, causal):
    B, H, Sq, D = Q.shape
    G = K.shape[1]; ratio = H // G
    O = np.zeros((B, H, Sq, D), np.float32)
    for b in range(B):
        for h in range(H):
            g = h // ratio
            s = scale * (Q[b, h].astype(np.float32) @ K[b, g].astype(np.float32).T)
            if causal:
                i = np.arange(Sq)[:, None]; j = np.arange(K.shape[2])[None, :]
                s = np.where(j > i, -1e30, s)
            s = s - s.max(-1, keepdims=True)
            p = np.exp(s); p = p / p.sum(-1, keepdims=True)
            O[b, h] = p @ V[b, g].astype(np.float32)
    return O


def _mr(p, n):
    return [ctypes.c_void_p(p.value), ctypes.c_void_p(p.value),
            ctypes.c_int64(0), ctypes.c_int64(n), ctypes.c_int64(1)]


@pytest.mark.parametrize("D,B,H,G,Sq,Sk,causal", [
    (16, 1, 8, 1, 32, 32, 0),    # MQA (G=1)
    (16, 2, 8, 2, 32, 48, 0),    # GQA (ratio 4)
    (64, 1, 8, 4, 48, 48, 1),    # GQA (ratio 2), causal
    (16, 1, 4, 4, 32, 32, 0),    # MHA equivalence (ratio 1)
])
def test_compiled_gqa_matches_numpy(D, B, H, G, Sq, Sk, causal):
    hsaco = _build(D)
    hip = _hip()
    if hip is None or hip.hipInit(0) != 0:
        pytest.skip("no ROCm host")
    mod = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(mod), hsaco) != 0:
        pytest.skip("no usable AMD GPU")
    fn = ctypes.c_void_p()
    assert hip.hipModuleGetFunction(ctypes.byref(fn), mod, b"fa") == 0

    rng = np.random.default_rng(5 + D + H + G + Sq + causal)
    q = (rng.standard_normal((B, H, Sq, D)) * 0.3).astype(np.float16)
    k = (rng.standard_normal((B, G, Sk, D)) * 0.3).astype(np.float16)
    v = (rng.standard_normal((B, G, Sk, D)) * 0.3).astype(np.float16)
    scale = 1.0 / float(np.sqrt(D))
    ratio = H // G

    nQ, nKV = B * H * Sq * D, B * G * Sk * D
    dq, dk, dv, do = (ctypes.c_void_p(), ctypes.c_void_p(),
                      ctypes.c_void_p(), ctypes.c_void_p())
    for d, nb in ((dq, 2 * nQ), (dk, 2 * nKV), (dv, 2 * nKV), (do, 4 * nQ)):
        if hip.hipMalloc(ctypes.byref(d), nb) != 0:
            pytest.skip("hipMalloc failed")
    hip.hipMemcpy(dq, q.ctypes.data_as(ctypes.c_void_p), 2 * nQ, 1)
    hip.hipMemcpy(dk, k.ctypes.data_as(ctypes.c_void_p), 2 * nKV, 1)
    hip.hipMemcpy(dv, v.ctypes.data_as(ctypes.c_void_p), 2 * nKV, 1)

    args = (_mr(dq, nQ) + _mr(dk, nKV) + _mr(dv, nKV) + _mr(do, nQ)
            + [ctypes.c_int64(Sq), ctypes.c_int64(Sk), ctypes.c_float(scale),
               ctypes.c_int64(causal), ctypes.c_int64(H), ctypes.c_int64(ratio)])
    arr = (ctypes.c_void_p * len(args))()
    for i, a in enumerate(args):
        arr[i] = ctypes.cast(ctypes.byref(a), ctypes.c_void_p)
    launch = hip.hipModuleLaunchKernel
    launch.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint] * 6
                       + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p])
    gx, gy = (Sq + 15) // 16, B * H
    assert launch(fn, gx, gy, 1, 32, 1, 1, 0, None, arr, None) == 0
    assert hip.hipDeviceSynchronize() == 0
    o = np.zeros((B, H, Sq, D), np.float32)
    hip.hipMemcpy(o.ctypes.data_as(ctypes.c_void_p), do, 4 * nQ, 2)
    for d in (dq, dk, dv, do):
        hip.hipFree(d)

    ref = _gqa_ref(q, k, v, scale, causal)
    maxerr = float(np.max(np.abs(o - ref)))
    assert maxerr < 2e-2, f"GQA maxerr={maxerr} D={D} H={H} G={G} causal={causal}"
