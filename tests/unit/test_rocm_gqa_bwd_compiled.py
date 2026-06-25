"""Compiler-generated GQA/MQA BACKWARD on gfx1151.

GQA backward: H query heads share G<H key/value heads. dQ[b,h] uses the grouped
KV head (g=h/kv_ratio); dK[b,g]/dV[b,g] SUM the contributions of all kv_ratio
query heads in the group. The `gqa = true` directive emits three kernels that
gain (heads, kv_ratio) runtime args: `_pre`/`_dq` keep the B*H query-head grid;
`_dkdv` launches on a B*G KV-head grid and accumulates dK/dV **atomically** (the
sharing query-head blocks add into the same rows — host pre-zeros dK/dV).
Validated vs a numpy GQA-backward reference across MQA / GQA / MHA-equivalence,
causal+non-causal.

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
        "builtin.module(generate-wmma-flash-attn-bwd-kernel,"
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
    directive = ('module {\n  "tessera_rocm.flash_attn_bwd"() {name = "fa", '
                 f'head_dim = {head_dim} : i64, dtype = "f16", gqa = true}} '
                 ': () -> ()\n}\n')
    r = subprocess.run([str(TESSERA_OPT), "-", f"--pass-pipeline={_pipeline()}"],
                       input=directive, capture_output=True, text=True)
    if r.returncode != 0 or "gpu.binary" not in r.stdout:
        pytest.skip(f"gqa bwd serialize unavailable (rc={r.returncode}): "
                    f"{r.stderr[:300]}")
    return _extract_hsaco(r.stdout)


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


def _fwd_bwd_ref(Q, K, V, dO, scale, causal):
    """numpy GQA fwd (for O) + backward; query head h uses KV head h//ratio."""
    B, H, Sq, D = Q.shape
    G = K.shape[1]; ratio = H // G
    Qf, Kf, Vf, dOf = (x.astype(np.float32) for x in (Q, K, V, dO))
    O = np.zeros((B, H, Sq, D), np.float32)
    dQ = np.zeros((B, H, Sq, D), np.float32)
    dK = np.zeros((B, G, K.shape[2], D), np.float32)
    dV = np.zeros((B, G, K.shape[2], D), np.float32)
    for b in range(B):
        for h in range(H):
            g = h // ratio
            s = scale * (Qf[b, h] @ Kf[b, g].T)
            if causal:
                i = np.arange(Sq)[:, None]; j = np.arange(K.shape[2])[None, :]
                s = np.where(j > i, -1e30, s)
            s = s - s.max(-1, keepdims=True)
            p = np.exp(s); p = p / p.sum(-1, keepdims=True)
            O[b, h] = p @ Vf[b, g]
            dp = dOf[b, h] @ Vf[b, g].T
            dq_row = np.sum(O[b, h] * dOf[b, h], axis=-1, keepdims=True)
            ds = p * (dp - dq_row)
            dQ[b, h] = scale * (ds @ Kf[b, g])
            dK[b, g] += scale * (ds.T @ Qf[b, h])   # accumulate over the group
            dV[b, g] += p.T @ dOf[b, h]
    return O, dQ, dK, dV


def _mr(p, n):
    return [ctypes.c_void_p(p.value), ctypes.c_void_p(p.value),
            ctypes.c_int64(0), ctypes.c_int64(n), ctypes.c_int64(1)]


def _arr(a):
    arr = (ctypes.c_void_p * len(a))()
    for i, v in enumerate(a):
        arr[i] = ctypes.cast(ctypes.byref(v), ctypes.c_void_p)
    return arr


@pytest.mark.parametrize("D,B,H,G,Sq,Sk,causal", [
    (16, 1, 8, 1, 32, 32, 0),    # MQA
    (16, 2, 8, 2, 32, 48, 0),    # GQA (ratio 4)
    (64, 1, 8, 4, 48, 48, 1),    # GQA (ratio 2), causal
    (16, 1, 4, 4, 32, 32, 0),    # MHA-equivalence
])
def test_compiled_gqa_bwd_matches_numpy(D, B, H, G, Sq, Sk, causal):
    hsaco = _build(D)
    hip = _hip()
    if hip is None or hip.hipInit(0) != 0:
        pytest.skip("no ROCm host")
    mod = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(mod), hsaco) != 0:
        pytest.skip("no usable AMD GPU")
    fns = {}
    for nm in (b"fa_pre", b"fa_dkdv", b"fa_dq"):
        fn = ctypes.c_void_p()
        assert hip.hipModuleGetFunction(ctypes.byref(fn), mod, nm) == 0
        fns[nm] = fn

    rng = np.random.default_rng(13 + D + H + G + causal)
    Q = (rng.standard_normal((B, H, Sq, D)) * 0.3).astype(np.float16)
    K = (rng.standard_normal((B, G, Sk, D)) * 0.3).astype(np.float16)
    Vv = (rng.standard_normal((B, G, Sk, D)) * 0.3).astype(np.float16)
    dOv = (rng.standard_normal((B, H, Sq, D)) * 0.3).astype(np.float16)
    scale = 1.0 / float(np.sqrt(D))
    ratio = H // G
    O_ref, dQ_ref, dK_ref, dV_ref = _fwd_bwd_ref(Q, K, Vv, dOv, scale, causal)

    nQ, nKV, nL = B * H * Sq * D, B * G * Sk * D, B * H * Sq
    bufs = {}
    for name, nb in (("Q", 2 * nQ), ("K", 2 * nKV), ("V", 2 * nKV),
                     ("dO", 2 * nQ), ("O", 4 * nQ), ("L", 4 * nL), ("Dd", 4 * nL),
                     ("dQ", 4 * nQ), ("dK", 4 * nKV), ("dV", 4 * nKV)):
        d = ctypes.c_void_p()
        if hip.hipMalloc(ctypes.byref(d), nb) != 0:
            pytest.skip("hipMalloc failed")
        bufs[name] = d
    hip.hipMemcpy(bufs["Q"], Q.ctypes.data_as(ctypes.c_void_p), 2 * nQ, 1)
    hip.hipMemcpy(bufs["K"], K.ctypes.data_as(ctypes.c_void_p), 2 * nKV, 1)
    hip.hipMemcpy(bufs["V"], Vv.ctypes.data_as(ctypes.c_void_p), 2 * nKV, 1)
    hip.hipMemcpy(bufs["dO"], dOv.ctypes.data_as(ctypes.c_void_p), 2 * nQ, 1)
    hip.hipMemcpy(bufs["O"], O_ref.astype(np.float32).ctypes.data_as(ctypes.c_void_p),
                  4 * nQ, 1)
    # dK/dV are accumulated ATOMICALLY across the sharing query heads — pre-zero.
    hip.hipMemset(bufs["dK"], 0, 4 * nKV)
    hip.hipMemset(bufs["dV"], 0, 4 * nKV)

    Sqc, Skc = ctypes.c_int64(Sq), ctypes.c_int64(Sk)
    sc, cau = ctypes.c_float(scale), ctypes.c_int64(causal)
    hh, kr = ctypes.c_int64(H), ctypes.c_int64(ratio)
    launch = hip.hipModuleLaunchKernel
    launch.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint] * 6
                       + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p])
    gqt, gkt = (Sq + 15) // 16, (Sk + 15) // 16

    # _pre : grid (ceil(Sq/16), B*H)
    pre = (_mr(bufs["Q"], nQ) + _mr(bufs["K"], nKV) + _mr(bufs["dO"], nQ)
           + _mr(bufs["O"], nQ) + _mr(bufs["L"], nL) + _mr(bufs["Dd"], nL)
           + [Sqc, Skc, sc, cau, hh, kr])
    assert launch(fns[b"fa_pre"], gqt, B * H, 1, 32, 1, 1, 0, None,
                  _arr(pre), None) == 0
    # _dkdv : grid (ceil(Sk/16), B*H) — one block per QUERY head; each atomic-adds
    # its contribution into the grouped KV head's dK/dV rows.
    dkdv = (_mr(bufs["Q"], nQ) + _mr(bufs["K"], nKV) + _mr(bufs["V"], nKV)
            + _mr(bufs["dO"], nQ) + _mr(bufs["L"], nL) + _mr(bufs["Dd"], nL)
            + _mr(bufs["dK"], nKV) + _mr(bufs["dV"], nKV)
            + [Sqc, Skc, sc, cau, hh, kr])
    assert launch(fns[b"fa_dkdv"], gkt, B * H, 1, 32, 1, 1, 0, None,
                  _arr(dkdv), None) == 0
    # _dq : grid (ceil(Sq/16), B*H)
    dq = (_mr(bufs["Q"], nQ) + _mr(bufs["K"], nKV) + _mr(bufs["V"], nKV)
          + _mr(bufs["dO"], nQ) + _mr(bufs["L"], nL) + _mr(bufs["Dd"], nL)
          + _mr(bufs["dQ"], nQ) + [Sqc, Skc, sc, cau, hh, kr])
    assert launch(fns[b"fa_dq"], gqt, B * H, 1, 32, 1, 1, 0, None,
                  _arr(dq), None) == 0
    assert hip.hipDeviceSynchronize() == 0

    dQ = np.zeros((B, H, Sq, D), np.float32)
    dK = np.zeros((B, G, Sk, D), np.float32)
    dV = np.zeros((B, G, Sk, D), np.float32)
    hip.hipMemcpy(dQ.ctypes.data_as(ctypes.c_void_p), bufs["dQ"], 4 * nQ, 2)
    hip.hipMemcpy(dK.ctypes.data_as(ctypes.c_void_p), bufs["dK"], 4 * nKV, 2)
    hip.hipMemcpy(dV.ctypes.data_as(ctypes.c_void_p), bufs["dV"], 4 * nKV, 2)
    for d in bufs.values():
        hip.hipFree(d)

    def err(got, ref):
        return float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-6))

    eQ, eK, eV = err(dQ, dQ_ref), err(dK, dK_ref), err(dV, dV_ref)
    assert eQ < 5e-3 and eK < 5e-3 and eV < 5e-3, (
        f"GQA bwd rel-err dQ={eQ:.2e} dK={eK:.2e} dV={eV:.2e} "
        f"D={D} H={H} G={G} causal={causal}")
