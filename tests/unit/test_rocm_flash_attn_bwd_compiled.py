"""The COMPILER-GENERATED RDNA WMMA flash-attention BACKWARD executes on gfx1151.

The backward analog of `test_rocm_flash_attn_compiled.py`: a single
`tessera_rocm.flash_attn_bwd` directive is expanded by
`generate-wmma-flash-attn-bwd-kernel` into THREE fragment-materialized kernels —
`fa_pre` (scalar logsumexp L + D=rowsum(O*dO)), `fa_dkdv` (per key-tile WMMA dK/dV),
`fa_dq` (per query-tile WMMA dQ) — all lowered through Stage J (real `rocdl.wmma`)
+ Stage I (→ hsaco, in tessera-opt, no mlir-opt) and executed in sequence:

    "tessera_rocm.flash_attn_bwd"{head_dim=D}
      --(generate-wmma-flash-attn-bwd-kernel, lower-tessera-target-to-rocdl,
         convert-gpu-to-rocdl, rocdl-attach-target{gfx1151}, gpu-module-to-binary)
      --(launch fa_pre -> fa_dkdv -> fa_dq)--> dQ, dK, dV

Compared to a numpy attention-backward reference (itself checked vs finite
differences in scratch) across head_dim 16/64, causal + non-causal, ragged.
The third compiler-generated op on ROCm after matmul + flash_attn forward.

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


def _directive(head_dim, dtype="f16"):
    return (
        'module {\n'
        '  "tessera_rocm.flash_attn_bwd"() {name = "fa", '
        f'head_dim = {head_dim} : i64, dtype = "{dtype}"}} : () -> ()\n'
        '}\n'
    )


def _pipeline():
    return (
        "builtin.module("
        "generate-wmma-flash-attn-bwd-kernel,"
        "lower-tessera-target-to-rocdl,"
        "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
        "reconcile-unrealized-casts),"
        f"rocdl-attach-target{{chip={CHIP}}},"
        "gpu-module-to-binary)"
    )


def _extract_hsaco(text: str) -> bytes:
    i = text.index('bin = "') + len('bin = "')
    out = bytearray()
    j = i
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


def _build_hsaco(head_dim: int) -> bytes:
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    r = subprocess.run([str(TESSERA_OPT), "-", f"--pass-pipeline={_pipeline()}"],
                       input=_directive(head_dim), capture_output=True, text=True)
    if r.returncode != 0 or "gpu.binary" not in r.stdout:
        pytest.skip(f"flash_attn_bwd serialize unavailable (rc={r.returncode}): "
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


def _fwd_bwd_ref(Q, K, V, dO, scale, causal):
    """numpy attention fwd (for O) + backward reference, f32 math from f16."""
    Qf, Kf, Vf = Q.astype(np.float32), K.astype(np.float32), V.astype(np.float32)
    dOf = dO.astype(np.float32)
    S = scale * np.einsum("bhqd,bhkd->bhqk", Qf, Kf)
    B, H, Sq, Sk = S.shape
    if causal:
        i = np.arange(Sq)[:, None]; j = np.arange(Sk)[None, :]
        S = np.where((j > i)[None, None], -1e30, S)
    S = S - S.max(axis=-1, keepdims=True)
    P = np.exp(S)
    P = P / P.sum(axis=-1, keepdims=True)
    O = np.einsum("bhqk,bhkd->bhqd", P, Vf).astype(np.float32)
    # backward
    dV = np.einsum("bhqk,bhqd->bhkd", P, dOf)
    dP = np.einsum("bhqd,bhkd->bhqk", dOf, Vf)
    Dq = np.sum(O * dOf, axis=-1, keepdims=True)
    dS = P * (dP - Dq)
    dQ = scale * np.einsum("bhqk,bhkd->bhqd", dS, Kf)
    dK = scale * np.einsum("bhqk,bhqd->bhkd", dS, Qf)
    return O, dQ, dK, dV


def _mr(p, n):
    return [ctypes.c_void_p(p.value), ctypes.c_void_p(p.value),
            ctypes.c_int64(0), ctypes.c_int64(n), ctypes.c_int64(1)]


def _arr(args):
    a = (ctypes.c_void_p * len(args))()
    for i, v in enumerate(args):
        a[i] = ctypes.cast(ctypes.byref(v), ctypes.c_void_p)
    return a


@pytest.mark.parametrize("D,B,H,Sq,Sk,causal", [
    (16, 1, 1, 16, 16, 0),
    (16, 1, 2, 32, 48, 0),
    (64, 2, 2, 48, 48, 1),     # causal
    (16, 1, 1, 20, 40, 0),     # ragged Sq/Sk
    (64, 1, 2, 32, 32, 1),     # causal, D=64
])
def test_compiled_flash_attn_bwd_matches_numpy(D, B, H, Sq, Sk, causal):
    hsaco = _build_hsaco(D)
    hip = _hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable — no ROCm host")
    if hip.hipInit(0) != 0:
        pytest.skip("hipInit failed")
    mod = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(mod), hsaco) != 0:
        pytest.skip("no usable AMD GPU (module load failed)")
    fns = {}
    for nm in (b"fa_pre", b"fa_dkdv", b"fa_dq"):
        fn = ctypes.c_void_p()
        assert hip.hipModuleGetFunction(ctypes.byref(fn), mod, nm) == 0
        fns[nm] = fn

    rng = np.random.default_rng(11 + D + Sq + Sk + causal)
    Q = (rng.standard_normal((B, H, Sq, D)) * 0.3).astype(np.float16)
    K = (rng.standard_normal((B, H, Sk, D)) * 0.3).astype(np.float16)
    Vv = (rng.standard_normal((B, H, Sk, D)) * 0.3).astype(np.float16)
    dOv = (rng.standard_normal((B, H, Sq, D)) * 0.3).astype(np.float16)
    scale = 1.0 / np.sqrt(D)
    O_ref, dQ_ref, dK_ref, dV_ref = _fwd_bwd_ref(Q, K, Vv, dOv, scale, causal)

    nQ, nKV, nL = B * H * Sq * D, B * H * Sk * D, B * H * Sq
    # device buffers
    bufs = {}
    for name, nb in (("Q", 2 * nQ), ("K", 2 * nKV), ("V", 2 * nKV),
                     ("dO", 2 * nQ), ("O", 4 * nQ), ("L", 4 * nL), ("Dd", 4 * nL),
                     ("dQ", 4 * nQ), ("dK", 4 * nKV), ("dV", 4 * nKV)):
        d = ctypes.c_void_p()
        if hip.hipMalloc(ctypes.byref(d), nb) != 0:
            pytest.skip("hipMalloc failed")
        bufs[name] = d
    O_f32 = O_ref.astype(np.float32)
    hip.hipMemcpy(bufs["Q"], Q.ctypes.data_as(ctypes.c_void_p), 2 * nQ, 1)
    hip.hipMemcpy(bufs["K"], K.ctypes.data_as(ctypes.c_void_p), 2 * nKV, 1)
    hip.hipMemcpy(bufs["V"], Vv.ctypes.data_as(ctypes.c_void_p), 2 * nKV, 1)
    hip.hipMemcpy(bufs["dO"], dOv.ctypes.data_as(ctypes.c_void_p), 2 * nQ, 1)
    hip.hipMemcpy(bufs["O"], O_f32.ctypes.data_as(ctypes.c_void_p), 4 * nQ, 1)

    Sqc, Skc = ctypes.c_int64(Sq), ctypes.c_int64(Sk)
    sc, cau = ctypes.c_float(scale), ctypes.c_int64(causal)
    launch = hip.hipModuleLaunchKernel
    launch.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint] * 6
                       + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p])
    gyt = B * H
    gqt, gkt = (Sq + 15) // 16, (Sk + 15) // 16

    # fa_pre : (Q,K,dO, O,L,Dd, Sq,Sk, scale, causal)
    pre_args = (_mr(bufs["Q"], nQ) + _mr(bufs["K"], nKV) + _mr(bufs["dO"], nQ)
                + _mr(bufs["O"], nQ) + _mr(bufs["L"], nL) + _mr(bufs["Dd"], nL)
                + [Sqc, Skc, sc, cau])
    assert launch(fns[b"fa_pre"], gqt, gyt, 1, 32, 1, 1, 0, None,
                  _arr(pre_args), None) == 0
    # fa_dkdv : (Q,K,V,dO, L,Dd, dK,dV, Sq,Sk, scale, causal)
    dkdv_args = (_mr(bufs["Q"], nQ) + _mr(bufs["K"], nKV) + _mr(bufs["V"], nKV)
                 + _mr(bufs["dO"], nQ) + _mr(bufs["L"], nL) + _mr(bufs["Dd"], nL)
                 + _mr(bufs["dK"], nKV) + _mr(bufs["dV"], nKV)
                 + [Sqc, Skc, sc, cau])
    assert launch(fns[b"fa_dkdv"], gkt, gyt, 1, 32, 1, 1, 0, None,
                  _arr(dkdv_args), None) == 0
    # fa_dq : (Q,K,V,dO, L,Dd, dQ, Sq,Sk, scale, causal)
    dq_args = (_mr(bufs["Q"], nQ) + _mr(bufs["K"], nKV) + _mr(bufs["V"], nKV)
               + _mr(bufs["dO"], nQ) + _mr(bufs["L"], nL) + _mr(bufs["Dd"], nL)
               + _mr(bufs["dQ"], nQ) + [Sqc, Skc, sc, cau])
    assert launch(fns[b"fa_dq"], gqt, gyt, 1, 32, 1, 1, 0, None,
                  _arr(dq_args), None) == 0
    assert hip.hipDeviceSynchronize() == 0

    dQ = np.zeros((B, H, Sq, D), np.float32)
    dK = np.zeros((B, H, Sk, D), np.float32)
    dV = np.zeros((B, H, Sk, D), np.float32)
    hip.hipMemcpy(dQ.ctypes.data_as(ctypes.c_void_p), bufs["dQ"], 4 * nQ, 2)
    hip.hipMemcpy(dK.ctypes.data_as(ctypes.c_void_p), bufs["dK"], 4 * nKV, 2)
    hip.hipMemcpy(dV.ctypes.data_as(ctypes.c_void_p), bufs["dV"], 4 * nKV, 2)
    for d in bufs.values():
        hip.hipFree(d)

    def err(got, ref):
        return float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-6))

    eQ, eK, eV = err(dQ, dQ_ref), err(dK, dK_ref), err(dV, dV_ref)
    # Measured rel-err on gfx1151 is ~2-4e-4 (f16 storage, f32 accumulate); the
    # 5e-3 bound is regression-sensitive while robust across RNG seeds.
    tol = 5e-3
    assert eQ < tol and eK < tol and eV < tol, (
        f"compiled flash_attn_bwd rel-err dQ={eQ:.3e} dK={eK:.3e} dV={eV:.3e} "
        f"at D={D} {B}x{H}x{Sq}x{Sk} causal={causal}")


@pytest.mark.parametrize("causal,kv_heads", [
    (False, 2), (True, 2), (False, 1), (True, 1),
])
def test_g6c_split_reduced_candidate_matches_one_wave(causal, kv_heads):
    from tessera import runtime as rt

    if not TESSERA_OPT.is_file() or _hip() is None:
        pytest.skip("ROCm compiler/device unavailable")
    rng = np.random.default_rng(606 + causal)
    shape = (1, 2, 33, 64)
    q = (rng.standard_normal(shape) * 0.2).astype(np.float16)
    kv_shape = (shape[0], kv_heads, shape[2], shape[3])
    k = (rng.standard_normal(kv_shape) * 0.2).astype(np.float16)
    v = (rng.standard_normal(kv_shape) * 0.2).astype(np.float16)
    do = (rng.standard_normal(shape) * 0.2).astype(np.float16)
    artifact = rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_flash_attn_bwd_compiled",
        "executable": True, "arg_names": ["do", "q", "k", "v"],
        "ops": [{"op_name": "tessera.flash_attn_bwd",
                 "operands": ["do", "q", "k", "v"],
                 "kwargs": {"causal": causal}}],
    })
    baseline = rt._execute_rocm_compiled_flash_attn_bwd(
        artifact, (do, q, k, v))
    candidate = rt._execute_rocm_compiled_flash_attn_bwd(
        artifact, (do, q, k, v), _split_reduced=True)
    for got, expected in zip(candidate, baseline, strict=True):
        np.testing.assert_allclose(got, expected, rtol=3e-3, atol=3e-4)
