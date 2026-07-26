"""The COMPILER-GENERATED RDNA WMMA flash-attention forward executes on gfx1151.

The attention analog of `test_rocm_wmma_gemm_*`: a `tessera_rocm.flash_attn`
directive is expanded by `generate-wmma-flash-attn-kernel` into a fragment-
materialized FA-2 forward kernel (LDS-staged Q, QK^T on WMMA, online softmax,
P@V on WMMA — the kernel BODY is emitted by the pass), which lowers through
Stage J (real `rocdl.wmma`) + Stage I (→ hsaco, all in tessera-opt, no mlir-opt)
and executes:

    "tessera_rocm.flash_attn"{head_dim=D}
      --(tessera-opt: generate-wmma-flash-attn-kernel, lower-tessera-target-to-
         rocdl, finish-lower, rocdl-attach-target{gfx1151}, gpu-module-to-binary)
      --(hipModuleLoadData + launch grid=(ceil(Sq/16), B*H))--> O

Compared to a numpy attention reference (f16 storage, f32 softmax + accumulate)
across head_dim 16/64, causal + non-causal, and ragged Sq/Sk. The second
compiler-generated op on ROCm after matmul.

Skip-clean: tessera-opt / mlir-opt not built, or no usable AMD GPU.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from tessera import runtime as tessera_runtime
from tessera.runtime import RuntimeArtifact

REPO = Path(__file__).resolve().parents[2]
TESSERA_OPT = REPO / "build" / "tools" / "tessera-opt" / "tessera-opt"
CHIP = os.environ.get("TESSERA_ROCM_CHIP", "gfx1151")


def _directive(head_dim, dtype="f16"):
    return (
        'module {\n'
        '  "tessera_rocm.flash_attn"() {name = "fa", '
        f'head_dim = {head_dim} : i64, dtype = "{dtype}"}} : () -> ()\n'
        '}\n'
    )


def _pipeline():
    return (
        "builtin.module("
        "generate-wmma-flash-attn-kernel,"
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
        pytest.skip(f"flash_attn serialize unavailable (rc={r.returncode}): "
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


def _ref(Q, K, V, scale, causal):
    """numpy FA reference, f32 math from the f16 storage values. Shapes
    Q[B,H,Sq,D], K/V[B,H,Sk,D]."""
    Qf = Q.astype(np.float32)
    Kf = K.astype(np.float32)
    Vf = V.astype(np.float32)
    S = scale * np.einsum("bhqd,bhkd->bhqk", Qf, Kf)
    B, H, Sq, Sk = S.shape
    if causal:
        i = np.arange(Sq)[:, None]
        j = np.arange(Sk)[None, :]
        S = np.where(j[None, None] > i[None, None], -1e30, S)
    S = S - S.max(axis=-1, keepdims=True)
    P = np.exp(S)
    P = P / P.sum(axis=-1, keepdims=True)
    return np.einsum("bhqk,bhkd->bhqd", P, Vf).astype(np.float32)


def _mr(p, nbytes_elems):
    return [ctypes.c_void_p(p.value), ctypes.c_void_p(p.value),
            ctypes.c_int64(0), ctypes.c_int64(nbytes_elems), ctypes.c_int64(1)]


@pytest.mark.parametrize("D,B,H,Sq,Sk,causal", [
    (16, 1, 1, 16, 16, 0),
    (16, 1, 2, 32, 48, 0),
    (64, 2, 2, 48, 48, 1),     # causal
    (16, 1, 1, 20, 40, 0),     # ragged Sq/Sk
    (64, 1, 2, 33, 17, 1),     # ragged + causal
])
def test_compiled_flash_attn_matches_numpy(D, B, H, Sq, Sk, causal):
    hsaco = _build_hsaco(D)
    hip = _hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable — no ROCm host")
    if hip.hipInit(0) != 0:
        pytest.skip("hipInit failed")
    mod = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(mod), hsaco) != 0:
        pytest.skip("no usable AMD GPU (module load failed)")
    fn = ctypes.c_void_p()
    assert hip.hipModuleGetFunction(ctypes.byref(fn), mod, b"fa") == 0

    rng = np.random.default_rng(7 + D + Sq + Sk + causal)
    Q = (rng.standard_normal((B, H, Sq, D)) * 0.3).astype(np.float16)
    K = (rng.standard_normal((B, H, Sk, D)) * 0.3).astype(np.float16)
    Vv = (rng.standard_normal((B, H, Sk, D)) * 0.3).astype(np.float16)
    scale = 1.0 / np.sqrt(D)

    dq, dk, dv, do = (ctypes.c_void_p(), ctypes.c_void_p(),
                      ctypes.c_void_p(), ctypes.c_void_p())
    nQ, nKV, nO = B * H * Sq * D, B * H * Sk * D, B * H * Sq * D
    for d, nb in ((dq, 2 * nQ), (dk, 2 * nKV), (dv, 2 * nKV), (do, 4 * nO)):
        if hip.hipMalloc(ctypes.byref(d), nb) != 0:
            pytest.skip("hipMalloc failed")
    hip.hipMemcpy(dq, Q.ctypes.data_as(ctypes.c_void_p), 2 * nQ, 1)
    hip.hipMemcpy(dk, K.ctypes.data_as(ctypes.c_void_p), 2 * nKV, 1)
    hip.hipMemcpy(dv, Vv.ctypes.data_as(ctypes.c_void_p), 2 * nKV, 1)

    args = (_mr(dq, nQ) + _mr(dk, nKV) + _mr(dv, nKV) + _mr(do, nO)
            + [ctypes.c_int64(Sq), ctypes.c_int64(Sk),
               ctypes.c_float(scale), ctypes.c_int64(causal)])
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
    O = np.zeros((B, H, Sq, D), dtype=np.float32)
    hip.hipMemcpy(O.ctypes.data_as(ctypes.c_void_p), do, 4 * nO, 2)
    for d in (dq, dk, dv, do):
        hip.hipFree(d)

    ref = _ref(Q, K, Vv, scale, causal)
    maxerr = float(np.max(np.abs(O - ref)))
    assert maxerr < 2e-2, f"compiled flash_attn maxerr={maxerr} at " \
        f"D={D} {B}x{H}x{Sq}x{Sk} causal={causal}"


def test_optimized_gqa_bias_softcap_window_dropout_matches_counter_oracle():
    """The optimized route composes every forward semantic feature and replays
    the scalar carrier's exact deterministic dropout counter."""
    B, Hq, Hkv, Sq, Sk, D = 1, 4, 2, 17, 19, 64
    scale, cap, window, dropout_p, seed = 0.125, 4.0, 8, 0.125, 17
    rng = np.random.default_rng(20260726)
    q = (rng.standard_normal((B, Hq, Sq, D)) * 0.2).astype(np.float16)
    k = (rng.standard_normal((B, Hkv, Sk, D)) * 0.2).astype(np.float16)
    v = (rng.standard_normal((B, Hkv, Sk, D)) * 0.2).astype(np.float16)
    bias = (rng.standard_normal((B, Hq, Sq, Sk)) * 0.05).astype(np.float32)

    artifact = RuntimeArtifact(
        metadata={
            "target": "rocm",
            "compiler_path": "rocm_flash_attn_compiled",
            "executable": True,
            "execution_kind": "native_gpu",
            "arg_names": ["q", "k", "v", "bias"],
            "output_name": "o",
            "ops": [
                {
                    "op_name": "tessera.flash_attn",
                    "result": "o",
                    "operands": ["q", "k", "v", "bias"],
                    "kwargs": {
                        "scale": scale,
                        "causal": True,
                        "window": window,
                        "logit_softcap": cap,
                        "dropout_p": dropout_p,
                        "dropout_seed": seed,
                    },
                }
            ],
        }
    )
    try:
        actual = tessera_runtime._execute_rocm_compiled_flash_attn(
            artifact, (q, k, v, bias)
        )
    except tessera_runtime._RocmCompiledUnavailable as exc:
        pytest.skip(str(exc))

    kq = np.repeat(k.astype(np.float32), Hq // Hkv, axis=1)
    vq = np.repeat(v.astype(np.float32), Hq // Hkv, axis=1)
    scores = scale * np.einsum(
        "bhqd,bhkd->bhqk", q.astype(np.float32), kq
    )
    scores = cap * np.tanh(scores / cap) + bias
    qpos = np.arange(Sq)[:, None]
    kpos = np.arange(Sk)[None, :]
    masked = (kpos > qpos) | ((qpos - kpos) >= window)
    scores = np.where(masked[None, None], -1e30, scores)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    bh = np.arange(B * Hq, dtype=np.uint64)[:, None, None]
    qindex = np.arange(Sq, dtype=np.uint64)[None, :, None]
    kindex = np.arange(Sk, dtype=np.uint64)[None, None, :]
    counter = (bh * Sq + qindex) * Sk + kindex
    hashed = (
        counter * np.uint64(1664525)
        + np.uint64(seed + 1013904223)
    ) & np.uint64(0xFFFFFFFF)
    keep = hashed >= np.uint64(int(dropout_p * 4294967296.0))
    replay = keep.reshape(B, Hq, Sq, Sk).astype(np.float32)
    replay *= np.float32(1.0 / (1.0 - dropout_p))
    expected = np.einsum("bhqk,bhkd->bhqd", probs * replay, vq)
    maxerr = float(np.max(np.abs(actual - expected)))
    assert maxerr < 2e-2, f"optimized feature combination maxerr={maxerr}"
