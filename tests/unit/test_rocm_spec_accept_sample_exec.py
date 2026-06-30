"""SD1-2 — tessera.spec_accept_sample (distribution-preserving Leviathan
rejection-sampling acceptance) executes as ONE device kernel on gfx1151.
GenerateROCMSpecAcceptSampleKernel lowers the linear-chain verify to a single
thread: per position accept draft[i] iff accept_u[i]*p_draft <= p_target (the
division-free form of accept_u[i] <= min(1, p_target/p_draft)); on the first
reject draw a corrected token from the residual normalize(relu(target-draft)) by
CDF inversion of resid_u; on full accept draw a bonus from target's extra row.

RNG is explicit (the two uniform streams are op inputs), so the op is a
deterministic, device-bit-exact function. The numpy oracle below mirrors the
kernel's EXACT semantics (the same accept_u*pd<=pt test and the same CDF-inversion
sampler — NOT numpy's rng.choice), so the comparison is bit-exact. Skip-clean
off-GPU / without tessera-opt.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
TESSERA_OPT = Path(
    os.environ.get("TESSERA_OPT_BIN", ROOT / "build/tools/tessera-opt/tessera-opt"))
CHIP = os.environ.get("TESSERA_ROCM_CHIP", "gfx1151")


def _load_hip():
    for name in ("libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _extract_hsaco(s: str) -> bytes:
    marker = 'bin = "'
    j = s.index(marker) + len(marker)
    out = bytearray()
    hexd = "0123456789abcdefABCDEF"
    simple = {"\\": 0x5C, '"': 0x22, "n": 0x0A, "t": 0x09, "r": 0x0D}
    while j < len(s):
        c = s[j]
        if c == '"':
            break
        if c == "\\":
            nxt = s[j + 1:j + 3]
            if len(nxt) == 2 and nxt[0] in hexd and nxt[1] in hexd:
                out.append(int(nxt, 16))
                j += 3
                continue
            if s[j + 1] in simple:
                out.append(simple[s[j + 1]])
                j += 2
                continue
        out.append(ord(c))
        j += 1
    return bytes(out)


def _cdf_invert(u, w):
    """smallest k with cumsum(w)[k] > u·Σw ; argmax(w) if Σw<=0 — the kernel's
    categorical sampler (here residual and bonus share fb=w on the Σw<=0 path,
    which the tests avoid by construction)."""
    s = float(w.sum())
    if s <= 0.0:
        return int(np.argmax(w))
    tgt = u * s
    cum = 0.0
    for k in range(len(w)):
        cum += float(w[k])
        if cum > tgt:
            return k
    return len(w) - 1


def _ref(draft, target_probs, draft_probs, accept_u, resid_u):
    D = draft.shape[0]
    out = np.zeros(D + 2, np.int32)
    ru = float(resid_u[0])
    accepted = 0
    done = False
    for i in range(D):
        tok = int(draft[i])
        pd = float(draft_probs[i, tok])
        pt = float(target_probs[i, tok])
        if pd > 0.0 and accept_u[i] * pd <= pt:
            out[1 + i] = tok
            accepted += 1
        else:
            resid = np.maximum(target_probs[i] - draft_probs[i], 0.0)
            out[1 + i] = _cdf_invert(ru, resid)
            done = True
            break
    if not done:
        out[1 + D] = _cdf_invert(ru, target_probs[D])
    out[0] = accepted
    return out


def _src(D: int, V: int) -> str:
    return f"""
func.func @f(%d: tensor<{D}xi32>, %tp: tensor<{D + 1}x{V}xf32>,
    %dp: tensor<{D}x{V}xf32>, %au: tensor<{D}xf32>, %ru: tensor<1xf32>)
    -> tensor<{D + 2}xi32> {{
  %r = "tessera.spec_accept_sample"(%d, %tp, %dp, %au, %ru)
       : (tensor<{D}xi32>, tensor<{D + 1}x{V}xf32>, tensor<{D}x{V}xf32>,
          tensor<{D}xf32>, tensor<1xf32>) -> tensor<{D + 2}xi32>
  return %r : tensor<{D + 2}xi32>
}}
"""


def _compile_to_hsaco(D: int, V: int) -> bytes:
    gen = subprocess.run(
        [str(TESSERA_OPT), "-", "--generate-rocm-spec-accept-sample-kernel"],
        input=_src(D, V), capture_output=True, text=True)
    assert gen.returncode == 0, f"kernel-gen failed: {gen.stderr}"
    pipe = ("builtin.module(convert-scf-to-cf,gpu.module(convert-gpu-to-rocdl),"
            f"rocdl-attach-target{{chip={CHIP}}},gpu-module-to-binary)")
    ser = subprocess.run(
        [str(TESSERA_OPT), "-", f"--pass-pipeline={pipe}"],
        input=gen.stdout, capture_output=True, text=True)
    assert ser.returncode == 0, f"serialize failed: {ser.stderr}"
    hsaco = _extract_hsaco(ser.stdout)
    assert hsaco[:4] == b"\x7fELF", f"not an ELF hsaco: {hsaco[:4]!r}"
    return hsaco


def _launch(hip, hsaco, draft, tp, dp, au, ru):
    D, V = dp.shape

    def f32(a):
        return np.ascontiguousarray(a, dtype=np.float32)

    draft = np.ascontiguousarray(draft, dtype=np.int32)
    tp, dp, au, ru = f32(tp), f32(dp), f32(au), f32(ru)
    # Discovery gate: hipInit / hipModuleLoadData failing means no usable AMD GPU
    # here → return None so the caller skips. PAST a successful module load the
    # device works, so symbol lookup / alloc / launch / sync failures are REAL
    # failures of this generated kernel and must fail the test, not be laundered
    # into a skip.
    if hip.hipInit(0) != 0:
        return None
    mod = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(mod), hsaco) != 0:
        return None
    fn = ctypes.c_void_p()
    assert hip.hipModuleGetFunction(
        ctypes.byref(fn), mod, b"tessera_spec_accept_sample_0") == 0, \
        "kernel symbol tessera_spec_accept_sample_0 not found in module"

    def dev(arr):
        nb = arr.nbytes
        p = ctypes.c_void_p()
        assert hip.hipMalloc(ctypes.byref(p), nb) == 0, "hipMalloc failed"
        hip.hipMemcpy(p, arr.ctypes.data_as(ctypes.c_void_p), nb, 1)
        return p, nb

    dd, _ = dev(draft)
    dtp, _ = dev(tp)
    ddp, _ = dev(dp)
    dau, _ = dev(au)
    dru, _ = dev(ru)
    do = ctypes.c_void_p()
    assert hip.hipMalloc(ctypes.byref(do), (D + 2) * 4) == 0, "hipMalloc(out) failed"

    def memref(p, n):
        return [ctypes.c_void_p(p.value), ctypes.c_void_p(p.value),
                ctypes.c_int64(0), ctypes.c_int64(n), ctypes.c_int64(1)]

    args = (memref(dd, D) + memref(dtp, (D + 1) * V) + memref(ddp, D * V)
            + memref(dau, D) + memref(dru, 1) + memref(do, D + 2))
    arr = (ctypes.c_void_p * len(args))()
    for i, a_ in enumerate(args):
        arr[i] = ctypes.cast(ctypes.byref(a_), ctypes.c_void_p)
    launch = hip.hipModuleLaunchKernel
    launch.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint] * 6
                       + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p])
    rc = launch(fn, 1, 1, 1, 1, 1, 1, 0, None, arr, None)  # single thread
    assert rc == 0, (
        f"hipModuleLaunchKernel failed (rc={rc}) — kernel failed to launch")
    rc = hip.hipDeviceSynchronize()
    assert rc == 0, f"hipDeviceSynchronize failed (rc={rc}) — kernel crashed"
    out = np.zeros(D + 2, dtype=np.int32)
    hip.hipMemcpy(out.ctypes.data_as(ctypes.c_void_p), do, (D + 2) * 4, 2)
    for p in (dd, dtp, ddp, dau, dru, do):
        hip.hipFree(p)
    return out


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_spec_accept_sample_on_gfx1151(seed):
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    hip = _load_hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable — no ROCm host")
    rng = np.random.default_rng(seed)
    D, V = 5, 8
    draft = rng.integers(0, V, size=D, dtype=np.int32)
    # full softmax distributions (the exactness condition) over the same support.
    tp = _softmax(rng.standard_normal((D + 1, V)) * 1.5).astype(np.float32)
    dp = _softmax(rng.standard_normal((D, V)) * 1.5).astype(np.float32)
    au = rng.random(D).astype(np.float32)
    ru = rng.random(1).astype(np.float32)
    out = _launch(hip, _compile_to_hsaco(D, V), draft, tp, dp, au, ru)
    if out is None:
        pytest.skip("no usable AMD GPU (hipInit / hipModuleLoadData failed)")
    ref = _ref(draft, tp, dp, au, ru)
    np.testing.assert_array_equal(out, ref)


def test_spec_accept_sample_full_accept_on_gfx1151():
    # accept_u all 0 → every draft token accepts (0*pd <= pt always) → bonus path.
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    hip = _load_hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable — no ROCm host")
    rng = np.random.default_rng(99)
    D, V = 4, 8
    draft = rng.integers(0, V, size=D, dtype=np.int32)
    tp = _softmax(rng.standard_normal((D + 1, V))).astype(np.float32)
    dp = _softmax(rng.standard_normal((D, V))).astype(np.float32)
    au = np.zeros(D, np.float32)
    ru = np.array([0.5], np.float32)
    out = _launch(hip, _compile_to_hsaco(D, V), draft, tp, dp, au, ru)
    if out is None:
        pytest.skip("no usable AMD GPU (hipInit / hipModuleLoadData failed)")
    ref = _ref(draft, tp, dp, au, ru)
    assert int(out[0]) == D  # all accepted
    np.testing.assert_array_equal(out, ref)
