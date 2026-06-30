"""CF4f — a CROSS-ELEMENT control_while (power iteration) executes as ONE
cooperative-workgroup device kernel on gfx1151. GenerateROCMControlWhileGemvKernel
lowers `h = h @ W  while Σh > eps` (bounded by max_iters) over a 1xK carry h and a
KxK capture W. Both the body (a GEMV) and the continuation cond (Σh) are reductions
over the whole carry — so it can't be the per-thread elementwise control_while
(CF4c-cont). The carry lives in LDS; every thread computes the SAME Σh reduction
and the SAME predicate (uniform continuation), which is what makes the
per-iteration gpu.barriers safe. The W capture is threaded as a kernel arg.

W is built with each row summing to r (positive entries) and h positive, so
Σ(h@W) = r·Σh decays geometrically — the loop's data-dependent stopping is
exercised BOTH ways: an early stop (eps above the decay floor) and a run to
max_iters (eps tiny). The numpy reference runs the identical loop. Skip-clean
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
BD = 256


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


def _src(k: int, max_iters: int, eps: float) -> str:
    h = f"tensor<1x{k}xf32>"
    w = f"tensor<{k}x{k}xf32>"
    return f"""
func.func @b(%h: {h}, %w: {w}) -> {h} {{
  %m = "tessera.matmul"(%h, %w) : ({h}, {w}) -> {h}
  return %m : {h}
}}
func.func @c(%h: {h}) -> tensor<1x1xf32> {{
  %s = "tessera.reduce"(%h) {{kind = "sum", axis = 1 : i64}} : ({h}) -> tensor<1x1xf32>
  return %s : tensor<1x1xf32>
}}
func.func @f(%init: {h}, %W: {w}) -> {h} {{
  %r = "tessera.control_while"(%init, %W) {{body = @b, cond = @c,
       carry_arg_index = 0 : i64, max_iters = {max_iters} : i64,
       tessera.while_cond_eps = {eps:.8e} : f32}} : ({h}, {w}) -> {h}
  return %r : {h}
}}
"""


def _compile_to_hsaco(k: int, max_iters: int, eps: float) -> bytes:
    gen = subprocess.run(
        [str(TESSERA_OPT), "-", "--generate-rocm-control-while-gemv-kernel"],
        input=_src(k, max_iters, eps), capture_output=True, text=True)
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


def _launch(hip, hsaco, init, W):
    k = init.size
    nb_c = k * 4
    nb_w = k * k * 4
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
        ctypes.byref(fn), mod, b"tessera_control_while_gemv_0") == 0, \
        "kernel symbol tessera_control_while_gemv_0 not found in module"
    dh, dw, do = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    assert hip.hipMalloc(ctypes.byref(dh), nb_c) == 0, "hipMalloc(h) failed"
    assert hip.hipMalloc(ctypes.byref(dw), nb_w) == 0, "hipMalloc(W) failed"
    assert hip.hipMalloc(ctypes.byref(do), nb_c) == 0, "hipMalloc(out) failed"
    hip.hipMemcpy(dh, init.ctypes.data_as(ctypes.c_void_p), nb_c, 1)
    hip.hipMemcpy(dw, W.ctypes.data_as(ctypes.c_void_p), nb_w, 1)

    def memref(p, n):
        return [ctypes.c_void_p(p.value), ctypes.c_void_p(p.value),
                ctypes.c_int64(0), ctypes.c_int64(n), ctypes.c_int64(1)]

    args = memref(dh, k) + memref(dw, k * k) + memref(do, k) + [ctypes.c_int64(k)]
    arr = (ctypes.c_void_p * len(args))()
    for i, a_ in enumerate(args):
        arr[i] = ctypes.cast(ctypes.byref(a_), ctypes.c_void_p)
    launch = hip.hipModuleLaunchKernel
    launch.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint] * 6
                       + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p])
    rc = launch(fn, 1, 1, 1, BD, 1, 1, 0, None, arr, None)
    assert rc == 0, (
        f"hipModuleLaunchKernel failed (rc={rc}) for k={k} — kernel failed to "
        "launch on a working GPU")
    rc = hip.hipDeviceSynchronize()
    assert rc == 0, f"hipDeviceSynchronize failed (rc={rc}) for k={k} — crashed"
    out = np.zeros(k, dtype=np.float32)
    hip.hipMemcpy(out.ctypes.data_as(ctypes.c_void_p), do, nb_c, 2)
    for d in (dh, dw, do):
        hip.hipFree(d)
    return out


def _ref(init, W, max_iters, eps):
    h = init.copy()
    i = 0
    while i < max_iters and float(h.sum()) > eps:
        h = (h @ W).astype(np.float32)
        i += 1
    return h.ravel(), i


@pytest.mark.parametrize("k,max_iters,mode", [
    (4, 16, "early"), (16, 16, "early"), (8, 5, "runmax"), (4, 6, "runmax")])
def test_control_while_gemv_executes_on_gfx1151(k, max_iters, mode):
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    hip = _load_hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable — no ROCm host")
    rng = np.random.default_rng(83 + k + max_iters)
    init = np.abs(rng.standard_normal((1, k))).astype(np.float32) + 0.1
    # W with each row summing to r=0.5 (positive) → Σ(h@W) = 0.5·Σh, monotone decay.
    Wraw = np.abs(rng.standard_normal((k, k))).astype(np.float32) + 1e-3
    W = (Wraw / Wraw.sum(axis=1, keepdims=True) * 0.5).astype(np.float32)
    s0 = float(init.sum())
    # early: eps above the floor → stops in a few steps; runmax: eps ~0 → runs all.
    eps = (s0 * 0.3) if mode == "early" else 1e-9
    out = _launch(hip, _compile_to_hsaco(k, max_iters, eps), init, W)
    if out is None:
        # Only pre-launch discovery failed; a launch/sync failure raises.
        pytest.skip("no usable AMD GPU (hipInit / hipModuleLoadData failed)")
    ref, niter = _ref(init, W, max_iters, eps)
    if mode == "early":
        assert 0 < niter < max_iters, f"expected early stop, ran {niter}/{max_iters}"
    else:
        assert niter == max_iters, f"expected run-to-max, ran {niter}/{max_iters}"
    np.testing.assert_allclose(out, ref, rtol=2e-3, atol=2e-3)
