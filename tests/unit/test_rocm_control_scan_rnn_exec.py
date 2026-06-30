"""CF4e-3 — a full nonlinear RNN-cell scan executes as ONE cooperative-workgroup
device kernel on gfx1151. GenerateROCMControlScanRnnKernel lowers a control_scan
with TWO captures (W, U), a bias b, a per-step input x_t, and a tanh activation —
the canonical Elman/GRU-style recurrent cell

    h_t = tanh(h_{t-1} @ W + x_t @ U + b) ,   y_t = h_t

over a 1xK carry h. Two GEMVs (h@W over the LDS carry, x@U over the per-step
input) + bias + tanh; h lives in LDS and a gpu.barrier separates steps, combined
with the per-step xs-in / stacked-ys-out streaming.

Compared step-for-step + element-for-element to the same recurrence in numpy.
tanh-bounded → tight tolerance. Skip-clean off-GPU / without tessera-opt.
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


def _src(k: int, trip: int) -> str:
    h = f"tensor<1x{k}xf32>"
    w = f"tensor<{k}x{k}xf32>"
    xs = f"tensor<{trip}x1x{k}xf32>"  # (trip, *y.shape), y = 1×K
    return f"""
func.func @sb(%h: {h}, %x: {h}, %w: {w}, %u: {w}, %b: {h}) -> ({h}, {h}) {{
  %m1 = "tessera.matmul"(%h, %w) : ({h}, {w}) -> {h}
  %m2 = "tessera.matmul"(%x, %u) : ({h}, {w}) -> {h}
  %s1 = "tessera.add"(%m1, %m2) : ({h}, {h}) -> {h}
  %s2 = "tessera.add"(%s1, %b) : ({h}, {h}) -> {h}
  %t = "tessera.tanh"(%s2) : ({h}) -> {h}
  return %t, %t : {h}, {h}
}}
func.func @f(%init: {h}, %xs: {xs}, %W: {w}, %U: {w}, %b: {h}) -> ({h}, {xs}) {{
  %c, %ys = "tessera.control_scan"(%init, %xs, %W, %U, %b) {{body = @sb,
       trip = {trip} : i64, carry_arg_index = 0 : i64}}
       : ({h}, {xs}, {w}, {w}, {h}) -> ({h}, {xs})
  return %c, %ys : {h}, {xs}
}}
"""


def _compile_to_hsaco(k: int, trip: int) -> bytes:
    gen = subprocess.run(
        [str(TESSERA_OPT), "-", "--generate-rocm-control-scan-rnn-kernel"],
        input=_src(k, trip), capture_output=True, text=True)
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


def _launch(hip, hsaco, init, xs, W, U, bvec):
    k = init.size
    trip = xs.shape[0]
    nb_c = k * 4
    nb_x = trip * k * 4
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
        ctypes.byref(fn), mod, b"tessera_control_scan_rnn_0") == 0, \
        "kernel symbol tessera_control_scan_rnn_0 not found in module"
    bufs = [ctypes.c_void_p() for _ in range(7)]  # INIT,XS,W,U,B,COUT,YS
    sizes = [nb_c, nb_x, nb_w, nb_w, nb_c, nb_c, nb_x]
    for d, sz in zip(bufs, sizes):
        assert hip.hipMalloc(ctypes.byref(d), sz) == 0, "hipMalloc failed"
    hip.hipMemcpy(bufs[0], init.ctypes.data_as(ctypes.c_void_p), nb_c, 1)
    hip.hipMemcpy(bufs[1], xs.ctypes.data_as(ctypes.c_void_p), nb_x, 1)
    hip.hipMemcpy(bufs[2], W.ctypes.data_as(ctypes.c_void_p), nb_w, 1)
    hip.hipMemcpy(bufs[3], U.ctypes.data_as(ctypes.c_void_p), nb_w, 1)
    hip.hipMemcpy(bufs[4], bvec.ctypes.data_as(ctypes.c_void_p), nb_c, 1)

    def memref(p, n):
        return [ctypes.c_void_p(p.value), ctypes.c_void_p(p.value),
                ctypes.c_int64(0), ctypes.c_int64(n), ctypes.c_int64(1)]

    counts = [k, trip * k, k * k, k * k, k, k, trip * k]
    args = []
    for p, n in zip(bufs, counts):
        args += memref(p, n)
    args += [ctypes.c_int64(k)]
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
    out_c = np.zeros(k, dtype=np.float32)
    out_y = np.zeros(trip * k, dtype=np.float32)
    hip.hipMemcpy(out_c.ctypes.data_as(ctypes.c_void_p), bufs[5], nb_c, 2)
    hip.hipMemcpy(out_y.ctypes.data_as(ctypes.c_void_p), bufs[6], nb_x, 2)
    for d in bufs:
        hip.hipFree(d)
    return out_c, out_y.reshape(trip, k)


@pytest.mark.parametrize("k,trip", [(4, 3), (16, 5), (8, 4), (64, 3)])
def test_control_scan_rnn_executes_on_gfx1151(k, trip):
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    hip = _load_hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable — no ROCm host")
    rng = np.random.default_rng(67 + k + trip)
    init = (rng.standard_normal((1, k)) * 0.25).astype(np.float32)
    W = (rng.standard_normal((k, k)) / np.sqrt(k)).astype(np.float32)
    U = (rng.standard_normal((k, k)) / np.sqrt(k)).astype(np.float32)
    bvec = (rng.standard_normal((1, k)) * 0.1).astype(np.float32)
    xs = (rng.standard_normal((trip, 1, k)) * 0.5).astype(np.float32)
    res = _launch(hip, _compile_to_hsaco(k, trip), init, xs, W, U, bvec)
    if res is None:
        # Only pre-launch discovery failed; a launch/sync failure raises.
        pytest.skip("no usable AMD GPU (hipInit / hipModuleLoadData failed)")
    out_c, out_ys = res
    # numpy reference: h_t = tanh(h_{t-1} @ W + x_t @ U + b) ; y_t = h_t.
    h = init.copy()
    ref_ys = np.zeros((trip, k), dtype=np.float32)
    for t in range(trip):
        h = np.tanh(h @ W + xs[t] @ U + bvec).astype(np.float32)
        ref_ys[t] = h.ravel()
    np.testing.assert_allclose(out_ys, ref_ys, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(out_c, h.ravel(), rtol=2e-3, atol=2e-3)
