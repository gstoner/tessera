"""CF4e-2 — a linear state-space scan executes as ONE cooperative-workgroup
device kernel on gfx1151. GenerateROCMControlScanGemvKernel lowers a control_scan
with a CROSS-ELEMENT body and a CAPTURE — the canonical linear SSM / linear-
attention state update

    h_t = h_{t-1} @ W + x_t ,   y_t = h_t

over a 1xK carry h, a KxK loop-invariant capture W, and a per-step input slice
x_t (the scan's xs). The `h @ W` is a GEMV (a reduction over the whole carry), so
it can't be the per-thread elementwise scan (CF4e-1); h lives in LDS and a
gpu.barrier separates steps, combined with the per-step xs-in / stacked-ys-out
streaming.

Compared step-for-step + element-for-element to the same recurrence in numpy.
The recurrence is linear and f32 → tight tolerance. Skip-clean off-GPU / without
tessera-opt.
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
    xs = f"tensor<{trip}x{k}xf32>"
    return f"""
func.func @sb(%h: {h}, %x: {h}, %w: {w}) -> ({h}, {h}) {{
  %m = "tessera.matmul"(%h, %w) : ({h}, {w}) -> {h}
  %s = "tessera.add"(%m, %x) : ({h}, {h}) -> {h}
  return %s, %s : {h}, {h}
}}
func.func @f(%init: {h}, %xs: {xs}, %W: {w}) -> ({h}, {xs}) {{
  %c, %ys = "tessera.control_scan"(%init, %xs, %W) {{body = @sb, trip = {trip} : i64,
       carry_arg_index = 0 : i64}} : ({h}, {xs}, {w}) -> ({h}, {xs})
  return %c, %ys : {h}, {xs}
}}
"""


def _compile_to_hsaco(k: int, trip: int) -> bytes:
    gen = subprocess.run(
        [str(TESSERA_OPT), "-", "--generate-rocm-control-scan-gemv-kernel"],
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


def _launch(hip, hsaco, init, xs, W):
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
        ctypes.byref(fn), mod, b"tessera_control_scan_gemv_0") == 0, \
        "kernel symbol tessera_control_scan_gemv_0 not found in module"
    di, dx, dw, dc, dy = (ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p(),
                          ctypes.c_void_p(), ctypes.c_void_p())
    assert hip.hipMalloc(ctypes.byref(di), nb_c) == 0, "hipMalloc(init) failed"
    assert hip.hipMalloc(ctypes.byref(dx), nb_x) == 0, "hipMalloc(xs) failed"
    assert hip.hipMalloc(ctypes.byref(dw), nb_w) == 0, "hipMalloc(W) failed"
    assert hip.hipMalloc(ctypes.byref(dc), nb_c) == 0, "hipMalloc(cout) failed"
    assert hip.hipMalloc(ctypes.byref(dy), nb_x) == 0, "hipMalloc(ys) failed"
    hip.hipMemcpy(di, init.ctypes.data_as(ctypes.c_void_p), nb_c, 1)
    hip.hipMemcpy(dx, xs.ctypes.data_as(ctypes.c_void_p), nb_x, 1)
    hip.hipMemcpy(dw, W.ctypes.data_as(ctypes.c_void_p), nb_w, 1)

    def memref(p, n):
        return [ctypes.c_void_p(p.value), ctypes.c_void_p(p.value),
                ctypes.c_int64(0), ctypes.c_int64(n), ctypes.c_int64(1)]

    # ABI: (INIT, XS, W, COUT, YS : memref<?xf32>, K : index)
    args = (memref(di, k) + memref(dx, trip * k) + memref(dw, k * k)
            + memref(dc, k) + memref(dy, trip * k) + [ctypes.c_int64(k)])
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
    hip.hipMemcpy(out_c.ctypes.data_as(ctypes.c_void_p), dc, nb_c, 2)
    hip.hipMemcpy(out_y.ctypes.data_as(ctypes.c_void_p), dy, nb_x, 2)
    for d in (di, dx, dw, dc, dy):
        hip.hipFree(d)
    return out_c, out_y.reshape(trip, k)


@pytest.mark.parametrize("k,trip", [(4, 3), (16, 5), (8, 4), (64, 3)])
def test_control_scan_gemv_executes_on_gfx1151(k, trip):
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    hip = _load_hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable — no ROCm host")
    rng = np.random.default_rng(53 + k + trip)
    init = (rng.standard_normal((1, k)) * 0.25).astype(np.float32)
    # W a contraction so h_t = h_{t-1}@W + x_t stays well inside f32.
    W = (rng.standard_normal((k, k)) / (2.0 * np.sqrt(k))).astype(np.float32)
    xs = (rng.standard_normal((trip, k)) * 0.25).astype(np.float32)
    res = _launch(hip, _compile_to_hsaco(k, trip), init, xs, W)
    if res is None:
        # Only pre-launch discovery failed; a launch/sync failure raises.
        pytest.skip("no usable AMD GPU (hipInit / hipModuleLoadData failed)")
    out_c, out_ys = res
    # numpy reference: h_t = h_{t-1} @ W + x_t ; y_t = h_t.
    h = init.copy()
    ref_ys = np.zeros((trip, k), dtype=np.float32)
    for t in range(trip):
        h = (h @ W + xs[t]).astype(np.float32)
        ref_ys[t] = h
    np.testing.assert_allclose(out_ys, ref_ys, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(out_c, h.ravel(), rtol=2e-3, atol=2e-3)
