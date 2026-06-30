"""CF4e-1 — an elementwise-body Tessera control_scan executes as ONE device
kernel on gfx1151 (RDNA3.5). GenerateROCMControlScanKernel lowers a
`control_scan { (carry, y) = body(carry, xs[t]) ; ys[t] = y }` (carry/xt rank-1
f32 of width K; xs/ys are T×K) to a per-thread gpu.func: thread g owns carry
element g and runs the trip-count loop locally, reading xs[t,g], updating the
carry, and writing ys[t,g]. The per-step xs input + stacked ys output are exactly
what scan adds over control_for.

Two bodies are exercised:
  * running sum   : carry' = carry + xt ; y = carry'        (cumulative state)
  * gated recur   : carry' = tanh(carry + xt) ; y = carry'  (bounded RNN cell)
Compared element-for-element + step-for-step to the same recurrence in numpy.
Skip-clean off-GPU / without tessera-opt.
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


# body kind → (MLIR body ops producing the next carry %c2 from %c (carry) and
#              %x (xt)), python reference step fn.
def _body_mlir(kind: str, k: int) -> str:
    t = f"tensor<{k}xf32>"
    if kind == "sum":
        return f'  %c2 = "tessera.add"(%c, %x) : ({t}, {t}) -> {t}\n' \
               f"  return %c2, %c2 : {t}, {t}"
    # gated: tanh(carry + xt)
    return (f'  %s = "tessera.add"(%c, %x) : ({t}, {t}) -> {t}\n'
            f'  %c2 = "tessera.tanh"(%s) : ({t}) -> {t}\n'
            f"  return %c2, %c2 : {t}, {t}")


def _step(kind: str, c, x):
    s = c + x
    return s if kind == "sum" else np.tanh(s)


def _src(kind: str, k: int, trip: int) -> str:
    t = f"tensor<{k}xf32>"
    xt = f"tensor<{trip}x{k}xf32>"
    return f"""
func.func @sb(%c: {t}, %x: {t}) -> ({t}, {t}) {{
{_body_mlir(kind, k)}
}}
func.func @f(%init: {t}, %xs: {xt}) -> ({t}, {xt}) {{
  %c, %ys = "tessera.control_scan"(%init, %xs) {{body = @sb, trip = {trip} : i64,
       carry_arg_index = 0 : i64}} : ({t}, {xt}) -> ({t}, {xt})
  return %c, %ys : {t}, {xt}
}}
"""


def _compile_to_hsaco(kind: str, k: int, trip: int) -> bytes:
    gen = subprocess.run(
        [str(TESSERA_OPT), "-", "--generate-rocm-control-scan-kernel"],
        input=_src(kind, k, trip), capture_output=True, text=True)
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


def _launch(hip, hsaco, init, xs):
    k = init.size
    trip = xs.shape[0]
    nb_c = k * 4
    nb_x = trip * k * 4
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
        ctypes.byref(fn), mod, b"tessera_control_scan_0") == 0, \
        "kernel symbol tessera_control_scan_0 not found in module"
    di, dx, dy, dc = (ctypes.c_void_p(), ctypes.c_void_p(),
                      ctypes.c_void_p(), ctypes.c_void_p())
    assert hip.hipMalloc(ctypes.byref(di), nb_c) == 0, "hipMalloc(init) failed"
    assert hip.hipMalloc(ctypes.byref(dx), nb_x) == 0, "hipMalloc(xs) failed"
    assert hip.hipMalloc(ctypes.byref(dy), nb_x) == 0, "hipMalloc(ys) failed"
    assert hip.hipMalloc(ctypes.byref(dc), nb_c) == 0, "hipMalloc(cout) failed"
    hip.hipMemcpy(di, init.ctypes.data_as(ctypes.c_void_p), nb_c, 1)
    hip.hipMemcpy(dx, xs.ctypes.data_as(ctypes.c_void_p), nb_x, 1)

    def memref(p, n):
        return [ctypes.c_void_p(p.value), ctypes.c_void_p(p.value),
                ctypes.c_int64(0), ctypes.c_int64(n), ctypes.c_int64(1)]

    args = (memref(di, k) + memref(dx, trip * k) + memref(dy, trip * k)
            + memref(dc, k) + [ctypes.c_int64(k)])
    arr = (ctypes.c_void_p * len(args))()
    for i, a_ in enumerate(args):
        arr[i] = ctypes.cast(ctypes.byref(a_), ctypes.c_void_p)
    launch = hip.hipModuleLaunchKernel
    launch.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint] * 6
                       + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p])
    # one block of BD threads (k <= BD), grid over the k carry elements.
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
    for d in (di, dx, dy, dc):
        hip.hipFree(d)
    return out_c, out_y.reshape(trip, k)


@pytest.mark.parametrize("kind,k,trip", [
    ("sum", 4, 3), ("sum", 16, 5), ("gated", 8, 4), ("gated", 64, 3)])
def test_control_scan_executes_on_gfx1151(kind, k, trip):
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    hip = _load_hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable — no ROCm host")
    rng = np.random.default_rng(31 + k + trip + len(kind))
    # keep the running sum well inside f32; gated is bounded by tanh anyway.
    init = (rng.standard_normal(k) * 0.25).astype(np.float32)
    xs = (rng.standard_normal((trip, k)) * 0.25).astype(np.float32)
    res = _launch(hip, _compile_to_hsaco(kind, k, trip), init, xs)
    if res is None:
        # Only pre-launch discovery failed; a launch/sync failure raises.
        pytest.skip("no usable AMD GPU (hipInit / hipModuleLoadData failed)")
    out_c, out_ys = res
    # numpy reference: same recurrence, carry threaded, ys[t] = carry'.
    c = init.copy()
    ref_ys = np.zeros((trip, k), dtype=np.float32)
    for t in range(trip):
        c = _step(kind, c, xs[t]).astype(np.float32)
        ref_ys[t] = c
    np.testing.assert_allclose(out_ys, ref_ys, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(out_c, c, rtol=2e-3, atol=2e-3)
