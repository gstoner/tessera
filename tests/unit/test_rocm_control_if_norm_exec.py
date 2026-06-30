"""CF4d-if — a CROSS-ELEMENT control_if executes as ONE cooperative-workgroup
device kernel on gfx1151. GenerateROCMControlIfNormKernel lowers
`O = flag>0 ? rmsnorm(x) : layer_norm(x)` over a 1xK carry to a single gpu.func:
x lives in LDS, the shape-(1) flag is read once and is UNIFORM across the
workgroup, and a uniform scf.if selects which cooperative norm every thread
computes (divergent — only the taken norm's reduction runs). Each branch is a
reduction over the whole carry, so this can't be the per-thread elementwise
control_if (CF4c).

Both flag>0 (→ rmsnorm) and flag<0 (→ layer_norm) are exercised and compared to
the matching numpy norm. Skip-clean off-GPU / without tessera-opt.
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
EPS = 1e-5


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


def _src(k: int) -> str:
    t = f"tensor<1x{k}xf32>"
    return f"""
func.func @t(%x: {t}) -> {t} {{
  %0 = "tessera.rmsnorm"(%x) {{eps = 1.000000e-05 : f64}} : ({t}) -> {t}
  return %0 : {t}
}}
func.func @e(%x: {t}) -> {t} {{
  %0 = "tessera.layer_norm"(%x) {{eps = 1.000000e-05 : f64}} : ({t}) -> {t}
  return %0 : {t}
}}
func.func @f(%x: {t}, %flag: tensor<1xf32>) -> {t} {{
  %o = "tessera.control_if"(%x, %flag) {{then_branch = @t, else_branch = @e,
       flag_arg_index = 1 : i64}} : ({t}, tensor<1xf32>) -> {t}
  return %o : {t}
}}
"""


def _compile_to_hsaco(k: int) -> bytes:
    gen = subprocess.run(
        [str(TESSERA_OPT), "-", "--generate-rocm-control-if-norm-kernel"],
        input=_src(k), capture_output=True, text=True)
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


def _rmsnorm(x):
    return x / np.sqrt(np.mean(x * x) + EPS)


def _layer_norm(x):
    mu = np.mean(x)
    return (x - mu) / np.sqrt(np.mean((x - mu) ** 2) + EPS)


def _launch(hip, hsaco, x, flag):
    k = x.size
    nb = k * 4
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
        ctypes.byref(fn), mod, b"tessera_control_if_norm_0") == 0, \
        "kernel symbol tessera_control_if_norm_0 not found in module"
    dx, df, do = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    assert hip.hipMalloc(ctypes.byref(dx), nb) == 0, "hipMalloc(x) failed"
    assert hip.hipMalloc(ctypes.byref(df), 4) == 0, "hipMalloc(flag) failed"
    assert hip.hipMalloc(ctypes.byref(do), nb) == 0, "hipMalloc(out) failed"
    hip.hipMemcpy(dx, x.ctypes.data_as(ctypes.c_void_p), nb, 1)
    fl = np.asarray([flag], dtype=np.float32)
    hip.hipMemcpy(df, fl.ctypes.data_as(ctypes.c_void_p), 4, 1)

    def memref(p, n):
        return [ctypes.c_void_p(p.value), ctypes.c_void_p(p.value),
                ctypes.c_int64(0), ctypes.c_int64(n), ctypes.c_int64(1)]

    args = memref(dx, k) + memref(df, 1) + memref(do, k) + [ctypes.c_int64(k)]
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
    hip.hipMemcpy(out.ctypes.data_as(ctypes.c_void_p), do, nb, 2)
    for d in (dx, df, do):
        hip.hipFree(d)
    return out


@pytest.mark.parametrize("k,flag", [(4, 1.0), (4, -1.0), (16, 1.0), (8, -1.0)])
def test_control_if_norm_executes_on_gfx1151(k, flag):
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    hip = _load_hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable — no ROCm host")
    rng = np.random.default_rng(41 + k + int(flag))
    x = (rng.standard_normal(k) * 3.0).astype(np.float32)
    out = _launch(hip, _compile_to_hsaco(k), x, flag)
    if out is None:
        # Only pre-launch discovery failed; a launch/sync failure raises.
        pytest.skip("no usable AMD GPU (hipInit / hipModuleLoadData failed)")
    # flag>0 → rmsnorm ; flag<0 → layer_norm.
    ref = _rmsnorm(x.astype(np.float64)) if flag > 0 \
        else _layer_norm(x.astype(np.float64))
    np.testing.assert_allclose(out, ref, rtol=2e-3, atol=2e-3)
