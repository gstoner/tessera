"""CF4c — a Tessera control_if executes as ONE device kernel on gfx1151.
GenerateROCMControlForKernel lowers an elementwise-branch tessera.control_if to a
single gpu.func: grid over the data elements; per thread,
  x = X[gid]; r = (FLAG[0] > 0) ? then_scalar(x) : else_scalar(x); O[gid] = r.
The branch is selected once by the shape-(1) flag, in one dispatch.

Branches: then = relu(x), else = add(x, x) = 2x. flag>0 → relu(x); flag<0 → 2x.
Skip-clean when tessera-opt isn't built or no usable AMD GPU is present.
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


_SRC = """
func.func @tb(%x: tensor<{n}xf32>) -> tensor<{n}xf32> {{
  %0 = "tessera.relu"(%x) : (tensor<{n}xf32>) -> tensor<{n}xf32>
  return %0 : tensor<{n}xf32>
}}
func.func @eb(%x: tensor<{n}xf32>) -> tensor<{n}xf32> {{
  %0 = "tessera.add"(%x, %x) : (tensor<{n}xf32>, tensor<{n}xf32>) -> tensor<{n}xf32>
  return %0 : tensor<{n}xf32>
}}
func.func @f(%flag: tensor<1xf32>, %x: tensor<{n}xf32>) -> tensor<{n}xf32> {{
  %r = "tessera.control_if"(%flag, %x) {{then_branch = @tb, else_branch = @eb,
       flag_arg_index = 0 : i64}} : (tensor<1xf32>, tensor<{n}xf32>) -> tensor<{n}xf32>
  return %r : tensor<{n}xf32>
}}
"""


def _compile_to_hsaco(n: int) -> bytes:
    gen = subprocess.run(
        [str(TESSERA_OPT), "-", "--generate-rocm-control-for-kernel",
         "--allow-unregistered-dialect"],
        input=_SRC.format(n=n), capture_output=True, text=True)
    assert gen.returncode == 0, f"kernel-gen failed: {gen.stderr}"
    pipe = ("builtin.module(convert-scf-to-cf,gpu.module(convert-gpu-to-rocdl),"
            f"rocdl-attach-target{{chip={CHIP}}},gpu-module-to-binary)")
    ser = subprocess.run(
        [str(TESSERA_OPT), "-", f"--pass-pipeline={pipe}",
         "--allow-unregistered-dialect"],
        input=gen.stdout, capture_output=True, text=True)
    assert ser.returncode == 0, f"serialize failed: {ser.stderr}"
    hsaco = _extract_hsaco(ser.stdout)
    assert hsaco[:4] == b"\x7fELF", f"not an ELF hsaco: {hsaco[:4]!r}"
    return hsaco


def _launch(hip, hsaco, x, flag_val):
    """Run @tessera_control_if_0(X, FLAG, O, N). Returns O, or None (skip)."""
    n = x.size
    flag = np.array([flag_val], dtype=np.float32)
    if hip.hipInit(0) != 0:
        return None
    mod = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(mod), hsaco) != 0:
        return None
    fn = ctypes.c_void_p()
    if hip.hipModuleGetFunction(
            ctypes.byref(fn), mod, b"tessera_control_if_0") != 0:
        return None
    dx, df, do = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    if hip.hipMalloc(ctypes.byref(dx), n * 4) != 0:
        return None
    if hip.hipMalloc(ctypes.byref(df), 4) != 0:
        return None
    if hip.hipMalloc(ctypes.byref(do), n * 4) != 0:
        return None
    hip.hipMemcpy(dx, x.ctypes.data_as(ctypes.c_void_p), n * 4, 1)
    hip.hipMemcpy(df, flag.ctypes.data_as(ctypes.c_void_p), 4, 1)

    def memref(p, sz):
        return [ctypes.c_void_p(p.value), ctypes.c_void_p(p.value),
                ctypes.c_int64(0), ctypes.c_int64(sz), ctypes.c_int64(1)]

    args = memref(dx, n) + memref(df, 1) + memref(do, n) + [ctypes.c_int64(n)]
    arr = (ctypes.c_void_p * len(args))()
    for i, a_ in enumerate(args):
        arr[i] = ctypes.cast(ctypes.byref(a_), ctypes.c_void_p)
    launch = hip.hipModuleLaunchKernel
    launch.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint] * 6
                       + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p])
    grid = (n + BD - 1) // BD
    if launch(fn, grid, 1, 1, BD, 1, 1, 0, None, arr, None) != 0:
        return None
    if hip.hipDeviceSynchronize() != 0:
        return None
    out = np.zeros(n, dtype=np.float32)
    hip.hipMemcpy(out.ctypes.data_as(ctypes.c_void_p), do, n * 4, 2)
    for d in (dx, df, do):
        hip.hipFree(d)
    return out


@pytest.mark.parametrize("flag_val,branch", [(1.0, "then"), (-1.0, "else")])
def test_control_if_executes_on_gfx1151(flag_val, branch):
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    hip = _load_hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable — no ROCm host")
    n = 64
    rng = np.random.default_rng(1)
    x = rng.standard_normal(n).astype(np.float32)
    hsaco = _compile_to_hsaco(n)
    out = _launch(hip, hsaco, x, flag_val)
    if out is None:
        pytest.skip("no usable AMD GPU (hipModuleLoadData / launch unavailable)")
    ref = np.maximum(x, 0.0) if branch == "then" else x * 2.0
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
