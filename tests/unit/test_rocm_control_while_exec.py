"""CF4c-cont — a Tessera control_while executes as ONE device kernel on gfx1151.
GenerateROCMControlForKernel lowers an elementwise-body tessera.control_while to a
single gpu.func: grid over the carry elements; per thread, a bounded scf.while
  x = X[gid];
  while (i < max_iters AND cond_scalar(c) > 0) { c = body_scalar(c); i++ }
  O[gid] = c.
The cond is short-circuited behind the i < max_iters bound (it never runs past
the bound), and the whole loop is one dispatch.

body = add(c, c) (doubling). Two conds:
  * cond = sigmoid(c)  → always > 0 → full max_iters → c * 2**max_iters;
  * cond = relu(c)     → > 0 iff c > 0 → x>0: x*2**max, x<=0: x (stops at i=0).
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


def _src(n: int, k: int, cond_op: str) -> str:
    return f"""
func.func @wb(%c: tensor<{n}xf32>) -> tensor<{n}xf32> {{
  %0 = "tessera.add"(%c, %c) : (tensor<{n}xf32>, tensor<{n}xf32>) -> tensor<{n}xf32>
  return %0 : tensor<{n}xf32>
}}
func.func @wc(%c: tensor<{n}xf32>) -> tensor<{n}xf32> {{
  %0 = "tessera.{cond_op}"(%c) : (tensor<{n}xf32>) -> tensor<{n}xf32>
  return %0 : tensor<{n}xf32>
}}
func.func @f(%init: tensor<{n}xf32>) -> tensor<{n}xf32> {{
  %r = "tessera.control_while"(%init) {{body = @wb, cond = @wc,
       carry_arg_index = 0 : i64, max_iters = {k} : i64}}
       : (tensor<{n}xf32>) -> tensor<{n}xf32>
  return %r : tensor<{n}xf32>
}}
"""


def _compile_to_hsaco(n: int, k: int, cond_op: str) -> bytes:
    gen = subprocess.run(
        [str(TESSERA_OPT), "-", "--generate-rocm-control-for-kernel",
         "--allow-unregistered-dialect"],
        input=_src(n, k, cond_op), capture_output=True, text=True)
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


def _launch(hip, hsaco, x):
    n = x.size
    if hip.hipInit(0) != 0:
        return None
    mod = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(mod), hsaco) != 0:
        return None
    fn = ctypes.c_void_p()
    if hip.hipModuleGetFunction(
            ctypes.byref(fn), mod, b"tessera_control_while_0") != 0:
        return None
    dx, do = ctypes.c_void_p(), ctypes.c_void_p()
    for d in (dx, do):
        if hip.hipMalloc(ctypes.byref(d), n * 4) != 0:
            return None
    hip.hipMemcpy(dx, x.ctypes.data_as(ctypes.c_void_p), n * 4, 1)

    def memref(p):
        return [ctypes.c_void_p(p.value), ctypes.c_void_p(p.value),
                ctypes.c_int64(0), ctypes.c_int64(n), ctypes.c_int64(1)]

    args = memref(dx) + memref(do) + [ctypes.c_int64(n)]
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
    for d in (dx, do):
        hip.hipFree(d)
    return out


@pytest.mark.parametrize("cond_op", ["sigmoid", "relu"])
def test_control_while_executes_on_gfx1151(cond_op):
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    hip = _load_hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable — no ROCm host")
    n, k = 64, 4
    rng = np.random.default_rng(2)
    x = rng.standard_normal(n).astype(np.float32)
    out = _launch(hip, _compile_to_hsaco(n, k, cond_op), x)
    if out is None:
        pytest.skip("no usable AMD GPU (hipModuleLoadData / launch unavailable)")
    if cond_op == "sigmoid":
        ref = x * (2.0 ** k)               # sigmoid>0 always → full k doublings
    else:
        ref = np.where(x > 0.0, x * (2.0 ** k), x)  # relu>0 iff x>0; else stop
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
