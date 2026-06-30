"""CF4b — a Tessera bounded control loop executes as ONE device kernel on
gfx1151. GenerateROCMControlForKernel lowers an elementwise-body
tessera.control_for to a single gpu.func (grid over the carry's elements;
per-thread scf.for over K iterations), which compiles
generate → convert-scf-to-cf → convert-gpu-to-rocdl → rocdl-attach-target →
gpu-module-to-binary → hsaco and runs via hipModuleLaunchKernel. The whole loop
is one dispatch, not one launch per iteration.

Body `add(carry, carry)` is exact: K doublings → element * 2**K. Skip-clean when
tessera-opt isn't built or no usable AMD GPU is present.
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
BD = 256  # must match GenerateROCMControlForKernel's block dim


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


def _doubling_loop_mlir(n: int, k: int) -> str:
    return f"""
func.func @loop_body(%c: tensor<{n}xf32>) -> tensor<{n}xf32> {{
  %0 = "tessera.add"(%c, %c) : (tensor<{n}xf32>, tensor<{n}xf32>) -> tensor<{n}xf32>
  return %0 : tensor<{n}xf32>
}}
func.func @f(%init: tensor<{n}xf32>) -> tensor<{n}xf32> {{
  %r = "tessera.control_for"(%init) {{body = @loop_body, start = 0 : i64,
       stop = {k} : i64, step = 1 : i64, carry_arg_index = 0 : i64}}
       : (tensor<{n}xf32>) -> tensor<{n}xf32>
  return %r : tensor<{n}xf32>
}}
"""


def _compile_to_hsaco(n: int, k: int) -> bytes:
    src = _doubling_loop_mlir(n, k)
    gen = subprocess.run(
        [str(TESSERA_OPT), "-", "--generate-rocm-control-for-kernel",
         "--allow-unregistered-dialect"],
        input=src, capture_output=True, text=True)
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


def _launch(hip, hsaco: bytes, x: np.ndarray) -> np.ndarray | None:
    """Run @tessera_control_for_0(X, O, N). Returns O, or None if the device is
    unusable (skip-clean)."""
    n = x.size
    if hip.hipInit(0) != 0:
        return None
    mod = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(mod), hsaco) != 0:
        return None
    fn = ctypes.c_void_p()
    if hip.hipModuleGetFunction(
            ctypes.byref(fn), mod, b"tessera_control_for_0") != 0:
        return None
    nb = n * 4
    dx, do = ctypes.c_void_p(), ctypes.c_void_p()
    for d in (dx, do):
        if hip.hipMalloc(ctypes.byref(d), nb) != 0:
            return None
    hip.hipMemcpy(dx, x.ctypes.data_as(ctypes.c_void_p), nb, 1)  # H2D

    # MLIR memref-descriptor ABI per memref<?xf32>:
    #   alloc_ptr, aligned_ptr, offset:i64, size:i64, stride:i64
    # then the trailing `index N` as i64.
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
    hip.hipMemcpy(out.ctypes.data_as(ctypes.c_void_p), do, nb, 2)  # D2H
    for d in (dx, do):
        hip.hipFree(d)
    return out


@pytest.mark.parametrize("k", [1, 4, 10])
def test_control_for_doubling_executes_on_gfx1151(k):
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    hip = _load_hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable — no ROCm host")
    n = 64
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n).astype(np.float32)
    hsaco = _compile_to_hsaco(n, k)
    out = _launch(hip, hsaco, x)
    if out is None:
        pytest.skip("no usable AMD GPU (hipModuleLoadData / launch unavailable)")
    ref = x * (2.0 ** k)   # K doublings inside the per-thread scf.for
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
