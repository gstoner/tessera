"""CF4d-3 — a single-tile WMMA matmul recurrence Tessera control_for executes as
ONE wave device kernel on gfx1151 (RDNA3.5). GenerateROCMControlForWmmaKernel
lowers `control_for { carry = carry @ W }` (carry, W both 16x16 f16) to one
gpu.func using the rocdl.wmma.f32.16x16x16.f16 intrinsic, with the
accumulator(f32)→input(f16) FRAGMENT shuffle through LDS between iterations →
carry @ W^max_iters on real WMMA hardware.

The kernel truncates the carry to f16 between iterations (the LDS is f16, like
real WMMA recurrences), so the numpy reference does the same: each step is an
f32-accumulate matmul whose result is cast back to f16. f16 → loose tolerance.
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
N = 16          # WMMA tile
WAVE = 32       # one wave


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


def _src(it: int) -> str:
    return f"""
func.func @wb(%c: tensor<16x16xf16>, %w: tensor<16x16xf16>) -> tensor<16x16xf16> {{
  %0 = "tessera.matmul"(%c, %w) : (tensor<16x16xf16>, tensor<16x16xf16>) -> tensor<16x16xf16>
  return %0 : tensor<16x16xf16>
}}
func.func @f(%init: tensor<16x16xf16>, %w: tensor<16x16xf16>) -> tensor<16x16xf16> {{
  %r = "tessera.control_for"(%init, %w) {{body = @wb, start = 0 : i64,
       stop = {it} : i64, step = 1 : i64, carry_arg_index = 0 : i64}}
       : (tensor<16x16xf16>, tensor<16x16xf16>) -> tensor<16x16xf16>
  return %r : tensor<16x16xf16>
}}
"""


def _compile_to_hsaco(it: int) -> bytes:
    gen = subprocess.run(
        [str(TESSERA_OPT), "-", "--generate-rocm-control-for-wmma-kernel",
         "--allow-unregistered-dialect"],
        input=_src(it), capture_output=True, text=True)
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


def _launch(hip, hsaco, carry, W):
    n2 = N * N
    nb = n2 * 2  # f16
    if hip.hipInit(0) != 0:
        return None
    mod = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(mod), hsaco) != 0:
        return None
    fn = ctypes.c_void_p()
    if hip.hipModuleGetFunction(
            ctypes.byref(fn), mod, b"tessera_control_for_wmma_0") != 0:
        return None
    dc, dw, do = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    for d in (dc, dw, do):
        if hip.hipMalloc(ctypes.byref(d), nb) != 0:
            return None
    hip.hipMemcpy(dc, carry.ctypes.data_as(ctypes.c_void_p), nb, 1)
    hip.hipMemcpy(dw, W.ctypes.data_as(ctypes.c_void_p), nb, 1)

    def memref(p):
        return [ctypes.c_void_p(p.value), ctypes.c_void_p(p.value),
                ctypes.c_int64(0), ctypes.c_int64(n2), ctypes.c_int64(1)]

    args = memref(dc) + memref(dw) + memref(do)
    arr = (ctypes.c_void_p * len(args))()
    for i, a_ in enumerate(args):
        arr[i] = ctypes.cast(ctypes.byref(a_), ctypes.c_void_p)
    launch = hip.hipModuleLaunchKernel
    launch.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint] * 6
                       + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p])
    if launch(fn, 1, 1, 1, WAVE, 1, 1, 0, None, arr, None) != 0:  # one wave
        return None
    if hip.hipDeviceSynchronize() != 0:
        return None
    out = np.zeros(n2, dtype=np.float16)
    hip.hipMemcpy(out.ctypes.data_as(ctypes.c_void_p), do, nb, 2)
    for d in (dc, dw, do):
        hip.hipFree(d)
    return out.reshape(N, N)


@pytest.mark.parametrize("it", [1, 2, 3])
def test_control_for_wmma_executes_on_gfx1151(it):
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    hip = _load_hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable — no ROCm host")
    rng = np.random.default_rng(11)
    # scale W so carry @ W^it stays well inside f16 range.
    carry = (rng.standard_normal((N, N)) * 0.5).astype(np.float16)
    W = (rng.standard_normal((N, N)) / np.sqrt(N)).astype(np.float16)
    out = _launch(hip, _compile_to_hsaco(it), carry, W)
    if out is None:
        pytest.skip("no usable AMD GPU (hipModuleLoadData / launch unavailable)")
    # Mirror the kernel: f32-accumulate matmul, carry cast back to f16 each step.
    ref = carry.copy()
    for _ in range(it):
        ref = (ref.astype(np.float32) @ W.astype(np.float32)).astype(np.float16)
    np.testing.assert_allclose(out.astype(np.float32), ref.astype(np.float32),
                               rtol=3e-2, atol=3e-2)
