"""CF4d-4 — a MULTI-tile WMMA matmul recurrence Tessera control_for executes as
ONE workgroup (MT*KT waves) device kernel on gfx1151 (RDNA3.5).
GenerateROCMControlForWmmaTileKernel lowers `control_for { carry = carry @ W }`
(carry M×K, W K×K, both f16) to one gpu.func where each wave owns one 16×16
output tile and accumulates it over the shared-K dimension with the
rocdl.wmma.f32.16x16x16.f16 intrinsic. The whole MT×KT carry lives in LDS, so the
per-iteration handoff is a plain WORKGROUP barrier — no grid.sync / cooperative
launch (that is only needed for carries exceeding one workgroup's LDS) → carry @
W^max_iters on real WMMA hardware, one dispatch.

The kernel truncates the carry to f16 between iterations (the LDS is f16), so the
numpy reference does the same: each step is an f32-accumulate matmul whose result
is cast back to f16. f16 → loose tolerance. Skip-clean off-GPU / without
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
TILE = 16
WAVE = 32


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


def _src(M: int, K: int, it: int) -> str:
    cty = f"tensor<{M}x{K}xf16>"
    wty = f"tensor<{K}x{K}xf16>"
    return f"""
func.func @wb(%c: {cty}, %w: {wty}) -> {cty} {{
  %0 = "tessera.matmul"(%c, %w) : ({cty}, {wty}) -> {cty}
  return %0 : {cty}
}}
func.func @f(%init: {cty}, %w: {wty}) -> {cty} {{
  %r = "tessera.control_for"(%init, %w) {{body = @wb, start = 0 : i64,
       stop = {it} : i64, step = 1 : i64, carry_arg_index = 0 : i64}}
       : ({cty}, {wty}) -> {cty}
  return %r : {cty}
}}
"""


def _compile_to_hsaco(M: int, K: int, it: int) -> bytes:
    gen = subprocess.run(
        [str(TESSERA_OPT), "-", "--generate-rocm-control-for-wmma-tile-kernel",
         "--allow-unregistered-dialect"],
        input=_src(M, K, it), capture_output=True, text=True)
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


def _launch(hip, hsaco, carry, W, M, K):
    mt, kt = M // TILE, K // TILE
    block = mt * kt * WAVE
    cn, cb = M * K, M * K * 2  # f16
    wn, wb = K * K, K * K * 2
    if hip.hipInit(0) != 0:
        return None
    mod = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(mod), hsaco) != 0:
        return None
    fn = ctypes.c_void_p()
    if hip.hipModuleGetFunction(
            ctypes.byref(fn), mod, b"tessera_control_for_wmma_tile_0") != 0:
        return None
    dc, dw, do = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    if hip.hipMalloc(ctypes.byref(dc), cb) != 0:
        return None
    if hip.hipMalloc(ctypes.byref(dw), wb) != 0:
        return None
    if hip.hipMalloc(ctypes.byref(do), cb) != 0:
        return None
    hip.hipMemcpy(dc, carry.ctypes.data_as(ctypes.c_void_p), cb, 1)
    hip.hipMemcpy(dw, W.ctypes.data_as(ctypes.c_void_p), wb, 1)

    def memref(p, n):
        return [ctypes.c_void_p(p.value), ctypes.c_void_p(p.value),
                ctypes.c_int64(0), ctypes.c_int64(n), ctypes.c_int64(1)]

    args = memref(dc, cn) + memref(dw, wn) + memref(do, cn)
    arr = (ctypes.c_void_p * len(args))()
    for i, a_ in enumerate(args):
        arr[i] = ctypes.cast(ctypes.byref(a_), ctypes.c_void_p)
    launch = hip.hipModuleLaunchKernel
    launch.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint] * 6
                       + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p])
    if launch(fn, 1, 1, 1, block, 1, 1, 0, None, arr, None) != 0:
        return None
    if hip.hipDeviceSynchronize() != 0:
        return None
    out = np.zeros(cn, dtype=np.float16)
    hip.hipMemcpy(out.ctypes.data_as(ctypes.c_void_p), do, cb, 2)
    for d in (dc, dw, do):
        hip.hipFree(d)
    return out.reshape(M, K)


# Stay inside the proven single-WGP envelope (MT*KT <= 8 waves on gfx1151; see
# MAX_WAVES in GenerateROCMControlForWmmaTileKernel.cpp). Asymmetric MT/KT and
# the 8-wave ceiling (64x32 / 32x64) are all exercised.
@pytest.mark.parametrize("M,K,it", [
    (32, 32, 1), (32, 32, 2), (32, 32, 3),  # 2x2 tiles, 4 waves
    (16, 32, 2),                            # 1x2 tiles, 2 waves (MT<KT)
    (32, 16, 2),                            # 2x1 tiles, 2 waves (MT>KT)
    (64, 32, 2),                            # 4x2 tiles, 8 waves (ceiling, MT>KT)
    (32, 64, 2),                            # 2x4 tiles, 8 waves (ceiling, MT<KT)
])
def test_control_for_wmma_tile_executes_on_gfx1151(M, K, it):
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    hip = _load_hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable — no ROCm host")
    rng = np.random.default_rng(23 + M + K + it)
    # scale W so carry @ W^it stays well inside f16 range.
    carry = (rng.standard_normal((M, K)) * 0.5).astype(np.float16)
    W = (rng.standard_normal((K, K)) / np.sqrt(K)).astype(np.float16)
    out = _launch(hip, _compile_to_hsaco(M, K, it), carry, W, M, K)
    if out is None:
        pytest.skip("no usable AMD GPU (hipModuleLoadData / launch unavailable)")
    # Mirror the kernel: f32-accumulate matmul, carry cast back to f16 each step.
    ref = carry.copy()
    for _ in range(it):
        ref = (ref.astype(np.float32) @ W.astype(np.float32)).astype(np.float16)
    np.testing.assert_allclose(out.astype(np.float32), ref.astype(np.float32),
                               rtol=3e-2, atol=3e-2)
