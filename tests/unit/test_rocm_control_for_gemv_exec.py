"""CF4d-1 — a CROSS-ELEMENT Tessera control_for (GEMV recurrence) executes as ONE
cooperative-workgroup device kernel on gfx1151. GenerateROCMControlForGemvKernel
lowers `control_for { carry = carry @ W }` (carry a 1xK vector, W a KxK matrix)
to a single gpu.func: the carry lives in LDS, thread j computes
o[j] = Σ_k carry[k]·W[k][j] by a serial dot product, and a gpu.barrier separates
loop iterations. The whole bounded loop is one dispatch → carry @ W^max_iters.

This is the first body the per-thread elementwise kernels can't express (each
output needs the whole carry). Skip-clean when tessera-opt isn't built or no
usable AMD GPU is present.
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


def _src(k: int, it: int) -> str:
    return f"""
func.func @wb(%c: tensor<1x{k}xf32>, %w: tensor<{k}x{k}xf32>) -> tensor<1x{k}xf32> {{
  %0 = "tessera.matmul"(%c, %w) : (tensor<1x{k}xf32>, tensor<{k}x{k}xf32>) -> tensor<1x{k}xf32>
  return %0 : tensor<1x{k}xf32>
}}
func.func @f(%init: tensor<1x{k}xf32>, %w: tensor<{k}x{k}xf32>) -> tensor<1x{k}xf32> {{
  %r = "tessera.control_for"(%init, %w) {{body = @wb, start = 0 : i64,
       stop = {it} : i64, step = 1 : i64, carry_arg_index = 0 : i64}}
       : (tensor<1x{k}xf32>, tensor<{k}x{k}xf32>) -> tensor<1x{k}xf32>
  return %r : tensor<1x{k}xf32>
}}
"""


def _compile_to_hsaco(k: int, it: int) -> bytes:
    gen = subprocess.run(
        [str(TESSERA_OPT), "-", "--generate-rocm-control-for-gemv-kernel",
         "--allow-unregistered-dialect"],
        input=_src(k, it), capture_output=True, text=True)
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
    """Run @tessera_control_for_gemv_0(CARRY, W, OUT, K). Returns OUT, or None."""
    k = carry.size
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
        ctypes.byref(fn), mod, b"tessera_control_for_gemv_0") == 0, \
        "kernel symbol tessera_control_for_gemv_0 not found in module"
    dc, dw, do = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    assert (hip.hipMalloc(ctypes.byref(dc), k * 4) == 0
            and hip.hipMalloc(ctypes.byref(dw), k * k * 4) == 0
            and hip.hipMalloc(ctypes.byref(do), k * 4) == 0), "hipMalloc failed"
    hip.hipMemcpy(dc, carry.ctypes.data_as(ctypes.c_void_p), k * 4, 1)
    hip.hipMemcpy(dw, W.ctypes.data_as(ctypes.c_void_p), k * k * 4, 1)

    def memref(p, sz):
        return [ctypes.c_void_p(p.value), ctypes.c_void_p(p.value),
                ctypes.c_int64(0), ctypes.c_int64(sz), ctypes.c_int64(1)]

    args = (memref(dc, k) + memref(dw, k * k) + memref(do, k)
            + [ctypes.c_int64(k)])
    arr = (ctypes.c_void_p * len(args))()
    for i, a_ in enumerate(args):
        arr[i] = ctypes.cast(ctypes.byref(a_), ctypes.c_void_p)
    launch = hip.hipModuleLaunchKernel
    launch.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint] * 6
                       + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p])
    # one workgroup (single carry), BD threads.
    rc = launch(fn, 1, 1, 1, BD, 1, 1, 0, None, arr, None)
    assert rc == 0, (
        f"hipModuleLaunchKernel failed (rc={rc}) for k={k} — kernel failed to "
        "launch on a working GPU")
    rc = hip.hipDeviceSynchronize()
    assert rc == 0, f"hipDeviceSynchronize failed (rc={rc}) for k={k} — crashed"
    out = np.zeros(k, dtype=np.float32)
    hip.hipMemcpy(out.ctypes.data_as(ctypes.c_void_p), do, k * 4, 2)
    for d in (dc, dw, do):
        hip.hipFree(d)
    return out


@pytest.mark.parametrize("k,it", [(4, 3), (16, 5), (64, 2)])
def test_control_for_gemv_executes_on_gfx1151(k, it):
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    hip = _load_hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable — no ROCm host")
    rng = np.random.default_rng(7)
    carry = rng.standard_normal((1, k)).astype(np.float32)
    # scale W so carry @ W^it stays finite for the larger it.
    W = (rng.standard_normal((k, k)) / np.sqrt(k)).astype(np.float32)
    out = _launch(hip, _compile_to_hsaco(k, it), carry, W)
    if out is None:
        # Only pre-launch discovery failed; a launch/sync failure raises.
        pytest.skip("no usable AMD GPU (hipInit / hipModuleLoadData failed)")
    ref = carry.copy()
    for _ in range(it):
        ref = ref @ W            # carry @ W^it
    np.testing.assert_allclose(out, ref.ravel(), rtol=2e-3, atol=2e-3)
