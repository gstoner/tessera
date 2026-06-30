"""CF4d-2 — a CROSS-ELEMENT norm-in-loop Tessera control_for executes as ONE
cooperative-workgroup device kernel on gfx1151. GenerateROCMControlForNormKernel
lowers `control_for { carry = rmsnorm(carry) }` (or layer_norm) over a 1xK carry
to a single gpu.func: the carry lives in LDS, each thread computes the
normalization statistic over the whole LDS-resident carry and normalizes its own
element, and a gpu.barrier separates loop iterations. Like the GEMV kernel, the
statistic is a reduction over all carry elements — the per-thread elementwise
model can't express it.

  rmsnorm(x)    = x / sqrt(mean(x²) + eps)
  layer_norm(x) = (x - mean) / sqrt(mean((x-mean)²) + eps)
Looped a small number of iterations and compared to the same numpy formula
applied iteratively. Skip-clean off-GPU / without tessera-opt.
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


def _src(k: int, it: int, kind: str) -> str:
    return f"""
func.func @nb(%c: tensor<1x{k}xf32>) -> tensor<1x{k}xf32> {{
  %0 = "tessera.{kind}"(%c) {{eps = 1.000000e-05 : f64}} : (tensor<1x{k}xf32>) -> tensor<1x{k}xf32>
  return %0 : tensor<1x{k}xf32>
}}
func.func @f(%init: tensor<1x{k}xf32>) -> tensor<1x{k}xf32> {{
  %r = "tessera.control_for"(%init) {{body = @nb, start = 0 : i64,
       stop = {it} : i64, step = 1 : i64, carry_arg_index = 0 : i64}}
       : (tensor<1x{k}xf32>) -> tensor<1x{k}xf32>
  return %r : tensor<1x{k}xf32>
}}
"""


def _compile_to_hsaco(k: int, it: int, kind: str) -> bytes:
    gen = subprocess.run(
        [str(TESSERA_OPT), "-", "--generate-rocm-control-for-norm-kernel",
         "--allow-unregistered-dialect"],
        input=_src(k, it, kind), capture_output=True, text=True)
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


def _launch(hip, hsaco, carry):
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
        ctypes.byref(fn), mod, b"tessera_control_for_norm_0") == 0, \
        "kernel symbol tessera_control_for_norm_0 not found in module"
    dc, do = ctypes.c_void_p(), ctypes.c_void_p()
    for d in (dc, do):
        assert hip.hipMalloc(ctypes.byref(d), k * 4) == 0, "hipMalloc failed"
    hip.hipMemcpy(dc, carry.ctypes.data_as(ctypes.c_void_p), k * 4, 1)

    def memref(p):
        return [ctypes.c_void_p(p.value), ctypes.c_void_p(p.value),
                ctypes.c_int64(0), ctypes.c_int64(k), ctypes.c_int64(1)]

    args = memref(dc) + memref(do) + [ctypes.c_int64(k)]
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
    hip.hipMemcpy(out.ctypes.data_as(ctypes.c_void_p), do, k * 4, 2)
    for d in (dc, do):
        hip.hipFree(d)
    return out


def _rmsnorm(x):
    return x / np.sqrt(np.mean(x * x) + EPS)


def _layer_norm(x):
    mu = np.mean(x)
    return (x - mu) / np.sqrt(np.mean((x - mu) ** 2) + EPS)


@pytest.mark.parametrize("kind,k,it", [
    ("rmsnorm", 4, 2), ("rmsnorm", 16, 3), ("layer_norm", 8, 2)])
def test_control_for_norm_executes_on_gfx1151(kind, k, it):
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    hip = _load_hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable — no ROCm host")
    rng = np.random.default_rng(9)
    carry = (rng.standard_normal((1, k)) * 3.0).astype(np.float32)
    out = _launch(hip, _compile_to_hsaco(k, it, kind), carry)
    if out is None:
        # Only pre-launch discovery failed; a launch/sync failure raises.
        pytest.skip("no usable AMD GPU (hipInit / hipModuleLoadData failed)")
    fn = _rmsnorm if kind == "rmsnorm" else _layer_norm
    ref = carry.ravel().astype(np.float64)
    for _ in range(it):
        ref = fn(ref)
    np.testing.assert_allclose(out, ref, rtol=2e-3, atol=2e-3)
