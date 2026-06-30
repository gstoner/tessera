"""SD1 — tessera.spec_accept (greedy speculative-decode acceptance) executes as
ONE cooperative-workgroup device kernel on gfx1151. GenerateROCMSpecAcceptKernel
lowers `spec_accept(draft P×D, target P×(D+1)) -> [path_idx, prefix_length,
bonus]` to a single gpu.func: thread p computes its path's run of leading matches
(draft[p,i]==target[p,i]) into LDS, a barrier, then thread 0 does the cross-path
argmax (longest prefix, first wins ties) and reads the bonus target[path, length].

This is the ROCm analogue of the Apple-GPU MSL proof; the numpy oracle is the same
_ref_spec_accept the tessera.speculative tests use. Deterministic (no RNG) →
bit-exact. Skip-clean off-GPU / without tessera-opt.
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


# The greedy reference the tessera.speculative tests use (test_apple_gpu_control_
# flow.py:_ref_spec_accept) — longest accepted prefix, first wins ties, bonus is
# target[path, length].
def _ref_spec_accept(draft, target):
    P, depth = draft.shape
    bp, bl, bb = 0, -1, 0
    for p in range(P):
        length = 0
        for i in range(depth):
            if int(draft[p, i]) == int(target[p, i]):
                length += 1
            else:
                break
        if length > bl:
            bl, bp = length, p
            bb = int(target[p, length])
    return bp, bl, bb


def _src(P: int, D: int) -> str:
    d = f"tensor<{P}x{D}xi32>"
    t = f"tensor<{P}x{D + 1}xi32>"
    return f"""
func.func @f(%d: {d}, %t: {t}) -> tensor<3xi32> {{
  %r = "tessera.spec_accept"(%d, %t) : ({d}, {t}) -> tensor<3xi32>
  return %r : tensor<3xi32>
}}
"""


def _compile_to_hsaco(P: int, D: int) -> bytes:
    gen = subprocess.run(
        [str(TESSERA_OPT), "-", "--generate-rocm-spec-accept-kernel"],
        input=_src(P, D), capture_output=True, text=True)
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


def _launch(hip, hsaco, draft, target):
    P, D = draft.shape
    nb_d = P * D * 4
    nb_t = P * (D + 1) * 4
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
        ctypes.byref(fn), mod, b"tessera_spec_accept_0") == 0, \
        "kernel symbol tessera_spec_accept_0 not found in module"
    dd, dt, do = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    assert hip.hipMalloc(ctypes.byref(dd), nb_d) == 0, "hipMalloc(draft) failed"
    assert hip.hipMalloc(ctypes.byref(dt), nb_t) == 0, "hipMalloc(target) failed"
    assert hip.hipMalloc(ctypes.byref(do), 3 * 4) == 0, "hipMalloc(out) failed"
    hip.hipMemcpy(dd, draft.ctypes.data_as(ctypes.c_void_p), nb_d, 1)
    hip.hipMemcpy(dt, target.ctypes.data_as(ctypes.c_void_p), nb_t, 1)

    def memref(p, n):
        return [ctypes.c_void_p(p.value), ctypes.c_void_p(p.value),
                ctypes.c_int64(0), ctypes.c_int64(n), ctypes.c_int64(1)]

    args = memref(dd, P * D) + memref(dt, P * (D + 1)) + memref(do, 3)
    arr = (ctypes.c_void_p * len(args))()
    for i, a_ in enumerate(args):
        arr[i] = ctypes.cast(ctypes.byref(a_), ctypes.c_void_p)
    launch = hip.hipModuleLaunchKernel
    launch.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint] * 6
                       + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p])
    # one workgroup of P threads (P ≤ 256).
    block = max(P, 1)
    rc = launch(fn, 1, 1, 1, block, 1, 1, 0, None, arr, None)
    assert rc == 0, (
        f"hipModuleLaunchKernel failed (rc={rc}) for P={P} — kernel failed to "
        "launch on a working GPU")
    rc = hip.hipDeviceSynchronize()
    assert rc == 0, f"hipDeviceSynchronize failed (rc={rc}) for P={P} — crashed"
    out = np.zeros(3, dtype=np.int32)
    hip.hipMemcpy(out.ctypes.data_as(ctypes.c_void_p), do, 3 * 4, 2)
    for d in (dd, dt, do):
        hip.hipFree(d)
    return out


def _cases():
    # The fixed examples from the Apple-GPU proof (same oracle).
    yield (np.array([[7, 3, 9, 1], [7, 8, 2, 4], [7, 3, 5, 6]], np.int32),
           np.array([[7, 3, 0, 0, 0], [7, 0, 0, 0, 0], [7, 3, 5, 0, 9]], np.int32))
    yield (np.array([[1, 2, 3]], np.int32),
           np.array([[1, 2, 3, 9]], np.int32))  # full accept → bonus 9


@pytest.mark.parametrize("draft,target", list(_cases()))
def test_spec_accept_fixed_on_gfx1151(draft, target):
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    hip = _load_hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable — no ROCm host")
    P, D = draft.shape
    out = _launch(hip, _compile_to_hsaco(P, D), draft, target)
    if out is None:
        pytest.skip("no usable AMD GPU (hipInit / hipModuleLoadData failed)")
    assert tuple(int(x) for x in out) == _ref_spec_accept(draft, target)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_spec_accept_random_on_gfx1151(seed):
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    hip = _load_hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable — no ROCm host")
    rng = np.random.default_rng(seed)
    P, D, V = 8, 7, 16
    draft = rng.integers(0, V, size=(P, D), dtype=np.int32)
    target = rng.integers(0, V, size=(P, D + 1), dtype=np.int32)
    # share the first column so accepted lengths vary across paths.
    target[:, 0] = draft[:, 0]
    out = _launch(hip, _compile_to_hsaco(P, D), draft, target)
    if out is None:
        pytest.skip("no usable AMD GPU (hipInit / hipModuleLoadData failed)")
    assert tuple(int(x) for x in out) == _ref_spec_accept(draft, target)
