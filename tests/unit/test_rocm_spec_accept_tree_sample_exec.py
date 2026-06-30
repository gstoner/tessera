"""SD1 (tree multi-path rejection) — tessera.spec_accept_tree_sample executes as
ONE device kernel on gfx1151. GenerateROCMSpecAcceptTreeSampleKernel lowers the
device form of tessera.speculative.batch_verify: over a draft tree of P paths each
D deep, the draft token at (p,i) is accepted iff
    accept_u[p,i] <= exp(target_log_probs[p,i] - draft_log_probs[p,i])
(the division-free Leviathan rule); each path's accepted prefix is the leading run
of accepts; the kernel picks the longest-prefix path (first wins ties) and writes
[accepted_path_idx, accepted_prefix_length]. RNG is explicit (accept_u is an
operand) → deterministic, device-bit-exact. The numpy oracle mirrors batch_verify
with the same explicit uniforms + f32 exp. Skip-clean off-GPU / without
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


def _ref(tlp, dlp, au):
    """Device-matching oracle: accept iff au <= exp(tlp - dlp) (f32), longest
    accepted prefix per path, first-wins-ties path select."""
    P, D = tlp.shape
    ap = np.exp((tlp - dlp).astype(np.float32))   # f32 exp, like the kernel
    accept = au <= ap
    best_p, best_len = 0, -1
    for p in range(P):
        length = 0
        for i in range(D):
            if bool(accept[p, i]):
                length += 1
            else:
                break
        if length > best_len:
            best_len, best_p = length, p
    return np.array([best_p, best_len], np.int32)


def _src(P: int, D: int) -> str:
    t = f"tensor<{P}x{D}xf32>"
    return f"""
func.func @f(%t: {t}, %d: {t}, %u: {t}) -> tensor<2xi32> {{
  %r = "tessera.spec_accept_tree_sample"(%t, %d, %u) : ({t}, {t}, {t}) -> tensor<2xi32>
  return %r : tensor<2xi32>
}}
"""


def _compile_to_hsaco(P: int, D: int) -> bytes:
    gen = subprocess.run(
        [str(TESSERA_OPT), "-", "--generate-rocm-spec-accept-tree-sample-kernel"],
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


def _launch(hip, hsaco, tlp, dlp, au):
    P, D = tlp.shape
    nb = P * D * 4
    if hip.hipInit(0) != 0:
        return None
    mod = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(mod), hsaco) != 0:
        return None
    fn = ctypes.c_void_p()
    assert hip.hipModuleGetFunction(
        ctypes.byref(fn), mod, b"tessera_spec_accept_tree_sample_0") == 0, \
        "kernel symbol tessera_spec_accept_tree_sample_0 not found"
    dt, dd, du, do = (ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p(),
                      ctypes.c_void_p())
    assert hip.hipMalloc(ctypes.byref(dt), nb) == 0, "hipMalloc(tlp) failed"
    assert hip.hipMalloc(ctypes.byref(dd), nb) == 0, "hipMalloc(dlp) failed"
    assert hip.hipMalloc(ctypes.byref(du), nb) == 0, "hipMalloc(au) failed"
    assert hip.hipMalloc(ctypes.byref(do), 2 * 4) == 0, "hipMalloc(out) failed"
    for dev, host in ((dt, tlp), (dd, dlp), (du, au)):
        hip.hipMemcpy(dev, np.ascontiguousarray(host, np.float32).ctypes.data_as(
            ctypes.c_void_p), nb, 1)

    def memref(p, n):
        return [ctypes.c_void_p(p.value), ctypes.c_void_p(p.value),
                ctypes.c_int64(0), ctypes.c_int64(n), ctypes.c_int64(1)]

    args = memref(dt, P * D) + memref(dd, P * D) + memref(du, P * D) + memref(do, 2)
    arr = (ctypes.c_void_p * len(args))()
    for i, a_ in enumerate(args):
        arr[i] = ctypes.cast(ctypes.byref(a_), ctypes.c_void_p)
    launch = hip.hipModuleLaunchKernel
    launch.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint] * 6
                       + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p])
    rc = launch(fn, 1, 1, 1, max(P, 1), 1, 1, 0, None, arr, None)
    assert rc == 0, f"hipModuleLaunchKernel failed (rc={rc})"
    rc = hip.hipDeviceSynchronize()
    assert rc == 0, f"hipDeviceSynchronize failed (rc={rc})"
    out = np.zeros(2, dtype=np.int32)
    hip.hipMemcpy(out.ctypes.data_as(ctypes.c_void_p), do, 2 * 4, 2)
    for d in (dt, dd, du, do):
        hip.hipFree(d)
    return out


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_spec_accept_tree_sample_on_gfx1151(seed):
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    hip = _load_hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable — no ROCm host")
    rng = np.random.default_rng(seed)
    P, D = 6, 5
    # target/draft log-probs of the chosen draft token at each (path, step).
    tlp = (rng.standard_normal((P, D)) - 1.0).astype(np.float32)
    dlp = (rng.standard_normal((P, D)) - 1.0).astype(np.float32)
    # keep u away from the accept boundary so f32-exp last-bit diffs never flip it.
    au = rng.uniform(0.02, 0.98, size=(P, D)).astype(np.float32)
    out = _launch(hip, _compile_to_hsaco(P, D), tlp, dlp, au)
    if out is None:
        pytest.skip("no usable AMD GPU (hipInit / hipModuleLoadData failed)")
    np.testing.assert_array_equal(out, _ref(tlp, dlp, au))
