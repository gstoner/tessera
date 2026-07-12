"""Spectral / FFT arbiter candidates — the ts-spectral-opt retarget.

Points the Spectral solver's ``lower-spectral-to-target-ir`` seam at the
Workstream-C D1 arbiter (Decision #28): the Target-IR call symbols that pass
emits (``ts_fft_stockham_cpu`` / ``ts_fft_stockham_amd``) are registered here as
first-class arbiter *candidates* for the ``spectral_fft`` op-kind, each F4-gated
against a ``numpy.fft`` reference.  Instead of the pass hard-wiring one symbol,
the arbiter enumerates the lanes for ``(op="spectral_fft", target)`` and picks
the fastest in-budget one — the crown-jewel HIP Stockham kernel stays a
first-class candidate, displaced only when a device_verified_jit lane measures faster and
in budget (lead-safety).

Two lanes, both running the *real shipped kernel* via ctypes:

* **CPU Stockham** (Tier 1, ``target="cpu"``) — compiles the shipped
  ``TargetHooks/CPU/StockhamRadix4.cpp`` and calls ``ts_fft_stockham_cpu``.
  Host-portable: it runs anywhere a C++ compiler is present, so the shipped
  kernel is F4-proven through the arbiter on any host.
* **ROCm Stockham** (Tier 3, ``target="rocm"``) — compiles the shipped
  ``TargetHooks/AMD/StockhamRadix4.hip`` (host-pointer wrapper) with ``hipcc``
  and runs it on a live gfx device; declines to the reference off-silicon so
  authoring/tests stay host-free.

Anywhere a lane cannot build/run it declines to ``region.reference`` and simply
drops out of arbitration — never a mislabeled kernel (Decision #21).
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from tessera.compiler.emit.candidate import (
    Candidate,
    Tier,
    register_candidate,
    register_op_kind,
    verify_by_reference,
)

#: op-kind tag emitted by LowerSpectralToTargetIR (tessera.target_ir.arbiter_op).
OP_SPECTRAL_FFT = "spectral_fft"

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CPU_SRC = (_REPO_ROOT / "src" / "solvers" / "spectral" / "lib" / "TargetHooks"
            / "CPU" / "StockhamRadix4.cpp")
_AMD_SRC = (_REPO_ROOT / "src" / "solvers" / "spectral" / "lib" / "TargetHooks"
            / "AMD" / "StockhamRadix4.hip")


# --- region + reference ------------------------------------------------------
class SpectralFFTRegion:
    """A 1-D complex FFT of length ``n``.  ``sign < 0`` forward (numpy.fft.fft),
    ``sign > 0`` inverse (numpy.fft.ifft, 1/N-scaled) — matching the sign/scale
    convention of the shipped Stockham kernels."""

    def __init__(self, n: int, sign: int = -1):
        self.n = int(n)
        self.sign = int(sign)

    def reference(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.complex64)
        y = np.fft.fft(x) if self.sign < 0 else np.fft.ifft(x)
        return y.astype(np.complex64)

    def probe_input(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return (rng.standard_normal(self.n) +
                1j * rng.standard_normal(self.n)).astype(np.complex64)


def _verify_fft(candidate: Candidate, region: SpectralFFTRegion, *,
                atol: float = 1e-3, seed: int = 0) -> bool:
    x = region.probe_input(seed)
    ref = region.reference(x)
    # fp32 FFT accumulates round-off ~ N * eps; scale the budget with N so a
    # correct kernel is not misread as a miscompile at large N.
    budget = max(atol, 1e-4 * region.n)
    return verify_by_reference(candidate, region, (x,), ref, atol=budget)


register_op_kind(OP_SPECTRAL_FFT, _verify_fft)


# --- compile cache (build the shipped kernel into a dlopen-able .so once) ----
_libs: dict[str, ctypes.CDLL | None] = {}


def _compile(key: str, argv: list[str], out: str) -> ctypes.CDLL | None:
    if key in _libs:
        return _libs[key]
    lib: ctypes.CDLL | None = None
    try:
        subprocess.check_call(argv, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
        lib = ctypes.CDLL(out)
    except Exception:
        lib = None
    _libs[key] = lib
    return lib


def _cpu_lib() -> ctypes.CDLL | None:
    cxx = os.environ.get("CXX", "c++")
    if not shutil.which(cxx) or not _CPU_SRC.exists():
        _libs["cpu"] = _libs.get("cpu")
        return _libs.get("cpu")
    d = tempfile.mkdtemp(prefix="tessera_spectral_cpu_")
    so = os.path.join(d, "libspectral_cpu.so")
    lib = _compile("cpu", [cxx, "-O2", "-std=c++17", "-shared", "-fPIC",
                          str(_CPU_SRC), "-o", so], so)
    if lib is not None:
        lib.ts_fft_stockham_cpu.restype = None
        lib.ts_fft_stockham_cpu.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    return lib


def _amd_lib() -> ctypes.CDLL | None:
    if not shutil.which("hipcc") or not _AMD_SRC.exists():
        _libs["amd"] = _libs.get("amd")
        return _libs.get("amd")
    arch = os.environ.get("TESSERA_ROCM_ARCH", "gfx1151")
    d = tempfile.mkdtemp(prefix="tessera_spectral_amd_")
    so = os.path.join(d, "libspectral_amd.so")
    lib = _compile("amd", ["hipcc", f"--offload-arch={arch}", "-O3",
                          "-std=c++17", "-shared", "-fPIC", str(_AMD_SRC),
                          "-o", so], so)
    if lib is not None and hasattr(lib, "ts_fft_stockham_amd_hostptr"):
        lib.ts_fft_stockham_amd_hostptr.restype = ctypes.c_int
        lib.ts_fft_stockham_amd_hostptr.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    return lib


def _cptr(a: np.ndarray) -> ctypes.c_void_p:
    return a.ctypes.data_as(ctypes.c_void_p)


# --- candidates --------------------------------------------------------------
class CpuStockhamFFTCandidate(Candidate):
    """Tier-1: the shipped CPU mixed-radix Stockham kernel, device_verified_jit + dlopened.
    Runs the real ``ts_fft_stockham_cpu`` symbol the Target-IR lowering names."""

    name = "cpu_stockham"
    tier = Tier.SYNTHESIZED
    target = "cpu"
    op = OP_SPECTRAL_FFT

    def available(self) -> bool:
        return _cpu_lib() is not None

    def run(self, region: SpectralFFTRegion, x: np.ndarray, *a: Any,
            **k: Any) -> tuple[Any, str]:
        lib = _cpu_lib()
        if lib is None:
            return region.reference(x), "reference"
        try:
            xin = np.ascontiguousarray(x, np.complex64)
            out = np.empty(region.n, np.complex64)
            lib.ts_fft_stockham_cpu(_cptr(xin), _cptr(out), region.n,
                                    region.sign)
            return out, "cpu_stockham"
        except Exception:
            return region.reference(x), "reference"


class RocmStockhamFFTCandidate(Candidate):
    """Tier-3: the shipped ROCm Stockham kernel on a live gfx device (crown-jewel
    lead candidate).  Declines off-silicon so tests stay host-free."""

    name = "rocm_stockham"
    tier = Tier.HAND_TUNED
    target = "rocm"
    op = OP_SPECTRAL_FFT

    def available(self) -> bool:
        # Needs hipcc AND a usable device; the host-pointer wrapper returns 0 on
        # success.  Probe cheaply: compile ok + a tiny transform round-trips.
        lib = _amd_lib()
        if lib is None or not hasattr(lib, "ts_fft_stockham_amd_hostptr"):
            return False
        try:
            x = np.ones(4, np.complex64)
            out = np.empty(4, np.complex64)
            rc = lib.ts_fft_stockham_amd_hostptr(_cptr(x), _cptr(out), 4, -1)
            return rc == 0
        except Exception:
            return False

    def run(self, region: SpectralFFTRegion, x: np.ndarray, *a: Any,
            **k: Any) -> tuple[Any, str]:
        lib = _amd_lib()
        if lib is None:
            return region.reference(x), "reference"
        try:
            xin = np.ascontiguousarray(x, np.complex64)
            out = np.empty(region.n, np.complex64)
            rc = lib.ts_fft_stockham_amd_hostptr(_cptr(xin), _cptr(out),
                                                 region.n, region.sign)
            if rc != 0:
                return region.reference(x), "reference"
            return out, "rocm_stockham"
        except Exception:
            return region.reference(x), "reference"


register_candidate(CpuStockhamFFTCandidate())
register_candidate(RocmStockhamFFTCandidate())
