"""Spectral / FFT arbiter candidates — the ts-spectral-opt retarget.

Points the Spectral solver's ``lower-spectral-to-target-ir`` seam at the
Workstream-C D1 arbiter (Decision #28): the Target-IR call symbols that pass
emits (``ts_fft_stockham_cpu`` / ``ts_fft_stockham_amd``) are registered here as
first-class arbiter *candidates* for the ``spectral_fft`` op-kind, each F4-gated
against a ``numpy.fft`` reference.  Instead of the pass hard-wiring one symbol,
the arbiter enumerates the lanes for ``(op="spectral_fft", target)`` and picks
the fastest in-budget one — the crown-jewel HIP Stockham kernel stays a
first-class candidate, displaced only when a compiled lane measures faster and
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
    on_candidate_registered,
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


def _supported_length(n: int) -> bool:
    """Whether the shipped CPU / ROCm kernels can transform a length-`n` signal.

    Every positive `n`, since the mixed-radix + Bluestein build-out. History,
    because the shape of the previous bug is worth keeping:

    The kernels were radix-4 with a radix-2 tail, handled only powers of two,
    and did NOT validate `n` -- the driver drained factors of 4 and 2 and
    stopped, returning a partially-transformed buffer under its own lane name.
    Measured against `numpy.fft`: powers of two from 1 to 1024 agreed to ~1e-7
    while 3, 12, 24, 48, 100, 255 and 257 came back with relative error ~1.0. A
    different answer, recorded by the arbiter as a successful kernel run.

    Now the driver plans first (shared `TargetHooks/Common/FFTPlan.h`): factors
    within the radix set run mixed-radix stages, and anything else -- large
    primes included -- goes through Bluestein over a power-of-two convolution.
    Verified against numpy across 63 sizes on CPU and 48 on live gfx1151.

    Kept as a named predicate rather than deleted: `applies_to` is the arbiter's
    hook for exactly this question, and a future target whose kernel is narrower
    (the NVIDIA hook has no Bluestein yet) needs somewhere to say so.
    """
    return n > 0


# --- candidates --------------------------------------------------------------
class CpuStockhamFFTCandidate(Candidate):
    """Tier-1: the shipped CPU mixed-radix Stockham kernel, compiled + dlopened.
    Runs the real ``ts_fft_stockham_cpu`` symbol the Target-IR lowering names."""

    name = "cpu_stockham"
    tier = Tier.SYNTHESIZED
    target = "cpu"
    op = OP_SPECTRAL_FFT

    def available(self) -> bool:
        return _cpu_lib() is not None

    def applies_to(self, region: Any) -> bool:
        return _supported_length(getattr(region, "n", 0))

    def run(self, region: SpectralFFTRegion, x: np.ndarray, *a: Any,
            **k: Any) -> tuple[Any, str]:
        lib = _cpu_lib()
        if lib is None or not _supported_length(region.n):
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

    def applies_to(self, region: Any) -> bool:
        return _supported_length(getattr(region, "n", 0))

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
        if lib is None or not _supported_length(region.n):
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


# ═════════════════════════════════════════════════════════════════════════════
# TSOL spectral family — the four remaining ops, COMPOSED over the FFT region
#
# `rfft` / `irfft` / `stft` / `istft` are not new kernels. Each decomposes to
# the complex FFT the shipped Stockham lanes already implement, so a candidate
# here delegates its inner transform to whatever `spectral_fft` candidate is
# registered for the same target. Two consequences worth being explicit about:
#
#   * One FFT lane per foundation lights up FIVE TSOL ops on that foundation.
#     When the NVIDIA `.cu` (written, not yet registered) and an Apple lane land
#     as `spectral_fft` candidates, these compose on top with no further work.
#   * A composed candidate is never "more available" than its inner FFT. If no
#     FFT lane can build or run for a target, the composition declines to the
#     reference exactly as the FFT candidate does (Decision #21 -- never a
#     mislabeled kernel).
#
# `spectral_filter` is deliberately NOT composed over the FFT: it is a pointwise
# product of two spectra (`Xf * Hf`), with no transform inside. Filing it with
# the transforms because its operands happen to be spectra would be the same
# one-reason-for-N-ops error this registry has already been bitten by twice.
# ═════════════════════════════════════════════════════════════════════════════

OP_SPECTRAL_RFFT = "spectral_rfft"
OP_SPECTRAL_IRFFT = "spectral_irfft"
OP_SPECTRAL_STFT = "spectral_stft"
OP_SPECTRAL_ISTFT = "spectral_istft"
OP_SPECTRAL_FILTER = "spectral_filter"


def _inner_fft(target: str, n: int, sign: int, x: np.ndarray) -> tuple[np.ndarray, str]:
    """Run the best available `spectral_fft` lane for `target`.

    Returns `(values, lane)`, where `lane` is `"reference"` when no registered
    FFT candidate could run -- so a composed op reports honestly which lane
    actually produced its inner transform rather than claiming its own name.
    """
    from tessera.compiler.emit.candidate import candidates_for

    region = SpectralFFTRegion(n=n, sign=sign)
    for cand in candidates_for(target, OP_SPECTRAL_FFT):
        try:
            if not cand.applies_to(region) or not cand.available():
                continue
            out, lane = cand.run(region, x)
            if lane != "reference":
                return np.asarray(out, np.complex64), lane
        except Exception:
            continue
    return region.reference(x), "reference"


class SpectralRFFTRegion:
    """Real-input FFT of length `n` -> the `n//2 + 1` non-redundant bins.

    A real signal's transform is conjugate-symmetric, so only half the spectrum
    carries information. The composition runs the full complex transform and
    keeps that half -- correct, and it reuses the shipped kernel rather than
    introducing a second FFT implementation to keep in step (Decision #31).
    """

    def __init__(self, n: int):
        self.n = int(n)

    @property
    def bins(self) -> int:
        return self.n // 2 + 1

    def reference(self, x: np.ndarray) -> np.ndarray:
        return np.fft.rfft(np.asarray(x, np.float32)).astype(np.complex64)

    def probe_input(self, seed: int) -> np.ndarray:
        return np.random.default_rng(seed).standard_normal(self.n).astype(np.float32)


class SpectralIRFFTRegion:
    """Half-spectrum -> real signal of length `n`; the inverse of `rfft`.

    `n` is explicit rather than inferred: `bins = n//2 + 1` maps two lengths
    (2*bins-2 and 2*bins-1) onto the same bin count, so an odd-length original
    is NOT recoverable from the spectrum alone. That ambiguity is exactly why
    `np.fft.irfft` takes an `n`, and why the W1 shape rule for `irfft` reads it.
    """

    def __init__(self, n: int):
        self.n = int(n)

    @property
    def bins(self) -> int:
        return self.n // 2 + 1

    def reference(self, xf: np.ndarray) -> np.ndarray:
        return np.fft.irfft(np.asarray(xf, np.complex64), n=self.n).astype(np.float32)

    def probe_input(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return np.fft.rfft(rng.standard_normal(self.n).astype(np.float32)).astype(np.complex64)


def _probe_window(length: int, seed: int) -> np.ndarray:
    """A deliberately NON-Hann probe window.

    `stft(x, win, hop)` takes the window as DATA. An earlier version of these
    regions manufactured `np.hanning(win)` internally and the candidates ignored
    the operand, so a caller's rectangular or custom window was silently
    computed as Hann -- and because the reference manufactured the same window,
    verification compared wrong against wrong and passed.

    So the probe window must be one no implementation would invent: asymmetric,
    not a standard taper, and not separable from the signal. If a lane
    manufactures its own, this fails immediately. Probing at a configuration
    where a right and a wrong implementation agree is the specific gate failure
    this whole thread keeps rediscovering.
    """
    rng = np.random.default_rng(seed + 9973)
    return (0.25 + rng.random(length)).astype(np.float32)


class SpectralSTFTRegion:
    """Framed short-time transform: `(frames, win//2 + 1)`.

    Composes over `rfft` per frame, matching `tessera.ops.stft`'s framing --
    `frames = (n - win) // hop + 1`, window applied before the transform.

    `win` here is the window LENGTH, which is a shape parameter the plan needs.
    The window VALUES are an operand and travel with the call.
    """

    def __init__(self, n: int, win: int, hop: int):
        self.n, self.win, self.hop = int(n), int(win), int(hop)

    @property
    def frames(self) -> int:
        return max(1, (self.n - self.win) // self.hop + 1)

    def reference(self, x: np.ndarray, win: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float32)
        w = np.asarray(win, np.float32)
        out = [np.fft.rfft(x[s:s + self.win] * w)
               for s in range(0, max(1, self.n - self.win + 1), self.hop)]
        return np.stack(out, axis=-2).astype(np.complex64)

    def probe_input(self, seed: int):
        sig = np.random.default_rng(seed).standard_normal(self.n).astype(np.float32)
        return sig, _probe_window(self.win, seed)


class SpectralISTFTRegion:
    """Overlap-add inverse of `stft` -> `(frames - 1) * hop + win` samples.

    The f64 accumulator is deliberate and matches the reference: overlap-add
    sums many windowed frames, and the accumulation width is a separate
    decision from the RESULT width (Decision #15a).
    """

    def __init__(self, frames: int, win: int, hop: int):
        self.frames, self.win, self.hop = int(frames), int(win), int(hop)

    @property
    def samples(self) -> int:
        return (self.frames - 1) * self.hop + self.win

    def reference(self, xf: np.ndarray, win: np.ndarray) -> np.ndarray:
        xf = np.asarray(xf, np.complex64)
        w = np.asarray(win, np.float32)
        out = np.zeros(self.samples, np.float64)
        weight = np.zeros_like(out)
        for i in range(self.frames):
            frame = np.fft.irfft(xf[i], n=self.win) * w
            s = i * self.hop
            out[s:s + self.win] += frame
            weight[s:s + self.win] += w * w
        return (out / np.maximum(weight, 1e-12)).astype(np.float32)

    def probe_input(self, seed: int):
        rng = np.random.default_rng(seed)
        sig = rng.standard_normal(self.samples).astype(np.float32)
        w = _probe_window(self.win, seed)
        spec = np.stack(
            [np.fft.rfft(sig[i * self.hop:i * self.hop + self.win] * w)
             for i in range(self.frames)], axis=-2).astype(np.complex64)
        return spec, w


class SpectralFilterRegion:
    """Pointwise product of two spectra — `Xf * Hf`. No transform inside.

    Kept apart from the transform family on purpose: it operates ON spectra
    rather than producing them, so composing it over the FFT region would be a
    wrong decomposition dressed as consistency.
    """

    def __init__(self, bins: int):
        self.bins = int(bins)

    def reference(self, xf: np.ndarray, hf: np.ndarray) -> np.ndarray:
        return (np.asarray(xf, np.complex64) * np.asarray(hf, np.complex64)).astype(np.complex64)

    def probe_input(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        def spec() -> np.ndarray:
            return (rng.standard_normal(self.bins)
                    + 1j * rng.standard_normal(self.bins)).astype(np.complex64)
        return spec(), spec()


# --- verifiers ---------------------------------------------------------------
def _fp32_budget(n: int, atol: float) -> float:
    """fp32 round-off grows ~ N*eps, so scale the budget with the transform size.

    A fixed tolerance either passes a miscompile at small N or fails a correct
    kernel at large N; the FFT verifier above already made this call and these
    reuse it so the family answers accuracy the same way throughout.
    """
    return max(atol, 1e-4 * max(1, n))


def _verify_region(candidate: Candidate, region: Any, *, atol: float = 1e-3,
                   seed: int = 0) -> bool:
    """Verify against the region's own reference, whatever its operand arity.

    The arity comes from `probe_input`: a tuple means several operands. One
    verifier rather than a per-arity family, because the arity is a property of
    the region and splitting it invites the two from drifting -- which is how
    `stft`'s window operand went unverified in the first place.
    """
    probe = region.probe_input(seed)
    args = probe if isinstance(probe, tuple) else (probe,)
    return verify_by_reference(candidate, region, args, region.reference(*args),
                               atol=_fp32_budget(getattr(region, "n", 1), atol))


register_op_kind(OP_SPECTRAL_RFFT, _verify_region)
register_op_kind(OP_SPECTRAL_IRFFT, _verify_region)
register_op_kind(OP_SPECTRAL_STFT, _verify_region)
register_op_kind(OP_SPECTRAL_ISTFT, _verify_region)
register_op_kind(OP_SPECTRAL_FILTER, _verify_region)


# --- composed candidates -----------------------------------------------------
class _ComposedSpectralCandidate(Candidate):
    """Base for an op that delegates its inner transform to a `spectral_fft` lane.

    `available()` asks the FFT candidates rather than answering for itself: a
    composition cannot be more available than the transform it is built on, and
    claiming otherwise is how a lane ends up reporting its own name for work the
    reference actually did.
    """

    tier = Tier.SYNTHESIZED

    #: Attribute on the region giving the length of the INNER complex
    #: transform. For the framed ops that is the window, not the signal.
    inner_len_attr = "n"

    def applies_to(self, region: Any) -> bool:
        """Decline whatever the inner FFT would decline.

        The composed ops inherit the kernels' power-of-two restriction, and it
        applies to the length actually transformed: `stft`/`istft` transform a
        WINDOW, so a 1000-sample signal with a 64-sample window is fine while a
        1024-sample signal with a 60-sample window is not. Reading the signal
        length there would be the wrong test in both directions.
        """
        return _supported_length(int(getattr(region, self.inner_len_attr, 0) or 0))

    def available(self) -> bool:
        from tessera.compiler.emit.candidate import candidates_for

        for cand in candidates_for(self.target, OP_SPECTRAL_FFT):
            try:
                if cand.available():
                    return True
            except Exception:
                continue
        return False


class RFFTCandidate(_ComposedSpectralCandidate):
    """Real FFT via the complex lane, keeping the non-redundant half."""

    op = OP_SPECTRAL_RFFT

    def run(self, region: SpectralRFFTRegion, x: np.ndarray, *a: Any,
            **k: Any) -> tuple[Any, str]:
        if not self.applies_to(region):
            return region.reference(x), "reference"
        try:
            full, lane = _inner_fft(self.target, region.n, -1,
                                    np.asarray(x, np.float32).astype(np.complex64))
            if lane == "reference":
                return region.reference(x), "reference"
            return np.asarray(full[:region.bins], np.complex64), f"{lane}+rfft"
        except Exception:
            return region.reference(x), "reference"


class IRFFTCandidate(_ComposedSpectralCandidate):
    """Inverse real FFT: rebuild the Hermitian spectrum, inverse-transform, take
    the real part.

    The mirrored bins are `conj(X[k])` at `n-k`, and the DC bin -- plus Nyquist
    when `n` is even -- are their own mirror and must NOT be written twice.
    Doubling them is the classic way this reconstruction goes subtly wrong: the
    result stays real and plausible, with the endpoints off.
    """

    op = OP_SPECTRAL_IRFFT

    def run(self, region: SpectralIRFFTRegion, xf: np.ndarray, *a: Any,
            **k: Any) -> tuple[Any, str]:
        if not self.applies_to(region):
            return region.reference(xf), "reference"
        try:
            half = np.asarray(xf, np.complex64)
            n = region.n
            full = np.zeros(n, np.complex64)
            full[:region.bins] = half[:region.bins]
            for k_idx in range(1, (n + 1) // 2):
                full[n - k_idx] = np.conj(half[k_idx])
            out, lane = _inner_fft(self.target, n, +1, full)
            if lane == "reference":
                return region.reference(xf), "reference"
            return np.asarray(np.real(out), np.float32), f"{lane}+irfft"
        except Exception:
            return region.reference(xf), "reference"


class STFTCandidate(_ComposedSpectralCandidate):
    """Framed transform: window each frame, then the composed real FFT."""

    op = OP_SPECTRAL_STFT
    inner_len_attr = "win"

    def run(self, region: SpectralSTFTRegion, x: np.ndarray,
            win: np.ndarray, *a: Any, **k: Any) -> tuple[Any, str]:
        if not self.applies_to(region):
            return region.reference(x, win), "reference"
        try:
            sig = np.asarray(x, np.float32)
            w = np.asarray(win, np.float32)
            inner = SpectralRFFTRegion(region.win)
            rfft = RFFTCandidate()
            rfft.target = self.target
            frames, lane_seen = [], None
            for start in range(0, max(1, region.n - region.win + 1), region.hop):
                out, lane = rfft.run(inner, sig[start:start + region.win] * w)
                if lane == "reference":
                    return region.reference(x, win), "reference"
                lane_seen = lane
                frames.append(np.asarray(out, np.complex64))
            return np.stack(frames, axis=-2), f"{lane_seen}+stft"
        except Exception:
            return region.reference(x, win), "reference"


class ISTFTCandidate(_ComposedSpectralCandidate):
    """Overlap-add inverse: composed inverse real FFT per frame, then accumulate.

    Accumulates in f64 and stores f32, matching the reference. The accumulator
    width is a separate decision from the result width, and conflating them is
    what made `istft` hard-code `float64` as its RESULT type before W1.4.
    """

    op = OP_SPECTRAL_ISTFT
    inner_len_attr = "win"

    def run(self, region: SpectralISTFTRegion, xf: np.ndarray,
            win: np.ndarray, *a: Any, **k: Any) -> tuple[Any, str]:
        if not self.applies_to(region):
            return region.reference(xf, win), "reference"
        try:
            spec = np.asarray(xf, np.complex64)
            w = np.asarray(win, np.float32)
            inner = SpectralIRFFTRegion(region.win)
            irfft = IRFFTCandidate()
            irfft.target = self.target
            out = np.zeros(region.samples, np.float64)
            weight = np.zeros_like(out)
            lane_seen = None
            for i in range(region.frames):
                frame, lane = irfft.run(inner, spec[i])
                if lane == "reference":
                    return region.reference(xf, win), "reference"
                lane_seen = lane
                start = i * region.hop
                out[start:start + region.win] += np.asarray(frame, np.float64) * w
                weight[start:start + region.win] += w * w
            return ((out / np.maximum(weight, 1e-12)).astype(np.float32),
                    f"{lane_seen}+istft")
        except Exception:
            return region.reference(xf, win), "reference"


class SpectralFilterCandidate(Candidate):
    """Pointwise spectral product. Needs no FFT lane, so it is always available
    -- unlike its four siblings, which cannot outrun their inner transform."""

    op = OP_SPECTRAL_FILTER
    tier = Tier.SYNTHESIZED

    def available(self) -> bool:
        return True

    def run(self, region: SpectralFilterRegion, xf: np.ndarray,
            hf: np.ndarray, *a: Any, **k: Any) -> tuple[Any, str]:
        try:
            return (np.asarray(xf, np.complex64) * np.asarray(hf, np.complex64)
                    ).astype(np.complex64), "spectral_filter"
        except Exception:
            return region.reference(xf, hf), "reference"


def _compose_for_target(target: str) -> None:
    """Attach the five composed ops to one target's FFT lane."""
    for cls in (RFFTCandidate, IRFFTCandidate, STFTCandidate, ISTFTCandidate):
        cand = cls()
        cand.target = target
        cand.name = f"{target}_{cls.op.removeprefix('spectral_')}"
        register_candidate(cand)
    filt = SpectralFilterCandidate()
    filt.target = target
    filt.name = f"{target}_spectral_filter"
    register_candidate(filt)


def register_composed_lanes() -> None:
    """Attach the composed ops to every target that currently has an FFT lane.

    IDEMPOTENT and re-runnable, which is the whole point. An earlier version ran
    once at import and snapshotted the registry, so a `spectral_fft` candidate
    registered LATER -- exactly the NVIDIA/Apple extension path this design
    advertises -- silently received none of the composed ops. The claim that
    "one FFT lane per foundation lights up five TSOL ops" was true only for
    lanes that happened to be imported first.

    `register_candidate` replaces by name within a bucket, so calling this again
    after a new FFT lane registers is safe and picks up the new target.
    """
    from tessera.compiler.emit.candidate import _CANDIDATES

    for target in sorted({t for (t, op) in _CANDIDATES if op == OP_SPECTRAL_FFT}):
        _compose_for_target(target)


def _on_registered(target: str, op: str) -> None:
    """Compose as soon as a target gains an FFT lane, whenever that happens."""
    if op == OP_SPECTRAL_FFT:
        _compose_for_target(target)


register_composed_lanes()
on_candidate_registered(_on_registered)
