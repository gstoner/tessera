"""Spectral FFT arbiter retarget — the ts-spectral-opt `lower-*-to-target-ir`
seam pointed at the D1 candidate arbiter.

Verifies the shipped Stockham kernel is registered as an F4-gated candidate for
the ``spectral_fft`` op-kind: the real compiled CPU kernel matches ``numpy.fft``
through the arbiter, a wrong candidate is refused even at a higher tier, and the
arbiter falls back honestly to the reference when nothing applies.  The CPU lane
compiles the shipped ``TargetHooks/CPU/StockhamRadix4.cpp``; if no C++ toolchain
is present it declines and the real-kernel assertions are skipped.
"""
from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.emit import candidate as C
from tessera.compiler.emit.candidate import Candidate, Tier
from tessera.compiler.emit import spectral_candidates as SC
from tessera.compiler.emit.spectral_candidates import (
    OP_SPECTRAL_FFT,
    SpectralFFTRegion,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    saved = {k: list(v) for k, v in C._CANDIDATES.items()}
    yield
    C._CANDIDATES.clear()
    C._CANDIDATES.update(saved)


def _cpu():
    return next(c for c in C.candidates_for("cpu", OP_SPECTRAL_FFT)
               if c.name == "cpu_stockham")


def test_op_kind_registered():
    assert OP_SPECTRAL_FFT in C._OP_KIND_VERIFY


def test_cpu_stockham_matches_numpy_fft():
    cpu = _cpu()
    if not cpu.available():
        pytest.skip("no C++ toolchain to build the shipped CPU kernel")
    for n in (64, 128, 256, 512, 1024):
        for sign in (-1, 1):
            reg = SpectralFFTRegion(n, sign=sign)
            x = reg.probe_input(0)
            out, tag = cpu.run(reg, x)
            assert tag == "cpu_stockham"
            assert np.allclose(out, reg.reference(x), atol=1e-2 * max(1, n / 64))


def test_arbiter_picks_verified_cpu_kernel():
    cpu = _cpu()
    if not cpu.available():
        pytest.skip("no C++ toolchain")
    reg = SpectralFFTRegion(256, sign=-1)
    assert C.verify_candidate(cpu, reg) is True
    win = C.arbitrate(reg, OP_SPECTRAL_FFT, "cpu")
    assert win is not None and win.name == "cpu_stockham"


def test_wrong_candidate_is_f4_rejected_even_at_higher_tier():
    if not _cpu().available():
        pytest.skip("no C++ toolchain")

    class _Wrong(Candidate):
        name, tier, target, op = "wrong", Tier.HAND_TUNED, "cpu", OP_SPECTRAL_FFT

        def run(self, region, x, *a, **k):
            return np.full(region.n, 9.0, np.complex64), "wrong_tag"

    C.register_candidate(_Wrong())
    reg = SpectralFFTRegion(128, sign=-1)
    assert C.verify_candidate(_Wrong(), reg) is False
    # Higher tier but wrong → arbiter still selects the correct CPU kernel.
    win = C.arbitrate(reg, OP_SPECTRAL_FFT, "cpu")
    assert win is not None and win.name == "cpu_stockham"


def test_run_arbitrated_end_to_end():
    if not _cpu().available():
        pytest.skip("no C++ toolchain")
    reg = SpectralFFTRegion(512, sign=-1)
    x = reg.probe_input(3)
    out, tag = C.run_arbitrated(reg, OP_SPECTRAL_FFT, "cpu", x)
    assert tag == "cpu_stockham"
    assert np.allclose(out, reg.reference(x), atol=1e-1)


def test_reference_fallback_when_no_candidate():
    reg = SpectralFFTRegion(64, sign=-1)
    x = reg.probe_input(0)
    # No candidates registered for this target → honest reference fallback.
    out, tag = C.run_arbitrated(reg, OP_SPECTRAL_FFT, "no_such_target", x)
    assert tag == "reference"
    assert np.allclose(out, reg.reference(x))


def test_canonical_rocm_loader_never_falls_back_to_source_compile(monkeypatch):
    saved = SC._libs.pop("amd_prebuilt", None)
    monkeypatch.setattr(SC, "_prebuilt_amd_paths", lambda: ())
    monkeypatch.setattr(
        SC,
        "_amd_source_lib",
        lambda: (_ for _ in ()).throw(
            AssertionError("canonical ROCm FFT attempted a source build")
        ),
    )
    try:
        assert SC._amd_lib() is None
    finally:
        if saved is not None:
            SC._libs["amd_prebuilt"] = saved


class _FakeAmdPackage:
    ts_fft_plan_create_for_artifact_amd = object()
    ts_fft_plan_artifact_digest_amd = object()
    ts_fft_plan_execute_hostptr_batch_amd = object()
    ts_fft_plan_workspace_elems_amd = object()
    ts_fft_plan_destroy_amd = object()

    def __init__(self, arch: bytes):
        self._arch = arch

    @staticmethod
    def ts_fft_package_abi_amd():
        return b"tessera.rocm.fft.plan.v1"

    @staticmethod
    def ts_spectral_composite_package_abi_amd():
        return b"tessera.rocm.spectral_composite.v7"

    def ts_spectral_composite_arch_amd(self):
        return self._arch


@pytest.mark.parametrize("arch", [b"gfx1200", b"gfx1250", b"unknown"])
def test_rocm_fft_fallback_is_not_a_composite_candidate(monkeypatch, tmp_path, arch):
    package = tmp_path / "libtessera_spectral_rocm.so"
    package.touch()
    fake = _FakeAmdPackage(arch)
    saved = dict(SC._libs)
    SC._libs.pop("amd_prebuilt", None)
    SC._libs.pop("amd_composite_prebuilt", None)
    monkeypatch.setattr(SC, "_prebuilt_amd_paths", lambda: (package,))
    monkeypatch.setattr(SC.ctypes, "CDLL", lambda _path: fake)
    monkeypatch.setattr(SC, "_configure_amd_lib", lambda lib: lib)
    try:
        assert SC._amd_lib() is fake  # Architecture-neutral FFT ABI remains usable.
        assert SC._amd_composite_lib() is None
    finally:
        SC._libs.clear()
        SC._libs.update(saved)


def test_rocm_composite_loader_accepts_exact_gfx1151_package(monkeypatch, tmp_path):
    package = tmp_path / "libtessera_spectral_rocm.so"
    package.touch()
    fake = _FakeAmdPackage(b"gfx1151")
    saved = dict(SC._libs)
    SC._libs.pop("amd_prebuilt", None)
    SC._libs.pop("amd_composite_prebuilt", None)
    monkeypatch.setattr(SC, "_prebuilt_amd_paths", lambda: (package,))
    monkeypatch.setattr(SC.ctypes, "CDLL", lambda _path: fake)
    monkeypatch.setattr(SC, "_configure_amd_lib", lambda lib: lib)
    try:
        assert SC._amd_composite_lib() is fake
        assert SC._libs["amd_composite_prebuilt"] is fake
    finally:
        SC._libs.clear()
        SC._libs.update(saved)


def test_rocm_composite_launch_rechecks_architecture(monkeypatch):
    from tessera.compiler import scheduled_spectral

    fake = _FakeAmdPackage(b"gfx1200")
    monkeypatch.setattr(SC, "_amd_composite_lib", lambda: fake)
    monkeypatch.setattr(
        scheduled_spectral,
        "validate_scheduled_spectral_metadata",
        lambda _metadata, input_shapes: object(),
    )
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        SC.run_rocm_spectral_composite({}, [])


# ── compile-cache hit must cost nothing (no scratch dir, no device probe) ────


def test_cpu_lib_creates_one_scratch_directory_for_the_whole_process(tmp_path,
                                                                    monkeypatch):
    """`_compile` serves every call after the first from `_libs`, so a scratch
    directory made BEFORE that check is abandoned empty on every later call --
    unbounded in TMPDIR, and (measured on this host) 64 us of mkdtemp per call
    on a path the composed STFT lane runs once per frame."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(SC, "_libs", {})
    monkeypatch.setattr(SC.tempfile, "tempdir", None, raising=False)
    for _ in range(5):
        SC._cpu_lib()
    made = list(tmp_path.glob("tessera_spectral_cpu_*"))
    assert len(made) <= 1


def test_cpu_lib_returns_the_cached_handle_without_recompiling(monkeypatch):
    SC._cpu_lib()
    if SC._libs.get("cpu") is None:
        pytest.skip("no C++ toolchain: the CPU lane declines, nothing to cache")
    monkeypatch.setattr(SC, "_compile", lambda *_a, **_k: pytest.fail(
        "cache hit must not reach the compiler"))
    monkeypatch.setattr(SC.tempfile, "mkdtemp", lambda *_a, **_k: pytest.fail(
        "cache hit must not create a scratch directory"))
    assert SC._cpu_lib() is SC._libs["cpu"]


def test_rocm_availability_probe_runs_once_per_process(monkeypatch):
    """`available()` is called per arbitration -- per composed STFT frame -- and
    its 4-point device transform measured 0.371 ms per call on gfx1151. The
    answer cannot change within a process, so it is probed once."""
    calls = []

    class _Lib:
        def ts_fft_stockham_amd_hostptr(self, *_a):
            calls.append(1)
            return 0

    lib = _Lib()
    monkeypatch.setattr(SC, "_amd_candidate_lib", lambda: lib)
    monkeypatch.setattr(SC, "_amd_probe", {})
    candidate = SC.RocmStockhamFFTCandidate()
    assert [candidate.available() for _ in range(5)] == [True] * 5
    assert len(calls) == 1


def test_rocm_availability_probe_memoizes_a_failed_probe_too(monkeypatch):
    calls = []

    class _Lib:
        def ts_fft_stockham_amd_hostptr(self, *_a):
            calls.append(1)
            return 1                       # device present but transform failed

    lib = _Lib()
    monkeypatch.setattr(SC, "_amd_candidate_lib", lambda: lib)
    monkeypatch.setattr(SC, "_amd_probe", {})
    candidate = SC.RocmStockhamFFTCandidate()
    assert [candidate.available() for _ in range(3)] == [False] * 3
    assert len(calls) == 1
