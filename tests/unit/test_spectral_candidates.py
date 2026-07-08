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
