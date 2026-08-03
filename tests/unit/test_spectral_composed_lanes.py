"""TSOL spectral family — `rfft`/`irfft`/`stft`/`istft` composed over the FFT lane.

Tessera ships its own FFT (Stockham radix-4) at
`src/solvers/spectral/lib/TargetHooks/{CPU,AMD,NVIDIA}/StockhamRadix4.*`, and
the CPU and AMD lanes were already registered as arbiter candidates for
`spectral_fft`. The other five TSOL spectral ops had no region and no candidate
on any target, so nothing arbitrated for them anywhere.

They are not new kernels. Each decomposes to the complex FFT the shipped lanes
already implement, so a candidate here delegates its inner transform and one FFT
lane per foundation lights up five TSOL ops on that foundation. When the NVIDIA
`.cu` — written, but not registered as a candidate — and an Apple lane land,
these compose on top with no further work: registration is driven off the FFT
registry rather than a hard-coded target list.

`spectral_filter` is deliberately NOT composed over the FFT. It is a pointwise
product of two spectra with no transform inside; filing it with the transforms
because its operands happen to be spectra would repeat the
one-reason-for-N-ops error that hid `kv_cache.read` among the cache mutators and
`irfft` among the complex-returning transforms.

Status note: this does NOT flip the TSOL `backend_kernel` axis to `complete`.
Two of four foundations are proven (CPU, and AMD on live gfx1151); NVIDIA and
Apple are not. `partial` is the accurate reading, and claiming otherwise would
be the same over-claim already backed out twice in this area.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.emit import spectral_candidates as SC
from tessera.compiler.emit.candidate import candidates_for

_COMPOSED_OPS = (
    SC.OP_SPECTRAL_RFFT,
    SC.OP_SPECTRAL_IRFFT,
    SC.OP_SPECTRAL_STFT,
    SC.OP_SPECTRAL_ISTFT,
)


def _regions():
    return {
        SC.OP_SPECTRAL_RFFT: (SC.SpectralRFFTRegion(256), 1),
        SC.OP_SPECTRAL_IRFFT: (SC.SpectralIRFFTRegion(256), 1),
        SC.OP_SPECTRAL_STFT: (SC.SpectralSTFTRegion(256, 64, 32), 1),
        SC.OP_SPECTRAL_ISTFT: (SC.SpectralISTFTRegion(7, 64, 32), 1),
        SC.OP_SPECTRAL_FILTER: (SC.SpectralFilterRegion(129), 2),
    }


@pytest.mark.parametrize("op", [*_COMPOSED_OPS, SC.OP_SPECTRAL_FILTER])
def test_every_fft_target_also_has_the_composed_op(op):
    """Registration is driven off the FFT registry, not a target list.

    That coupling is the point: a foundation cannot end up with a `stft`
    candidate but no FFT to build it on, and adding an FFT lane cannot leave the
    composed ops behind.
    """
    from tessera.compiler.emit.candidate import _CANDIDATES

    fft_targets = {t for (t, kind) in _CANDIDATES if kind == SC.OP_SPECTRAL_FFT}
    assert fft_targets, "no spectral_fft candidates registered at all"
    for target in fft_targets:
        assert candidates_for(target, op), (
            f"{target} has an FFT lane but no {op} candidate"
        )


@pytest.mark.parametrize("op", [*_COMPOSED_OPS, SC.OP_SPECTRAL_FILTER])
def test_composed_lanes_match_the_reference(op):
    """The F4 gate: every registered lane agrees with the numpy reference.

    Relative error, not absolute: fp32 FFT round-off scales with the transform
    size and with the signal magnitude, so a fixed tolerance either passes a
    miscompile at small N or fails a correct kernel at large N.
    """
    region, arity = _regions()[op]
    probe = region.probe_input(11)
    args = probe if arity == 2 else (probe,)
    reference = np.asarray(region.reference(*args))

    checked = 0
    for target in ("cpu", "rocm"):
        for candidate in candidates_for(target, op):
            if not candidate.available():
                continue  # no toolchain / no device — declines by design
            out, lane = candidate.run(region, *args)
            out = np.asarray(out)
            assert out.shape == reference.shape, (candidate.name, out.shape)
            assert out.dtype == reference.dtype, (candidate.name, out.dtype)
            scale = float(np.max(np.abs(reference))) or 1.0
            rel = float(np.max(np.abs(out - reference))) / scale
            assert rel < 1e-4, f"{candidate.name} ({lane}) rel_err={rel:.3e}"
            checked += 1
    assert checked, f"no available {op} lane could be exercised"


def test_lane_provenance_names_the_inner_transform():
    """A composed lane reports the FFT lane it actually used.

    `rocm_stockham+irfft+istft` says the inner transform ran on the ROCm
    Stockham kernel. A composition that reported only its own name would be
    indistinguishable from one that silently fell back to the reference — which
    is the specific way a kernel claim goes wrong (Decision #21).
    """
    region = SC.SpectralISTFTRegion(7, 64, 32)
    probe = region.probe_input(3)
    for target in ("cpu", "rocm"):
        for candidate in candidates_for(target, SC.OP_SPECTRAL_ISTFT):
            if not candidate.available():
                continue
            _out, lane = candidate.run(region, probe)
            assert lane != "reference", f"{candidate.name} silently fell back"
            assert "stockham" in lane, f"{candidate.name} lane={lane!r}"
            assert lane.endswith("istft"), lane


def test_a_composed_op_is_never_more_available_than_its_fft():
    """`available()` asks the FFT candidates rather than answering for itself.

    A composition cannot outrun the transform it is built on. Claiming
    otherwise is how a lane ends up reporting its own name for work the
    reference actually did.
    """
    from tessera.compiler.emit.candidate import _CANDIDATES

    for target in {t for (t, kind) in _CANDIDATES if kind == SC.OP_SPECTRAL_FFT}:
        fft_ok = any(c.available() for c in candidates_for(target, SC.OP_SPECTRAL_FFT))
        for op in _COMPOSED_OPS:
            for candidate in candidates_for(target, op):
                if candidate.available():
                    assert fft_ok, (
                        f"{candidate.name} claims availability with no usable "
                        f"{target} FFT lane"
                    )


def test_spectral_filter_needs_no_fft_lane():
    """It is a pointwise spectral product — always available, unlike its siblings."""
    for target in ("cpu", "rocm"):
        for candidate in candidates_for(target, SC.OP_SPECTRAL_FILTER):
            assert candidate.available(), candidate.name


def test_irfft_does_not_double_the_self_conjugate_bins():
    """DC — and Nyquist when `n` is even — are their own mirror.

    Writing them twice is the classic way this reconstruction goes subtly
    wrong: the output stays real and plausible with only the endpoints off, so
    a shape-and-dtype check passes it. Exercised at even AND odd `n`, since
    only the even case has a Nyquist bin to get wrong.
    """
    for n in (256, 255):
        region = SC.SpectralIRFFTRegion(n)
        spectrum = region.probe_input(5)
        reference = region.reference(spectrum)
        for target in ("cpu", "rocm"):
            for candidate in candidates_for(target, SC.OP_SPECTRAL_IRFFT):
                if not candidate.available():
                    continue
                out, lane = candidate.run(region, spectrum)
                if lane == "reference":
                    continue
                rel = (float(np.max(np.abs(np.asarray(out) - reference)))
                       / (float(np.max(np.abs(reference))) or 1.0))
                assert rel < 1e-4, f"n={n} {candidate.name} rel_err={rel:.3e}"


def test_stft_istft_round_trips_through_the_composed_lanes():
    """End-to-end: analysis then synthesis returns the signal.

    Neither op's own gate would catch a consistent framing error — an off-by-one
    in the hop or window would be self-consistent in each direction and only
    show up in the round trip.
    """
    forward = SC.SpectralSTFTRegion(256, 64, 32)
    signal = forward.probe_input(9)
    inverse = SC.SpectralISTFTRegion(forward.frames, forward.win, forward.hop)

    for target in ("cpu", "rocm"):
        stfts = candidates_for(target, SC.OP_SPECTRAL_STFT)
        istfts = candidates_for(target, SC.OP_SPECTRAL_ISTFT)
        if not (stfts and istfts) or not stfts[0].available():
            continue
        spectrum, lane_a = stfts[0].run(forward, signal)
        restored, lane_b = istfts[0].run(inverse, np.asarray(spectrum))
        if "reference" in (lane_a, lane_b):
            continue
        # Compare on the interior: overlap-add normalisation is only valid where
        # the window sum is fully covered, so the first and last hop are
        # legitimately attenuated rather than wrong.
        interior = slice(forward.win, forward.n - forward.win)
        rel = (float(np.max(np.abs(np.asarray(restored)[interior] - signal[interior])))
               / (float(np.max(np.abs(signal[interior]))) or 1.0))
        assert rel < 1e-3, f"{target} round-trip rel_err={rel:.3e}"


# ── The shipped kernels are power-of-two only, and did not say so ──────────
#
# Measured against numpy on both the CPU and ROCm lanes: every power of two from
# 1 to 1024 agrees to ~1e-7, while 3, 12, 24, 48, 100, 255 and 257 came back
# with relative error ~1.0 — a different answer, not a precision shortfall.
#
# They did not decline. Both returned their own lane name, so the arbiter
# recorded a successful `cpu_stockham` / `rocm_stockham` run for a wrong
# transform. Decision #21 requires an unsupported case to say so; a
# plausible-looking wrong array is worse than the silent no-op it prohibits.
# The F4 verifier never caught it because it only ever ran at power-of-two
# sizes — a gate is only as good as the configuration it measures, which is now
# the fourth time that has bitten in this registry work.

_POW2_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
_NON_POW2_SIZES = (3, 12, 24, 48, 100, 255, 257)


@pytest.mark.parametrize("n", _POW2_SIZES)
def test_power_of_two_transforms_use_the_kernel_and_are_correct(n):
    region = SC.SpectralFFTRegion(n=n, sign=-1)
    x = region.probe_input(2)
    reference = region.reference(x)
    for target in ("cpu", "rocm"):
        for candidate in candidates_for(target, SC.OP_SPECTRAL_FFT):
            if not candidate.available():
                continue
            assert candidate.applies_to(region), f"{candidate.name} declines n={n}"
            out, lane = candidate.run(region, x)
            assert lane != "reference", f"{candidate.name} fell back at n={n}"
            rel = (float(np.max(np.abs(np.asarray(out) - reference)))
                   / (float(np.max(np.abs(reference))) or 1.0))
            assert rel < 1e-4, f"{candidate.name} n={n} rel_err={rel:.3e}"


@pytest.mark.parametrize("n", _NON_POW2_SIZES)
def test_non_power_of_two_declines_rather_than_answering_wrongly(n):
    """The headline: a wrong answer under a kernel's own name is the worst
    outcome available, worse than declining and worse than erroring."""
    region = SC.SpectralFFTRegion(n=n, sign=-1)
    x = region.probe_input(2)
    reference = region.reference(x)
    for target in ("cpu", "rocm"):
        for candidate in candidates_for(target, SC.OP_SPECTRAL_FFT):
            assert not candidate.applies_to(region), (
                f"{candidate.name} claims n={n}, which it computes incorrectly"
            )
            out, lane = candidate.run(region, x)
            assert lane == "reference", (
                f"{candidate.name} returned lane={lane!r} for unsupported n={n}"
            )
            # Whatever it returns must still be RIGHT — declining is only
            # honest if the fallback actually computes the transform.
            rel = (float(np.max(np.abs(np.asarray(out) - reference)))
                   / (float(np.max(np.abs(reference))) or 1.0))
            assert rel < 1e-4, f"reference fallback wrong at n={n}"


def test_framed_ops_restrict_on_the_WINDOW_not_the_signal():
    """`stft`/`istft` transform a window, so that is the length that must be a
    power of two. A 1000-sample signal with a 64-sample window is fine; a
    1024-sample signal with a 60-sample window is not. Reading the signal length
    would be wrong in both directions."""
    ok = SC.SpectralSTFTRegion(1000, 64, 32)
    bad = SC.SpectralSTFTRegion(1024, 60, 30)
    for target in ("cpu", "rocm"):
        for candidate in candidates_for(target, SC.OP_SPECTRAL_STFT):
            assert candidate.applies_to(ok), "declined a power-of-two window"
            assert not candidate.applies_to(bad), "accepted a 60-sample window"
