"""AD-TSOL-STFT-BWD-1 (integrated-plan queue order 7) — the STFT/ISTFT adjoint
contract and the mathematical oracle for the bounded native kernels.

The row asks for native STFT/ISTFT backward packages on AVX-512 and gfx1151.
Initial measured state on 2026-08-25:

  x86     forward `tessera_x86_stft_f32` / `tessera_x86_istft_f32` execute
          natively, and there is even a native ISTFT **JVP**
          (`tessera_x86_istft_jvp_f32`) — but no backward symbol at that point.
  gfx1151 no STFT/ISTFT at all. `tile.spectral_backward_kernel` accepts
          `tessera.spectral_conv` and `tessera.spectral_filter` and refuses
          `tessera.stft` with "compound spectral adjoint kind has no gfx1151
          native package" (`gfx1151_spectral_backward_fail_closed.mlir`).
          Correct today, and a bigger lift than the x86 half.

The bounded AVX-512 backward symbols landed on 2026-08-26 for contiguous,
uncentered, onesided f32/complex64 with explicit hop and ``n_fft == window``;
the broader x86 envelopes and independent gfx1151 package remain open. Both
operators are LINEAR, so the VJP is the adjoint — and that makes the contract
checkable exactly rather than approximately, which is what this file does.

── The adjoint ──

For ``X[t,f] = sum_n x[n + t*H] * w[n] * exp(-2i*pi*f*n/N)`` over onesided
bins ``f = 0..N/2``:

    STFT^H(Xbar)[n + t*H] += w[n] * Re( sum_f Xbar[t,f] * exp(+2i*pi*f*n/N) )

Each STORED bin is counted once. That is the whole subtlety, and it is why the
two obvious shortcuts below are wrong.

── Two refuted shortcuts, both measured ──

Reusing the existing forward ISTFT kernel would make the native package nearly
free, so it is worth knowing precisely that it cannot:

* ``ISTFT`` is not ``STFT^H`` up to any global scale — best-fit residual 0.968
  for every normalization (backward / forward / ortho).
* Nor is it ``STFT^H`` divided by the COLA window-sum: multiplying back by
  ``sum_t w^2[n - tH]`` still leaves 0.887.

The division is pointwise and the windowing differs; the adjoint is a
different program.

── What it IS composable from ──

``irfft`` reconstructs the Hermitian pair, so it counts each interior bin
TWICE. Halving the interior bins — leaving DC and Nyquist alone, since they
are self-conjugate — makes ``N * irfft`` the per-frame adjoint transform:

    STFT^H = overlap_add( w * (N * irfft(Xbar * c)) ),  c = [1, ½ … ½, 1]

verified to 1.4e-15, where the same expression WITHOUT the halving is wrong by
127%. So the native backward needs no new FFT kernel — bin scaling, an inverse
real transform, a window multiply, and an overlap-add scatter.
"""

from __future__ import annotations

import numpy as np
import pytest

N, HOP, T = 16, 8, 6
L = HOP * (T - 1) + N


@pytest.fixture(scope="module")
def window():
    return np.hanning(N).astype(np.float64)


def _adjoint_direct(Xbar, w):
    """STFT^H from the definition: each stored bin counted once."""
    out = np.zeros(L)
    n = np.arange(N)
    for t in range(T):
        acc = np.zeros(N)
        for f in range(N // 2 + 1):
            acc += np.real(Xbar[t, f] * np.exp(2j * np.pi * f * n / N))
        out[t * HOP:t * HOP + N] += w * acc
    return out


def _spectrogram(seed=3):
    rs = np.random.RandomState(seed)
    return rs.randn(T, N // 2 + 1) + 1j * rs.randn(T, N // 2 + 1)


# ── the reference VJP is the oracle; check the oracle first ────────────────

def test_reference_stft_vjp_satisfies_the_adjoint_identity(window):
    """STFT is linear, so <STFT(x), Xbar> == <x, STFT^H(Xbar)> must hold
    EXACTLY — no tolerance argument, and no finite differences needed. It also
    catches the onesided trap, which a scalar loss check can miss."""
    from tessera import ops
    from tessera.autodiff import vjp

    rs = np.random.RandomState(0)
    x = rs.randn(L)
    X = np.asarray(ops.stft(x, window, n_fft=N, hop=HOP, onesided=True))
    Xbar = rs.randn(*X.shape) + 1j * rs.randn(*X.shape)
    xbar = np.asarray(
        vjp._VJPS["stft"](Xbar, x, window, n_fft=N, hop=HOP, onesided=True)[0])

    lhs = float(np.real(np.sum(np.conj(Xbar) * X)))
    rhs = float(np.dot(x, xbar))
    assert abs(lhs - rhs) <= 1e-12 * max(abs(lhs), 1.0), (lhs, rhs)


def test_reference_istft_vjp_satisfies_the_adjoint_identity(window):
    from tessera import ops
    from tessera.autodiff import vjp

    rs = np.random.RandomState(1)
    X = _spectrogram(1)
    y = np.asarray(ops.istft(X, window, n_fft=N, hop=HOP,
                             onesided=True, length=L))
    ybar = rs.randn(*y.shape)
    Xbar = np.asarray(vjp._VJPS["istft"](ybar, X, window, n_fft=N, hop=HOP,
                                         onesided=True, length=L)[0])
    lhs = float(np.dot(ybar, y))
    rhs = float(np.real(np.sum(np.conj(Xbar) * X)))
    assert abs(lhs - rhs) <= 1e-12 * max(abs(lhs), 1.0), (lhs, rhs)


def test_overlap_add_identity_holds_on_the_cola_interior(window):
    """ISTFT(STFT(x)) == x where the Hann window at 50% overlap is COLA.
    Edges are excluded because the first and last frames are not fully
    overlapped — stating that is the point, since a test that silently
    included them would need a loose tolerance and would stop detecting
    anything."""
    from tessera import ops

    rs = np.random.RandomState(2)
    x = rs.randn(L)
    X = np.asarray(ops.stft(x, window, n_fft=N, hop=HOP, onesided=True))
    back = np.asarray(ops.istft(X, window, n_fft=N, hop=HOP,
                                onesided=True, length=L))
    interior = slice(N, L - N)
    assert np.abs(back[interior] - x[interior]).max() < 1e-12


# ── the contract the native kernels must implement ─────────────────────────

def test_the_definition_reproduces_the_reference_adjoint(window):
    from tessera.autodiff import vjp

    rs = np.random.RandomState(3)
    Xbar = _spectrogram(3)
    ref = np.asarray(vjp._VJPS["stft"](Xbar, rs.randn(L), window,
                                       n_fft=N, hop=HOP, onesided=True)[0])
    mine = _adjoint_direct(Xbar, window)
    assert np.abs(mine - ref).max() <= 1e-12 * max(np.abs(ref).max(), 1.0)


def test_the_adjoint_is_composable_from_irfft_with_interior_bin_halving(window):
    """The implementation strategy: no new FFT kernel is needed."""
    Xbar = _spectrogram(5)
    correction = np.ones(N // 2 + 1)
    correction[1:N // 2] = 0.5           # DC and Nyquist are self-conjugate

    composed = np.zeros(L)
    for t in range(T):
        frame = np.fft.irfft(Xbar[t] * correction, n=N) * N
        composed[t * HOP:t * HOP + N] += window * frame

    direct = _adjoint_direct(Xbar, window)
    assert np.abs(composed - direct).max() <= 1e-12 * max(np.abs(direct).max(), 1.0)


def test_omitting_the_halving_is_wrong_by_more_than_a_tolerance(window):
    """The control. `irfft` reconstructs the Hermitian pair and counts each
    interior bin twice; the adjoint counts each stored bin once. Without this
    row the composition test above would pass for a kernel that got the bin
    weights wrong on a DC-heavy signal."""
    Xbar = _spectrogram(5)
    naive = np.zeros(L)
    for t in range(T):
        naive[t * HOP:t * HOP + N] += window * (np.fft.irfft(Xbar[t], n=N) * N)
    direct = _adjoint_direct(Xbar, window)
    relative = np.abs(naive - direct).max() / max(np.abs(direct).max(), 1e-30)
    assert relative > 1.0, relative


# ── two shortcuts that would have made the native package nearly free ──────

def test_istft_is_not_the_stft_adjoint_up_to_a_scale(window):
    """Refuted so the next implementer does not spend a day on it."""
    from tessera import ops
    from tessera.autodiff import vjp

    rs = np.random.RandomState(3)
    Xbar = _spectrogram(3)
    ref = np.asarray(vjp._VJPS["stft"](Xbar, rs.randn(L), window,
                                       n_fft=N, hop=HOP, onesided=True)[0])
    for norm in ("backward", "forward", "ortho"):
        cand = np.asarray(ops.istft(Xbar, window, n_fft=N, hop=HOP,
                                    onesided=True, length=L, norm=norm))
        denom = float(np.dot(cand, cand))
        scale = float(np.dot(cand, ref) / denom) if denom > 1e-30 else 0.0
        residual = (np.abs(scale * cand - ref).max()
                    / max(np.abs(ref).max(), 1e-30))
        assert residual > 0.1, (norm, residual)


def test_istft_times_the_window_sum_is_not_the_adjoint_either(window):
    """The second guess: undo the COLA division. Also wrong — the division is
    pointwise and the windowing differs, so the adjoint is a different
    program, not a rescaled one."""
    from tessera import ops
    from tessera.autodiff import vjp

    rs = np.random.RandomState(3)
    Xbar = _spectrogram(3)
    ref = np.asarray(vjp._VJPS["stft"](Xbar, rs.randn(L), window,
                                       n_fft=N, hop=HOP, onesided=True)[0])
    denominator = np.zeros(L)
    for t in range(T):
        denominator[t * HOP:t * HOP + N] += window * window
    cand = np.asarray(ops.istft(Xbar, window, n_fft=N, hop=HOP,
                                onesided=True, length=L)) * denominator
    live = denominator > 1e-12
    residual = (np.abs(cand[live] - ref[live]).max()
                / max(np.abs(ref).max(), 1e-30))
    assert residual > 0.1, residual
