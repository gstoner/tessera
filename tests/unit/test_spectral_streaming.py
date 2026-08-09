import numpy as np
import pytest

import tessera
from tessera.autodiff.jvp import get_jvp
from tessera.autodiff.vjp import get_vjp
from tessera.compiler.spectral_streaming import (
    StreamingSTFTPolicy,
    stream_stft_chunk,
)


@pytest.mark.parametrize("axis", [-1, 1])
def test_chunked_stft_matches_monolithic_strided_policy(axis):
    rng = np.random.default_rng(808)
    shape = (2, 23) if axis == -1 else (2, 23, 3)
    signal = rng.standard_normal(shape).astype(np.float32)
    window = np.hanning(6).astype(np.float32)
    policy = StreamingSTFTPolicy(
        axis=axis,
        n_fft=8,
        window_length=6,
        hop=4,
        max_chunk_samples=9,
    )
    pieces = np.split(signal, [7, 16], axis=axis)
    state = None
    outputs = []
    for piece in pieces:
        output, state = stream_stft_chunk(piece, window, policy, state)
        outputs.append(output)
    axis_index = axis if axis >= 0 else signal.ndim + axis
    actual = np.concatenate(outputs, axis=axis_index)
    expected = tessera.ops.stft(
        signal, window, hop=4, axis=axis, n_fft=8, center=False
    )
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    assert state is not None
    assert state.policy_digest == policy.digest
    assert state.samples_consumed == 23
    assert state.frames_emitted == expected.shape[axis_index]


def test_streaming_centered_policy_fails_closed_without_lookahead():
    with pytest.raises(ValueError, match="lookahead lineage"):
        StreamingSTFTPolicy(
            axis=-1, n_fft=8, window_length=8, hop=4, center=True
        )


def test_stft_istft_extended_policy_round_trip_and_length():
    rng = np.random.default_rng(809)
    signal = rng.standard_normal((2, 24, 3)).astype(np.float32)
    window = np.hanning(8).astype(np.float32)
    spectrum = tessera.ops.stft(
        signal,
        window,
        hop=4,
        axis=1,
        n_fft=8,
        center=True,
        pad_mode="reflect",
        onesided=True,
    )
    restored = tessera.ops.istft(
        spectrum,
        window,
        hop=4,
        axis=2,
        n_fft=8,
        center=True,
        length=24,
        onesided=True,
    )
    assert restored.shape == signal.shape
    np.testing.assert_allclose(restored, signal, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("dct_type", [1, 2, 3, 4])
def test_dct_jvp_and_vjp_match_linear_and_adjoint_identities(dct_type):
    rng = np.random.default_rng(810 + dct_type)
    x = rng.standard_normal((2, 8))
    dx = rng.standard_normal((2, 8))
    dy = rng.standard_normal((2, 8))
    primal, tangent = get_jvp("dct")(
        (x,), (dx,), axis=-1, type=dct_type
    )
    expected_primal = tessera.ops.dct(x, type=dct_type)
    expected_tangent = tessera.ops.dct(dx, type=dct_type)
    np.testing.assert_allclose(primal, expected_primal, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(tangent, expected_tangent, atol=2e-12, rtol=2e-12)
    (gradient,) = get_vjp("dct")(dy, x, axis=-1, type=dct_type)
    np.testing.assert_allclose(
        np.vdot(expected_primal, dy), np.vdot(x, gradient),
        atol=2e-11, rtol=2e-11,
    )


@pytest.mark.parametrize("normalization", ["backward", "forward", "ortho"])
def test_complex_fft_vjps_obey_adjoint_identity(normalization):
    rng = np.random.default_rng(820)
    x = rng.standard_normal(9) + 1j * rng.standard_normal(9)
    dy = rng.standard_normal(9) + 1j * rng.standard_normal(9)
    for name, forward in (("fft", np.fft.fft), ("ifft", np.fft.ifft)):
        (dx,) = get_vjp(name)(dy, x, norm=normalization)
        np.testing.assert_allclose(
            np.vdot(forward(x, norm=normalization), dy),
            np.vdot(x, dx),
            atol=2e-11,
            rtol=2e-11,
        )


@pytest.mark.parametrize("length", [8, 9])
@pytest.mark.parametrize("normalization", ["backward", "forward", "ortho"])
def test_real_fft_vjps_weight_hermitian_endpoints(length, normalization):
    rng = np.random.default_rng(830 + length)
    x = rng.standard_normal(length)
    spectrum = np.fft.rfft(x, norm=normalization)
    dy_half = rng.standard_normal(spectrum.shape) + 1j * rng.standard_normal(
        spectrum.shape
    )
    (dx,) = get_vjp("rfft")(dy_half, x, norm=normalization)
    np.testing.assert_allclose(
        np.real(np.vdot(spectrum, dy_half)),
        np.vdot(x, dx),
        atol=2e-11,
        rtol=2e-11,
    )

    dy_real = rng.standard_normal(length)
    (dhalf,) = get_vjp("irfft")(
        dy_real, spectrum, n=length, norm=normalization
    )
    np.testing.assert_allclose(
        np.vdot(np.fft.irfft(spectrum, n=length, norm=normalization), dy_real),
        np.real(np.vdot(spectrum, dhalf)),
        atol=2e-11,
        rtol=2e-11,
    )


def test_spectral_filter_vjp_conjugates_and_differentiates_filter():
    rng = np.random.default_rng(842)
    x = rng.standard_normal((2, 5)) + 1j * rng.standard_normal((2, 5))
    filt = rng.standard_normal((1, 5)) + 1j * rng.standard_normal((1, 5))
    dy = rng.standard_normal((2, 5)) + 1j * rng.standard_normal((2, 5))
    dx, df = get_vjp("spectral_filter")(dy, x, filt)
    pairing = np.real(np.vdot(x * filt, dy))
    np.testing.assert_allclose(pairing, np.real(np.vdot(x, dx)), atol=2e-11)
    np.testing.assert_allclose(pairing, np.real(np.vdot(filt, df)), atol=2e-11)


@pytest.mark.parametrize("normalization", ["backward", "forward", "ortho"])
def test_stft_vjp_transposes_framing_fft_and_window(normalization):
    rng = np.random.default_rng(850)
    x = rng.standard_normal((2, 11))
    window = rng.standard_normal(4)
    y = tessera.ops.stft(
        x, window, hop=2, n_fft=4, norm=normalization
    )
    dy = rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape)
    dx, dwindow = get_vjp("stft")(
        dy,
        x,
        window,
        hop=2,
        n_fft=4,
        norm=normalization,
    )
    pairing = np.real(np.vdot(y, dy))
    np.testing.assert_allclose(pairing, np.vdot(x, dx), atol=3e-11)
    np.testing.assert_allclose(pairing, np.vdot(window, dwindow), atol=3e-11)


def test_centered_reflect_stft_vjp_folds_padding_on_arbitrary_axis():
    rng = np.random.default_rng(853)
    x = rng.standard_normal((2, 7, 3))
    window = rng.standard_normal(4)
    y = tessera.ops.stft(
        x,
        window,
        hop=2,
        axis=1,
        n_fft=4,
        center=True,
        pad_mode="reflect",
        norm="ortho",
    )
    dy = rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape)
    dx, dwindow = get_vjp("stft")(
        dy,
        x,
        window,
        hop=2,
        axis=1,
        n_fft=4,
        center=True,
        pad_mode="reflect",
        norm="ortho",
    )
    pairing = np.real(np.vdot(y, dy))
    np.testing.assert_allclose(pairing, np.vdot(x, dx), atol=4e-11)
    np.testing.assert_allclose(pairing, np.vdot(window, dwindow), atol=4e-11)


def test_istft_vjp_transposes_overlap_add_and_differentiates_window():
    rng = np.random.default_rng(851)
    signal = rng.standard_normal((2, 12))
    window = np.hanning(4) + 0.2
    spectrum = tessera.ops.stft(signal, window, hop=2, n_fft=4)
    output = tessera.ops.istft(spectrum, window, hop=2, n_fft=4)
    dout = rng.standard_normal(output.shape)
    d_spectrum, d_window = get_vjp("istft")(
        dout, spectrum, window, hop=2, n_fft=4
    )
    np.testing.assert_allclose(
        np.vdot(output, dout),
        np.real(np.vdot(spectrum, d_spectrum)),
        atol=3e-11,
    )
    epsilon = 1e-6
    finite_difference = np.zeros_like(window)
    for index in range(window.size):
        positive = window.copy()
        negative = window.copy()
        positive[index] += epsilon
        negative[index] -= epsilon
        finite_difference[index] = (
            np.vdot(
                tessera.ops.istft(spectrum, positive, hop=2, n_fft=4),
                dout,
            )
            - np.vdot(
                tessera.ops.istft(spectrum, negative, hop=2, n_fft=4),
                dout,
            )
        ) / (2.0 * epsilon)
    np.testing.assert_allclose(d_window, finite_difference, atol=2e-8, rtol=2e-8)


def test_spectral_conv_vjp_matches_full_convolution_pairings():
    rng = np.random.default_rng(852)
    x = rng.standard_normal(7)
    kernel = rng.standard_normal(4)
    output = tessera.ops.spectral_conv(x, kernel)
    dout = rng.standard_normal(output.shape)
    dx, dkernel = get_vjp("spectral_conv")(dout, x, kernel)
    pairing = np.vdot(output, dout)
    np.testing.assert_allclose(pairing, np.vdot(x, dx), atol=2e-11)
    np.testing.assert_allclose(pairing, np.vdot(kernel, dkernel), atol=2e-11)


@pytest.mark.parametrize("normalization", ["backward", "forward", "ortho"])
def test_spectral_conv_vjp_handles_axis_and_broadcasting(normalization):
    rng = np.random.default_rng(854)
    x = rng.standard_normal((6, 2, 3))
    kernel = rng.standard_normal((3, 1, 3))
    output = tessera.ops.spectral_conv(
        x, kernel, axis=0, norm=normalization
    )
    dout = rng.standard_normal(output.shape)
    dx, dkernel = get_vjp("spectral_conv")(
        dout, x, kernel, axis=0, norm=normalization
    )
    pairing = np.vdot(output, dout)
    np.testing.assert_allclose(pairing, np.vdot(x, dx), atol=4e-11)
    np.testing.assert_allclose(pairing, np.vdot(kernel, dkernel), atol=4e-11)
    assert dx.shape == x.shape
    assert dkernel.shape == kernel.shape
