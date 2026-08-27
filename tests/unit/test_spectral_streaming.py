import numpy as np
import pytest
from dataclasses import replace

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
    assert len(state.state_digest) == 64
    assert len(state.parent_state_digest) == 64


def test_streaming_state_lineage_rejects_tail_counter_window_and_parent_drift():
    signal = np.arange(17, dtype=np.float32)
    window = np.hanning(6).astype(np.float32)
    policy = StreamingSTFTPolicy(
        axis=-1, n_fft=8, window_length=6, hop=4, max_chunk_samples=9
    )
    _, state = stream_stft_chunk(signal[:9], window, policy)

    altered_tail = state.tail.copy()
    altered_tail[0] += 1
    for altered, message in (
        (replace(state, tail=altered_tail), "tail was altered"),
        (replace(state, samples_consumed=10), "lineage digest mismatch"),
        (replace(state, parent_state_digest="0" * 64), "lineage digest mismatch"),
    ):
        with pytest.raises(ValueError, match=message):
            stream_stft_chunk(signal[9:], window, policy, altered)

    changed_window = window.copy()
    changed_window[2] += 0.25
    with pytest.raises(ValueError, match="window changed"):
        stream_stft_chunk(signal[9:], changed_window, policy, state)


@pytest.mark.parametrize("onesided", [True, False])
@pytest.mark.parametrize("target", ["x86", "rocm", "nvidia_sm120"])
def test_physical_streaming_broadcast_strides_and_artifact_lineage(onesided, target):
    from tessera import runtime

    if target == "x86" and not runtime._x86_elementwise_available():
        pytest.skip("x86 spectral physical package is unavailable")
    if target == "rocm" and not runtime._rocm_wmma_runtime_available():
        pytest.skip("gfx1151 spectral physical package is unavailable")
    if target == "nvidia_sm120":
        lib = runtime._load_nvidia_fft_runtime()
        if lib is None or lib.tessera_nvidia_spectral_arch() != 120:
            pytest.skip("SM120 spectral physical package is unavailable")
    rng = np.random.default_rng(812 + int(onesided))
    signal = rng.standard_normal((2, 46, 3)).astype(np.float32)[:, ::2, :]
    window = np.stack((np.hanning(6), np.hamming(6)), axis=0).astype(
        np.float32
    )[:, None, :]
    policy = StreamingSTFTPolicy(
        axis=1, n_fft=8, window_length=6, hop=4, onesided=onesided,
        max_chunk_samples=9,
    )
    pieces = np.split(signal, [7, 16], axis=1)
    state = None
    outputs = []
    for piece in pieces:
        output, state = stream_stft_chunk(
            piece, window, policy, state, target=target
        )
        outputs.append(output)
    actual = np.concatenate(outputs, axis=1)
    expected = tessera.ops.stft(
        signal, window, axis=1, n_fft=8, hop=4, center=False,
        onesided=onesided,
    )
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)
    assert state is not None and len(state.artifact_digest) == 64
    assert state.execution_certificate["origin"] == "runtime"
    assert state.execution_certificate["architecture_identity"] == (
        "zen5-avx512" if target == "x86" else
        "gfx1151" if target == "rocm" else "sm_120"
    )
    with pytest.raises(ValueError, match="different physical artifact"):
        stream_stft_chunk(
            signal[:, :1, :], window, policy, state, target="reference"
        )


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


def test_per_batch_window_broadcast_matches_independent_channel_oracle_and_vjp():
    from tessera.autodiff.vjp import get_vjp

    rng = np.random.default_rng(811)
    signal = rng.standard_normal((2, 24, 3)).astype(np.float32)
    window = np.stack(
        [np.hanning(8), np.hamming(8)], axis=0
    ).astype(np.float32)[:, None, :]
    actual = tessera.ops.stft(
        signal, window, hop=4, axis=1, n_fft=8, onesided=False
    )
    expected = np.empty_like(actual)
    for batch in range(2):
        for channel in range(3):
            expected[batch, :, :, channel] = tessera.ops.stft(
                signal[batch, :, channel], window[batch, 0], hop=4,
                n_fft=8, onesided=False,
            )
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

    dy = (rng.standard_normal(actual.shape) +
          1j * rng.standard_normal(actual.shape)).astype(np.complex64)
    dx, dw = get_vjp("stft")(
        dy, signal, window, hop=4, axis=1, n_fft=8, onesided=False
    )
    expected_dx = np.empty_like(dx)
    expected_dw = np.zeros_like(window, dtype=np.float64)
    for batch in range(2):
        for channel in range(3):
            local_dx, local_dw = get_vjp("stft")(
                dy[batch, :, :, channel], signal[batch, :, channel],
                window[batch, 0], hop=4, n_fft=8, onesided=False,
            )
            expected_dx[batch, :, channel] = local_dx
            expected_dw[batch, 0] += local_dw
    np.testing.assert_allclose(dx, expected_dx, atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(dw, expected_dw, atol=3e-5, rtol=3e-5)

    restored = tessera.ops.istft(
        actual, window, hop=4, axis=2, n_fft=8, onesided=False, length=24
    )
    assert restored.shape == signal.shape


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
