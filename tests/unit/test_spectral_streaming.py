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
