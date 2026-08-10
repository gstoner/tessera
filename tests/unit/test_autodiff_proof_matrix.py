"""Mathematical proof obligations for compiler-owned forward rules."""

from __future__ import annotations

import numpy as np

from tessera.compiler.autodiff_ledger import _ir_tangent_classes
from tessera.compiler.autodiff_proof import (
    DERIVATIVE_PROOF_MATRIX,
    ProofKind,
    assert_directional_close,
    assert_forward_reverse_duality,
    central_directional_difference,
)
from tessera.compiler.es_rng_contract import member_normal
from tessera.rng import RNGKey
from tessera import rng_device


def test_every_native_tangent_has_an_authored_proof_contract():
    assert _ir_tangent_classes() <= set(DERIVATIVE_PROOF_MATRIX)


def test_softmax_jvp_directional_and_dual_identity():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(3, 7))
    dx = rng.normal(size=x.shape)
    w = rng.normal(size=x.shape)

    def softmax(value):
        shifted = value - np.max(value, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=-1, keepdims=True)

    y = softmax(x)
    tangent = y * (dx - np.sum(y * dx, axis=-1, keepdims=True))
    finite = central_directional_difference(softmax, (x,), (dx,))
    assert_directional_close(tangent, finite)
    cotangent = y * (w - np.sum(y * w, axis=-1, keepdims=True))
    assert_forward_reverse_duality(tangent, w, (dx,), (cotangent,))


def test_sum_and_mean_reduction_duality():
    rng = np.random.default_rng(23)
    x = rng.normal(size=(2, 5, 3))
    dx = rng.normal(size=x.shape)
    for kind in ("sum", "mean"):
        reducer = np.sum if kind == "sum" else np.mean
        tangent = reducer(dx, axis=1)
        w = rng.normal(size=tangent.shape)
        scale = 1.0 if kind == "sum" else 1.0 / x.shape[1]
        cotangent = np.broadcast_to(w[:, None, :], x.shape) * scale
        assert_forward_reverse_duality(tangent, w, (dx,), (cotangent,))


def test_fft_jvp_and_adjoint_duality_for_ortho_normalization():
    rng = np.random.default_rng(31)
    dx = rng.normal(size=16) + 1j * rng.normal(size=16)
    w = rng.normal(size=16) + 1j * rng.normal(size=16)
    tangent = np.fft.fft(dx, norm="ortho")
    cotangent = np.fft.ifft(w, norm="ortho")
    assert_forward_reverse_duality(tangent, w, (dx,), (cotangent,))


def test_dropout_tangent_replays_exact_philox_mask():
    seed, counter, p = 77, 13, 0.25
    dx = np.linspace(-2.0, 2.0, 33, dtype=np.float32)
    first = rng_device.dropout_mask(seed, dx.size, p, counter)
    second = rng_device.dropout_mask(seed, dx.size, p, counter)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(dx * first, dx * second)


def test_eggroll_fixed_key_map_is_linear_in_x():
    key = RNGKey.from_seed(9)
    factors = member_normal(key, epoch=3, pair=2, shape=(7 + 5, 1))
    b, a = factors[:7], factors[7:]
    rng = np.random.default_rng(41)
    x = rng.normal(size=(4, 7))
    dx = rng.normal(size=x.shape)
    scale = 0.03

    def correction(value):
        return scale * (value @ b) @ a.T

    actual = correction(dx)
    finite = central_directional_difference(correction, (x,), (dx,))
    assert_directional_close(actual, finite, rtol=1.0e-8, atol=1.0e-9)
    # The fixed-key correction is the explicit matrix M = scale * B A^T.
    # Its transpose is therefore M^T = scale * A B^T; prove the defining
    # inner-product identity independently of the forward finite difference.
    cotangent = rng.normal(size=actual.shape)
    input_cotangent = scale * (cotangent @ a) @ b.T
    assert_forward_reverse_duality(actual, cotangent, (dx,), (input_cotangent,))
    assert DERIVATIVE_PROOF_MATRIX["es_low_rank_correction"].kind is ProofKind.STOCHASTIC_REPLAY


def test_implicit_jvp_satisfies_residual_linearization():
    # R(x,u)=u^2-x=0; du=dx/(2u) must make R_x*dx+R_u*du vanish.
    x = np.array([1.0, 4.0, 9.0, 16.0])
    u = np.sqrt(x)
    dx = np.array([0.3, -0.2, 0.5, -0.7])
    du = dx / (2.0 * u)
    linearized_residual = -dx + 2.0 * u * du
    np.testing.assert_allclose(linearized_residual, 0.0, atol=1.0e-14)
