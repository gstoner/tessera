"""Reference certificates honor the same bias intent as typed Graph attention."""
import numpy as np
import pytest

from tessera import ops
from tessera.autodiff.jvp import _JVPS
from tessera.autodiff.vjp import _VJPS


def _oracle(q, k, v, bias):
    k = np.repeat(k, q.shape[1] // k.shape[1], axis=1)
    v = np.repeat(v, q.shape[1] // v.shape[1], axis=1)
    scores = (q @ k.swapaxes(-1, -2)) / np.sqrt(q.shape[-1]) + bias
    sq, sk = scores.shape[-2:]
    legal = np.arange(sk)[None, :] <= np.arange(sq)[:, None] + max(sk-sq, 0)
    scores = np.where(legal, scores, -np.inf)
    maximum = scores.max(axis=-1, keepdims=True)
    exponent = np.exp(scores-maximum)
    total = exponent.sum(axis=-1, keepdims=True)
    return exponent / total @ v, maximum[..., 0] + np.log(total[..., 0])


@pytest.mark.parametrize("bias_shape", [(1, 2, 3, 5), (1, 2, 1, 5)])
def test_bias_alias_saved_primal_jvp_and_vjp_use_same_scores(bias_shape):
    rng = np.random.default_rng(934)
    primals = tuple(rng.normal(0, .2, shape) for shape in
                    ((1, 2, 3, 4), (1, 1, 5, 4), (1, 1, 5, 3)))
    bias = rng.normal(0, .3, bias_shape)
    seeds = tuple(rng.normal(0, .1, value.shape) for value in primals)
    expected = _oracle(*primals, bias)
    actual = ops.flash_attn(*primals, bias=bias, causal=True, lse_checkpoint="saved")
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, reference, rtol=2e-6, atol=2e-7)
    primal, tangent = _JVPS["flash_attn"](
        primals, seeds, bias=bias, causal=True, lse_checkpoint="saved")
    h = 1e-5
    positive = _oracle(*(x+h*dx for x, dx in zip(primals, seeds, strict=True)), bias)
    negative = _oracle(*(x-h*dx for x, dx in zip(primals, seeds, strict=True)), bias)
    for value, plus, minus in zip(tangent, positive, negative, strict=True):
        np.testing.assert_allclose(value, (plus-minus)/(2*h), rtol=2e-6, atol=2e-8)
    for index in (0, 1):
        cotangent = rng.normal(size=expected[index].shape)
        gradients = _VJPS["flash_attn"](
            cotangent, *primals, bias=bias, causal=True,
            lse_checkpoint="saved", _output_index=index)
        assert len(gradients) == 3
        projected = sum(np.sum(g*dx) for g, dx in zip(gradients, seeds, strict=True))
        finite_difference = np.sum(cotangent*(positive[index]-negative[index]))/(2*h)
        np.testing.assert_allclose(projected, finite_difference, rtol=2e-6, atol=2e-8)
    for value, reference in zip(primal, expected, strict=True):
        np.testing.assert_allclose(value, reference, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("product", ["primal", "jvp", "vjp"])
def test_conflicting_bias_aliases_are_rejected(product):
    q = k = v = np.ones((1, 1, 2, 2))
    bias = np.zeros((1, 1, 2, 2))
    with pytest.raises(ValueError, match="only one of bias and attn_bias"):
        if product == "primal":
            ops.flash_attn(q, k, v, bias=bias, attn_bias=bias)
        elif product == "jvp":
            _JVPS["flash_attn"]((q, k, v), (q, k, v), bias=bias, attn_bias=bias)
        else:
            _VJPS["flash_attn"](q, q, k, v, bias=bias, attn_bias=bias)
