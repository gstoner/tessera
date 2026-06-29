"""EBM / diffusion loss lane on x86 AVX-512 (P7 of S_SERIES_GAP_CLOSURE_PLAN) —
score_matching / denoising_score_matching / implicit_score_matching /
contrastive_divergence / persistent_cd / ddpm_noise_pred / vlb / load_balance.
All compose on the device binary + reduce kernels (no new kernel): the
diff/square + reductions run on AVX-512, the structure on the host. Reachable
via `compiler_path="x86_ebm_loss_compiled"`. Validated vs tessera.losses. Skip-
clean: libtessera_x86_elementwise.so not built.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import losses


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt, op, n_operands, kwargs):
    names = [f"a{i}" for i in range(n_operands)]
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_ebm_loss_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": kwargs}]})


def _run(rt, op, *arrs, **kwargs):
    res = rt.launch(_art(rt, op, len(arrs), kwargs), arrs)
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_ebm_loss_compiled"
    return np.asarray(res["output"])


_RNG = np.random.default_rng(23)


def _rn(*shape):
    return _RNG.standard_normal(shape).astype(np.float32)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ddpm_noise_pred(reduction):
    rt = _rt_or_skip()
    pred, true = _rn(4, 8), _rn(4, 8)
    got = _run(rt, "tessera.loss.ddpm_noise_pred", pred, true,
               reduction=reduction)
    ref = losses.ddpm_noise_pred_loss(pred, true, reduction=reduction)
    np.testing.assert_allclose(got, np.asarray(ref), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_score_matching(reduction):
    rt = _rt_or_skip()
    score, target = _rn(5, 6), _rn(5, 6)
    got = _run(rt, "tessera.loss.score_matching", score, target,
               reduction=reduction)
    ref = losses.score_matching_loss(score, target, reduction=reduction)
    np.testing.assert_allclose(got, np.asarray(ref), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_contrastive_divergence_and_persistent_cd(reduction):
    rt = _rt_or_skip()
    ep, en = _rn(7), _rn(7)
    np.testing.assert_allclose(
        _run(rt, "tessera.loss.contrastive_divergence", ep, en,
             reduction=reduction),
        np.asarray(losses.contrastive_divergence_loss(ep, en,
                                                      reduction=reduction)),
        rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        _run(rt, "tessera.loss.persistent_cd", ep, en, reduction=reduction),
        np.asarray(losses.persistent_cd_loss(ep, en, reduction=reduction)),
        rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_vlb(reduction):
    rt = _rt_or_skip()
    terms = _rn(4, 3)
    got = _run(rt, "tessera.loss.vlb", terms, reduction=reduction)
    ref = losses.vlb_loss(terms, reduction=reduction)
    np.testing.assert_allclose(got, np.asarray(ref), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_implicit_score_matching(reduction):
    rt = _rt_or_skip()
    score = _rn(6, 4)
    div = _rn(6)
    got = _run(rt, "tessera.loss.implicit_score_matching", score, div,
               reduction=reduction)
    ref = losses.implicit_score_matching_loss(score, div, reduction=reduction)
    np.testing.assert_allclose(got, np.asarray(ref), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_denoising_score_matching(reduction):
    rt = _rt_or_skip()
    s, yc, yn = _rn(5, 4), _rn(5, 4), _rn(5, 4)
    sigma = 0.7
    got = _run(rt, "tessera.loss.denoising_score_matching", s, yc, yn,
               sigma=sigma, reduction=reduction)
    ref = losses.denoising_score_matching_loss(s, yc, yn, sigma,
                                               reduction=reduction)
    np.testing.assert_allclose(got, np.asarray(ref), rtol=1e-4, atol=1e-4)


def test_denoising_rejects_nonpositive_sigma():
    rt = _rt_or_skip()
    s, yc, yn = _rn(2, 3), _rn(2, 3), _rn(2, 3)
    res = rt.launch(_art(rt, "tessera.loss.denoising_score_matching", 3,
                         {"sigma": 0.0}), (s, yc, yn))
    assert res["ok"] is False
    assert "sigma" in res.get("reason", "")


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_load_balance(reduction):
    rt = _rt_or_skip()
    # router probs (B, T, E), post-softmax
    logits = _rn(2, 8, 4)
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs = (probs / probs.sum(-1, keepdims=True)).astype(np.float32)
    got = _run(rt, "tessera.loss.load_balance_loss", probs, reduction=reduction)
    ref = losses.load_balance_loss(probs, reduction=reduction)
    np.testing.assert_allclose(got, np.asarray(ref), rtol=1e-4, atol=1e-4)
