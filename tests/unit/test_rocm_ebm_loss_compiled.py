"""EBM / diffusion loss lane on AMD ROCm gfx1151 (P7 of
S_SERIES_GAP_CLOSURE_PLAN) — score_matching / denoising_score_matching /
implicit_score_matching / contrastive_divergence / persistent_cd /
ddpm_noise_pred / vlb / load_balance. All compose on the gfx1151 binary +
reduce kernels (no new kernel): the diff/square + reductions run on-device, the
structure on the host. Reachable via `compiler_path="rocm_ebm_loss_compiled"`.
Validated vs tessera.losses on gfx1151. Skip-clean: tessera-opt not built / no
GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import losses


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _art(rt, op, n_operands, kwargs):
    names = [f"a{i}" for i in range(n_operands)]
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_ebm_loss_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": kwargs}]})


def _run(rt, op, *arrs, **kwargs):
    res = rt.launch(_art(rt, op, len(arrs), kwargs), arrs)
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_ebm_loss_compiled"
    return np.asarray(res["output"])


_RNG = np.random.default_rng(23)


def _rn(*shape):
    return _RNG.standard_normal(shape).astype(np.float32)


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_ddpm_and_score_matching(reduction):
    rt = _rocm_or_skip()
    a, b = _rn(4, 8), _rn(4, 8)
    np.testing.assert_allclose(
        _run(rt, "tessera.loss.ddpm_noise_pred", a, b, reduction=reduction),
        np.asarray(losses.ddpm_noise_pred_loss(a, b, reduction=reduction)),
        rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        _run(rt, "tessera.loss.score_matching", a, b, reduction=reduction),
        np.asarray(losses.score_matching_loss(a, b, reduction=reduction)),
        rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_contrastive_divergence_and_vlb(reduction):
    rt = _rocm_or_skip()
    ep, en = _rn(7), _rn(7)
    np.testing.assert_allclose(
        _run(rt, "tessera.loss.contrastive_divergence", ep, en,
             reduction=reduction),
        np.asarray(losses.contrastive_divergence_loss(ep, en,
                                                      reduction=reduction)),
        rtol=1e-4, atol=1e-4)
    terms = _rn(4, 3)
    np.testing.assert_allclose(
        _run(rt, "tessera.loss.vlb", terms, reduction=reduction),
        np.asarray(losses.vlb_loss(terms, reduction=reduction)),
        rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_implicit_and_denoising(reduction):
    rt = _rocm_or_skip()
    score, div = _rn(6, 4), _rn(6)
    np.testing.assert_allclose(
        _run(rt, "tessera.loss.implicit_score_matching", score, div,
             reduction=reduction),
        np.asarray(losses.implicit_score_matching_loss(score, div,
                                                       reduction=reduction)),
        rtol=1e-4, atol=1e-4)
    s, yc, yn = _rn(5, 4), _rn(5, 4), _rn(5, 4)
    np.testing.assert_allclose(
        _run(rt, "tessera.loss.denoising_score_matching", s, yc, yn,
             sigma=0.7, reduction=reduction),
        np.asarray(losses.denoising_score_matching_loss(s, yc, yn, 0.7,
                                                        reduction=reduction)),
        rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_load_balance(reduction):
    rt = _rocm_or_skip()
    logits = _rn(2, 8, 4)
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs = (probs / probs.sum(-1, keepdims=True)).astype(np.float32)
    np.testing.assert_allclose(
        _run(rt, "tessera.loss.load_balance", probs, reduction=reduction),
        np.asarray(losses.load_balance_loss(probs, reduction=reduction)),
        rtol=1e-4, atol=1e-4)
