"""Every claim the loss docstrings make, as an executable check.

Correctness-audit finding M-3 (MSW-3): 20 of 34 public losses carried no
docstring. Documenting them is only half the job — an unchecked docstring
rots into a confident lie, and these particular functions are ones where
the plausible reading is the wrong one. So each non-obvious claim is
pinned here, and the last test refuses a newly added loss with no
docstring at all.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from tessera import losses as L


# --- kl_divergence: the argument order is inverse to PyTorch --------------

P = np.array([0.1, 0.2, 0.7])
Q = np.array([0.3, 0.3, 0.4])


def test_kl_divergence_is_kl_p_given_q_from_log_p_and_q():
    expected = float(np.sum(P * np.log(P / Q)))
    got = float(L.kl_divergence(np.log(P), Q, reduction="sum"))
    assert got == pytest.approx(expected, abs=1e-12)


def test_pytorch_argument_order_silently_computes_the_reverse_divergence():
    """The documented footgun, with the documented numbers.

    A caller porting `F.kl_div(input=log q, target=p)` across passes
    `(log q, p)` here and gets `KL(q||p)`. Nothing raises; the value is
    finite and looks reasonable. The docstring quotes 0.2274 vs 0.2008 —
    these assertions are what keep those numbers honest.
    """
    forward = float(L.kl_divergence(np.log(P), Q, reduction="sum"))
    reversed_ = float(L.kl_divergence(np.log(Q), P, reduction="sum"))
    assert forward == pytest.approx(0.2008, abs=5e-5)
    assert reversed_ == pytest.approx(0.2274, abs=5e-5)
    assert forward != pytest.approx(reversed_, abs=1e-3), "KL must be asymmetric here"
    assert np.isfinite(reversed_), "the wrong call must NOT raise — that is the hazard"


def test_kl_divergence_drops_zero_probability_terms_instead_of_nan():
    """`log_softmax` over masked logits produces -inf routinely."""
    p = np.array([0.0, 0.5, 0.5])
    with np.errstate(divide="ignore"):
        log_p = np.log(p)
    assert np.isneginf(log_p[0])
    got = float(L.kl_divergence(log_p, Q, reduction="sum"))
    assert np.isfinite(got)
    expected = float(np.sum(p[1:] * (np.log(p[1:]) - np.log(Q[1:]))))
    assert got == pytest.approx(expected, abs=1e-12)


# --- huber vs smooth_l1: related, not interchangeable ---------------------


@pytest.mark.parametrize("d", [0.5, 1.0, 2.0])
def test_huber_equals_delta_times_smooth_l1(d):
    e = np.linspace(-3.0, 3.0, 25)
    z = np.zeros_like(e)
    huber = np.asarray(L.huber_loss(e, z, delta=d, reduction="none"))
    smooth = np.asarray(L.smooth_l1_loss(e, z, beta=d, reduction="none"))
    np.testing.assert_allclose(huber, d * smooth, rtol=0, atol=1e-12)


def test_huber_and_smooth_l1_coincide_only_at_one():
    e = np.array([0.4, 1.7])
    z = np.zeros_like(e)
    same = np.allclose(np.asarray(L.huber_loss(e, z, delta=1.0, reduction="none")),
                       np.asarray(L.smooth_l1_loss(e, z, beta=1.0, reduction="none")))
    differ = np.allclose(np.asarray(L.huber_loss(e, z, delta=2.0, reduction="none")),
                         np.asarray(L.smooth_l1_loss(e, z, beta=2.0, reduction="none")))
    assert same and not differ


# --- the "takes logits" pair ----------------------------------------------


def test_binary_cross_entropy_consumes_logits_not_probabilities():
    logits = np.array([-2.0, 0.0, 3.0])
    targets = np.array([0.0, 1.0, 1.0])
    stable = np.maximum(logits, 0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))
    assert float(L.binary_cross_entropy_loss(logits, targets, reduction="sum")) == \
        pytest.approx(float(stable.sum()), abs=1e-12)


def test_binary_cross_entropy_stays_exact_where_log_sigmoid_underflows():
    """The stable form is why the docstring calls it exact for large |x|."""
    logits = np.array([-800.0, 800.0])
    targets = np.array([0.0, 1.0])
    out = np.asarray(L.binary_cross_entropy_loss(logits, targets, reduction="none"))
    assert np.all(np.isfinite(out)) and np.allclose(out, 0.0, atol=1e-300)


def test_label_smoothing_is_the_excluded_class_form_not_a_uniform_mix():
    """Which smoothing distribution, pinned by the difference between them.

    The docstring first said "mixes the one-hot target with the uniform
    distribution" (review on #697). It does not: it puts `1 - s` on the
    target and `s/(C-1)` on each OTHER class, leaving nothing extra on the
    target. The uniform mix would leave `1 - s + s/C` there. Both are
    finite and close -- 2.5476 vs 2.5510 at s=0.1 -- which is exactly why
    an unchecked docstring could carry the wrong one indefinitely.
    """
    rng = np.random.default_rng(2)
    logits = rng.standard_normal((1, 5))
    targets = np.array([2])
    s, C = 0.1, 5
    got = float(L.cross_entropy_loss(logits, targets, label_smoothing=s,
                                     reduction="sum"))

    shift = logits.max(axis=-1, keepdims=True)
    log_probs = (logits - shift
                 - np.log(np.sum(np.exp(logits - shift), axis=-1, keepdims=True)))
    lp = log_probs[0]
    off = lp.sum() - lp[2]
    excluded = float(-((1 - s) * lp[2] + (s / (C - 1)) * off))
    uniform = float(-((1 - s + s / C) * lp[2] + (s / C) * off))

    assert got == pytest.approx(excluded, abs=1e-12)
    assert got != pytest.approx(uniform, abs=1e-6), (
        "the two forms must be distinguishable, or this test proves nothing")


def test_label_smoothing_is_refused_for_probability_targets():
    """Integer targets only — the docstring's other corrected claim."""
    with pytest.raises(ValueError, match="not supported for probability targets"):
        L.cross_entropy_loss(np.zeros((1, 3)), np.full((1, 3), 1 / 3),
                             label_smoothing=0.1)


def test_cross_entropy_ignore_index_leaves_the_mean_undiluted():
    logits = np.zeros((3, 4))
    targets = np.array([1, -100, 2])
    kept = float(L.cross_entropy_loss(logits[[0, 2]], targets[[0, 2]]))
    with_ignored = float(L.cross_entropy_loss(logits, targets))
    assert with_ignored == pytest.approx(kept, abs=1e-12)


# --- the "it is exactly X" / "it is not X" claims -------------------------


def test_score_matching_is_half_mse_and_ddpm_is_mse():
    a = np.array([0.3, -1.2, 2.0])
    b = np.array([0.1, 0.4, 1.5])
    mse = float(L.mse_loss(a, b))
    assert float(L.score_matching_loss(a, b)) == pytest.approx(0.5 * mse, abs=1e-12)
    assert float(L.ddpm_noise_pred_loss(a, b)) == pytest.approx(mse, abs=1e-12)


def test_vlb_loss_only_reduces():
    terms = np.array([1.0, 2.0, 3.0])
    assert float(L.vlb_loss(terms, reduction="sum")) == pytest.approx(6.0)
    np.testing.assert_allclose(np.asarray(L.vlb_loss(terms, reduction="none")), terms)


def test_seq2seq_mean_divides_by_the_mask_sum_not_the_element_count():
    rng = np.random.default_rng(4)
    logits = rng.standard_normal((5, 4))
    targets = np.array([0, 1, 2, 3, 0])
    mask = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    per_token = np.asarray(L.cross_entropy_loss(logits, targets, reduction="none"))
    expected = float(np.sum(per_token * mask) / mask.sum())
    assert float(L.seq2seq_loss(logits, targets, mask)) == pytest.approx(expected, abs=1e-12)
    # ... and not the element-count mean, which is the tempting reading
    assert float(L.seq2seq_loss(logits, targets, mask)) != \
        pytest.approx(float(np.sum(per_token * mask) / mask.size), abs=1e-6)


def test_seq2seq_weighted_mean_is_invariant_to_the_scale_of_the_weights():
    """A weighted mean cannot depend on how the weights are scaled.

    It did: `max(sum(mask), 1.0)` divided every sub-unit mask sum by 1.0, so
    `[0.1, 0.1]` returned one fifth of the weighted mean (review on #697).
    The original mask test used 0/1 weights summing to 2 and never saw it —
    which is the whole lesson: the docstring said "float weights are
    allowed" and nothing exercised a float weight.
    """
    rng = np.random.default_rng(4)
    logits = rng.standard_normal((5, 4))
    targets = np.array([0, 1, 2, 3, 0])
    base = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    reference = float(L.seq2seq_loss(logits, targets, base))
    for scale in (0.1, 0.5, 2.0, 100.0):
        scaled = float(L.seq2seq_loss(logits, targets, base * scale))
        assert scaled == pytest.approx(reference, rel=1e-12), (
            f"scaling the mask by {scale} moved the loss")


def test_seq2seq_fully_padded_row_is_zero_not_a_division_by_zero():
    """The clamp's one defensible job, kept explicitly."""
    rng = np.random.default_rng(5)
    logits = rng.standard_normal((3, 4))
    targets = np.array([0, 1, 2])
    out = float(L.seq2seq_loss(logits, targets, np.zeros(3)))
    assert out == 0.0 and np.isfinite(out)


# --- target conventions that differ between two neighbouring losses -------


def test_contrastive_target_one_means_similar():
    a = np.array([[0.0, 0.0]])
    b = np.array([[3.0, 4.0]])          # distance 5
    similar = float(L.contrastive_loss(a, b, np.array([1.0]), margin=1.0))
    assert similar == pytest.approx(25.0, abs=1e-12)   # d**2, pulled together
    dissimilar = float(L.contrastive_loss(a, b, np.array([0.0]), margin=1.0))
    assert dissimilar == pytest.approx(0.0, abs=1e-12)  # beyond the margin


def test_cosine_embedding_reads_target_as_a_predicate():
    a = np.array([[1.0, 0.0]])
    b = np.array([[1.0, 0.0]])
    for positive in (1.0, 0.5, 7.0):
        assert float(L.cosine_embedding_loss(a, b, np.array([positive]))) == \
            pytest.approx(0.0, abs=1e-9)
    # -1 and 0 both mean "dissimilar" under `target > 0`
    assert float(L.cosine_embedding_loss(a, b, np.array([-1.0]))) == \
        pytest.approx(float(L.cosine_embedding_loss(a, b, np.array([0.0]))), abs=1e-12)


def test_cosine_embedding_handles_a_zero_vector_without_nan():
    out = float(L.cosine_embedding_loss(np.zeros((1, 2)), np.array([[1.0, 0.0]]),
                                        np.array([1.0])))
    assert np.isfinite(out)


# --- structural claims ----------------------------------------------------


def test_nt_xent_averages_multiple_positives_and_survives_having_none():
    z = np.eye(4)
    labels = np.array([0, 0, 0, 1])          # three positives for rows 0-2, none for 3
    per = np.asarray(L.nt_xent_loss(z, labels, reduction="none"))
    assert np.all(np.isfinite(per))
    assert per[3] == pytest.approx(0.0, abs=1e-12), "an anchor with no positive is 0, not NaN"


def test_wasserstein_is_the_order_statistic_matching():
    x = np.array([3.0, 1.0, 2.0])
    y = np.array([2.0, 5.0, 4.0])
    expected = float(np.mean(np.abs(np.sort(x) - np.sort(y))))
    assert float(L.wasserstein_distance(x, y)) == pytest.approx(expected, abs=1e-12)
    # permuting either input cannot change an order-statistic distance
    assert float(L.wasserstein_distance(x[::-1], y)) == pytest.approx(expected, abs=1e-12)


def test_js_divergence_is_symmetric_and_bounded_by_ln2():
    a = float(L.js_divergence(P, Q, reduction="sum"))
    b = float(L.js_divergence(Q, P, reduction="sum"))
    assert a == pytest.approx(b, abs=1e-12)
    disjoint = float(L.js_divergence(np.array([1.0, 0.0]), np.array([0.0, 1.0]),
                                     reduction="sum"))
    assert disjoint == pytest.approx(np.log(2.0), abs=1e-6)


def test_log_cosh_does_not_overflow_on_large_negative_error():
    out = float(L.log_cosh_loss(np.array([-400.0]), np.array([0.0]), reduction="sum"))
    assert np.isfinite(out) and out == pytest.approx(400.0 - np.log(2.0), abs=1e-9)


# --- the ratchet ----------------------------------------------------------


def test_every_public_loss_is_documented():
    """M-3's gate: it may shrink to zero and must never grow again."""
    source = Path(inspect.getsourcefile(L)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    undocumented = [
        n.name for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and not n.name.startswith("_")
        and not ast.get_docstring(n)
    ]
    assert undocumented == [], (
        f"public losses with no docstring: {undocumented}. Every one of these "
        "is reachable from `tessera.losses` and several have a convention a "
        "caller cannot guess (argument order, logits vs probabilities, which "
        "target value means 'similar')."
    )


def test_seq2seq_gradient_matches_the_fixed_forward_on_a_fractional_mask():
    """Forward and VJP must move together, or the gradient is silently wrong.

    The masked-mean denominator is duplicated in the forward, the VJP, the
    JVP and the runtime executor. Fixing one and not the others would trade
    a wrong loss for a wrong gradient, which is harder to notice. Finite
    differences against the fixed forward is what makes that concrete.
    """
    from tessera.autodiff import vjp as _vjp

    rng = np.random.default_rng(7)
    logits = rng.standard_normal((4, 3))
    targets = np.array([0, 2, 1, 0])
    mask = np.array([0.1, 0.1, 0.0, 0.2])      # sums to 0.4, below the old clamp

    rule = _vjp._VJPS["seq2seq_loss"]
    grad = np.asarray(rule(1.0, logits, targets, mask, reduction="mean")[0],
                      dtype=np.float64)

    h = 1e-6
    numeric = np.zeros_like(logits)
    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            up, dn = logits.copy(), logits.copy()
            up[i, j] += h
            dn[i, j] -= h
            numeric[i, j] = (float(L.seq2seq_loss(up, targets, mask))
                             - float(L.seq2seq_loss(dn, targets, mask))) / (2 * h)
    np.testing.assert_allclose(grad, numeric, rtol=1e-5, atol=1e-8)


def test_seq2seq_jvp_matches_the_fixed_forward_on_a_fractional_mask():
    # `tessera.autodiff.jvp` is the exported FUNCTION, not the module, so
    # the registry has to be reached through importlib.
    import importlib
    _jvp = importlib.import_module("tessera.autodiff.jvp")

    rng = np.random.default_rng(8)
    logits = rng.standard_normal((4, 3))
    targets = np.array([0, 2, 1, 0])
    mask = np.array([0.1, 0.1, 0.0, 0.2])
    tangent = rng.standard_normal(logits.shape)

    primal, tan = _jvp._JVPS["seq2seq_loss"](
        (logits, targets, mask), (tangent, None, None), reduction="mean")
    assert float(primal) == pytest.approx(
        float(L.seq2seq_loss(logits, targets, mask)), abs=1e-12)

    h = 1e-6
    numeric = (float(L.seq2seq_loss(logits + h * tangent, targets, mask))
               - float(L.seq2seq_loss(logits - h * tangent, targets, mask))) / (2 * h)
    assert float(tan) == pytest.approx(numeric, rel=1e-5, abs=1e-8)
