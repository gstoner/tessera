"""NUMPOL-CARRIER-1 / FORGE §W5 — the precision-realizability oracle, checked
against a simulation rather than against numbers copied from a document.

Queue row 3b names this as its acceptance target: the fused-epilogue
fp32-accumulator verdict must be decided by the CARRIED `numeric_policy`
together with the state dtypes, not by a special case. It is the strongest
argument for the carrier existing at all — only a compiler sees the
accumulator contract and the optimizer-state dtypes at the same time.

The rows below re-derive FORGE_ASSESSMENT §1.3 by running the training loop,
so a wrong oracle fails here even if it happens to agree with the write-up.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.precision_realizability import (
    RealizabilityVerdict,
    fused_epilogue_realizability,
    significand_bits,
)

f32 = np.float32


def _bf16():
    return pytest.importorskip("ml_dtypes").bfloat16


def _round(x, dt):
    return np.asarray(np.asarray(x, dtype=dt), dtype=np.float64)


def _train(master_dt, state_dt, grad_store, fused, steps=20, n=4096, seed=0):
    """20 momentum-SGD steps against an fp64 shadow.

    The gradient is PRODUCED at accumulator precision by the matmul. The
    standard path writes it to `grad_store` before the optimizer reads it; the
    fused path hands it over at accumulator precision, so that write never
    happens. Everything else is identical, which is what makes the ratio
    attributable to the fusion and nothing else.
    """
    rs = np.random.RandomState(seed)
    w0 = rs.randn(n) * 0.1
    w, w_ref = _round(w0, master_dt), w0.copy()
    m, m_ref = _round(np.zeros(n), state_dt), np.zeros(n)
    lr, beta = 0.05, 0.9
    for _ in range(steps):
        g = rs.randn(n) * 0.01
        g_used = g if fused else _round(g, grad_store)
        m = _round(beta * m + (1 - beta) * g_used, state_dt)
        w = _round(w - lr * m, master_dt)
        m_ref = beta * m_ref + (1 - beta) * g
        w_ref = w_ref - lr * m_ref
    return float(np.linalg.norm(w - w_ref) / np.linalg.norm(w_ref))


def _measured_ratio(master_dt, state_dt, grad_store):
    return (_train(master_dt, state_dt, grad_store, fused=False)
            / _train(master_dt, state_dt, grad_store, fused=True))


POLICY = {"storage": "bf16", "accum": "fp32"}


def test_the_benefit_is_real_when_states_are_fp32():
    bf16 = _bf16()
    measured = _measured_ratio(f32, f32, bf16)
    assert measured > 50, f"simulation shows no benefit to detect ({measured})"

    verdict = fused_epilogue_realizability(
        POLICY, state_dtype="fp32", master_dtype="fp32")
    assert verdict.realizable
    # No number is quoted for the unmasked case, on purpose: how large the
    # benefit gets depends on gradient distribution and step count, which the
    # oracle does not model. Measured here at ~208x where the assessment
    # records 913x — same conclusion, different magnitude, so a quoted factor
    # would be a fabricated precision.
    assert verdict.expected_improvement is None
    assert "measure it" in verdict.explanation


def test_bf16_states_mask_the_benefit_entirely():
    """FORGE's own recipe. The paper measures its precision claim inside a
    configuration that hides it, and reports 4.4%."""
    bf16 = _bf16()
    measured = _measured_ratio(f32, bf16, bf16)
    assert measured < 2.0, measured

    verdict = fused_epilogue_realizability(
        POLICY, state_dtype="bf16", master_dtype="fp32")
    assert not verdict.realizable
    assert verdict.dominant_surviving == "optimizer state"
    # The estimate is sound exactly where it matters — the masked case, where
    # a surviving term dominates.
    assert abs(verdict.expected_improvement - measured) < 0.5, (
        verdict.expected_improvement, measured)


def test_bf16_weights_and_states_mask_it_too():
    bf16 = _bf16()
    measured = _measured_ratio(bf16, bf16, bf16)
    assert measured < 2.0, measured
    verdict = fused_epilogue_realizability(
        POLICY, state_dtype="bf16", master_dtype="bf16")
    assert not verdict.realizable
    assert abs(verdict.expected_improvement - measured) < 0.5


def test_the_verdict_flips_purely_on_state_dtype():
    """The row's actual claim: the verdict is a function of accum x state
    dtype. Same policy, same master weights — only the state dtype moves."""
    realizable = fused_epilogue_realizability(
        POLICY, state_dtype="fp32", master_dtype="fp32")
    masked = fused_epilogue_realizability(
        POLICY, state_dtype="bf16", master_dtype="fp32")
    assert realizable.realizable and not masked.realizable


def test_an_accumulator_no_wider_than_storage_removes_nothing():
    verdict = fused_epilogue_realizability(
        {"storage": "bf16", "accum": "bf16"},
        state_dtype="fp32", master_dtype="fp32")
    assert not verdict.realizable
    assert verdict.expected_improvement == 1.0
    assert "no more precise" in verdict.explanation


def test_the_policy_is_required_rather_than_defaulted():
    """The verdict IS the relationship between accum and storage, so an absent
    one makes the question unanswerable — not defaultable (#21a)."""
    with pytest.raises(ValueError, match="unanswerable"):
        fused_epilogue_realizability(
            {"storage": "bf16"}, state_dtype="fp32", master_dtype="fp32")
    with pytest.raises(ValueError, match="unanswerable"):
        fused_epilogue_realizability(
            None, state_dtype="fp32", master_dtype="fp32")


def test_an_unknown_dtype_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="unknown dtype"):
        fused_epilogue_realizability(
            POLICY, state_dtype="float128", master_dtype="fp32")


def test_significand_bits_distinguishes_equal_width_formats():
    """fp16 and bf16 are both 16 bits and are not interchangeable here."""
    assert significand_bits("fp16") == 11
    assert significand_bits("bf16") == 8
    assert significand_bits("tf32") == 11
    assert significand_bits("fp32") == 24


def test_the_diagnostic_names_what_to_fix():
    verdict = fused_epilogue_realizability(
        POLICY, state_dtype="bf16", master_dtype="fp32")
    text = verdict.diagnostic()
    assert "optimizer state" in text
    assert "bandwidth" in text          # says what the fusion IS still worth
    assert isinstance(verdict, RealizabilityVerdict)
