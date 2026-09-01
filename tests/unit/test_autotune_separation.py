"""A margin smaller than the measurement's own noise is not a verdict.

The arbiter took one median per candidate and picked `min`, so a `MeasureRecord`
could assert a winner with nothing to say whether the difference meant anything.
It did. On sm_120 at 256x256x256 the two NVIDIA matmul lanes measured

    delegate      median 0.01300 ms   sd 14.5%
    emitted PTX   median 0.01057 ms   sd 39.1%

and the 18.7% gap was recorded as a clean 1.63x win. Against a 39.1% per-lane
spread it is not a result: that shape sits at the launch-overhead floor, where
run-to-run variation swamps the difference between kernels. The same comparison
at 2048^3 is a real 1.66x against spreads of 2.2% and 0.6% -- and a record that
stores only medians cannot tell the two apart.

A tie does not block dispatch; something still has to run. What it blocks is
*claiming* one candidate is faster, and re-picking by noise on every run.
"""
from __future__ import annotations

import pytest

from tessera.compiler.emit.autotune import (
    SEPARATION_FACTOR,
    MeasureRecord,
    relative_spread,
    separation_verdict,
)


# --------------------------------------------------------------------------
# The measured case that motivated this, both ends of it.
# --------------------------------------------------------------------------

def test_the_256_cubed_matmul_race_is_not_separated():
    """The verdict this file exists to prevent."""
    verdict = separation_verdict(
        {"nvidia_mma_gemm_shipped": 0.01300, "nvidia_ptx_gemm_emitted": 0.01057},
        {"nvidia_mma_gemm_shipped": 0.145, "nvidia_ptx_gemm_emitted": 0.391},
        "nvidia_ptx_gemm_emitted")
    assert verdict is not None
    assert verdict["separated"] is False
    assert verdict["margin"] == pytest.approx(0.187, abs=0.005)
    assert verdict["noise"] == pytest.approx(0.391)


def test_the_2048_cubed_matmul_race_is_separated():
    """The same 1.6x ratio, and here it is real -- the spreads are what differ."""
    verdict = separation_verdict(
        {"nvidia_mma_gemm_shipped": 2.45369, "nvidia_ptx_gemm_emitted": 1.47545},
        {"nvidia_mma_gemm_shipped": 0.022, "nvidia_ptx_gemm_emitted": 0.006},
        "nvidia_ptx_gemm_emitted")
    assert verdict["separated"] is True
    assert verdict["margin"] == pytest.approx(0.399, abs=0.005)


def test_a_ratio_alone_does_not_decide_it():
    """Guards against 'just threshold the ratio' -- the two rows above have
    nearly the same ratio and opposite verdicts."""
    tie = separation_verdict({"a": 1.30, "b": 1.057}, {"a": 0.145, "b": 0.391}, "b")
    real = separation_verdict({"a": 1.30, "b": 1.057}, {"a": 0.02, "b": 0.006}, "b")
    assert tie["margin"] == pytest.approx(real["margin"])
    assert (tie["separated"], real["separated"]) == (False, True)


# --------------------------------------------------------------------------
# The rule itself.
# --------------------------------------------------------------------------

def test_a_single_timed_candidate_has_no_margin_to_defend():
    """`None`, not `True`: a sole candidate is chosen by applicability, not by
    a race, so there is no comparison to have separated."""
    assert separation_verdict({"a": 1.0}, {"a": 0.0}, "a") is None


def test_untimed_candidates_are_not_counted_as_a_field():
    """`inf` means "could not be timed", not "was slower" -- a candidate that
    scored it never entered the race and cannot be the runner-up."""
    assert separation_verdict(
        {"a": 1.0, "b": float("inf"), "c": float("nan")}, {"a": 0.0}, "a") is None


def test_zero_noise_makes_any_positive_margin_separated():
    v = separation_verdict({"a": 1.0, "b": 2.0}, {"a": 0.0, "b": 0.0}, "a")
    assert v["separated"] is True


def test_identical_latencies_are_never_separated():
    v = separation_verdict({"a": 1.0, "b": 1.0}, {"a": 0.0, "b": 0.0}, "a")
    assert v["margin"] == 0.0 and v["separated"] is False


def test_the_noisier_of_the_two_lanes_sets_the_floor():
    """A quiet winner does not earn a verdict over a noisy runner-up: the gap
    has to clear whichever lane is less certain."""
    v = separation_verdict({"a": 1.0, "b": 1.3}, {"a": 0.001, "b": 0.30}, "a")
    assert v["noise"] == pytest.approx(0.30)
    assert v["separated"] is False


def test_the_runner_up_is_named():
    v = separation_verdict({"a": 1.0, "b": 2.0, "c": 5.0}, {}, "a")
    assert v["runner_up"] == "b", "the margin is against the second-fastest"


def test_the_factor_is_recorded_with_the_verdict():
    """So a later change to the bar is visible in old records rather than
    silently re-interpreting them."""
    v = separation_verdict({"a": 1.0, "b": 2.0}, {}, "a")
    assert v["factor"] == SEPARATION_FACTOR


# --------------------------------------------------------------------------
# relative_spread
# --------------------------------------------------------------------------

def test_spread_of_a_single_sample_is_zero_not_an_error():
    assert relative_spread([1.0]) == 0.0
    assert relative_spread([]) == 0.0


def test_spread_is_relative_so_it_compares_across_magnitudes():
    """The arbiter races candidates across four orders of magnitude; an
    absolute sd would make every fast lane look quiet."""
    small = relative_spread([1.0, 1.1, 0.9])
    large = relative_spread([1000.0, 1100.0, 900.0])
    assert small == pytest.approx(large)


def test_spread_ignores_non_finite_and_non_positive_samples():
    assert relative_spread([1.0, float("nan"), -1.0, 1.0]) == 0.0


# --------------------------------------------------------------------------
# The record: absence must not read as the favourable answer.
# --------------------------------------------------------------------------

def test_a_record_without_separation_reports_none_not_true():
    """Same rule as `unmeasured`: a record that never asked is not one that
    passed. A publisher must treat None and False alike."""
    assert MeasureRecord(winner="a", latency_ms=1.0).is_separated() is None


def test_a_separated_record_reports_true():
    rec = MeasureRecord(winner="a", latency_ms=1.0,
                        separation={"separated": True, "margin": 0.4})
    assert rec.is_separated() is True


def test_an_unseparated_record_reports_false():
    rec = MeasureRecord(winner="a", latency_ms=1.0,
                        separation={"separated": False, "margin": 0.19})
    assert rec.is_separated() is False


def test_separation_survives_a_json_round_trip():
    rec = MeasureRecord(winner="a", latency_ms=1.0, candidates={"a": 1.0, "b": 1.2},
                        separation={"separated": False, "margin": 0.17,
                                    "noise": 0.39, "runner_up": "b", "factor": 2.0})
    back = MeasureRecord.from_json(rec.as_json())
    assert back.separation == rec.separation
    assert back.is_separated() is False


def test_a_legacy_record_round_trips_without_inventing_a_verdict():
    legacy = {"winner": "a", "latency_ms": 1.0, "candidates": {"a": 1.0}}
    assert MeasureRecord.from_json(legacy).is_separated() is None


def test_separation_is_omitted_from_json_when_absent():
    """Keeps the committed corpus diffable and keeps 'absent' distinguishable
    from 'judged and found unseparated'."""
    assert "separation" not in MeasureRecord(winner="a", latency_ms=1.0).as_json()


# --------------------------------------------------------------------------
# End to end through `measured_arbitrate`, where the behaviour has consequences.
# --------------------------------------------------------------------------

import time  # noqa: E402

import numpy as np  # noqa: E402

from tessera.compiler.emit import autotune as AT  # noqa: E402
from tessera.compiler.emit.candidate import (  # noqa: E402
    OP_MATMUL,
    Candidate,
    Tier,
    register_candidate,
)


class _Region:
    dtype = "bfloat16"

    def reference(self, A, B):
        return np.asarray(A, np.float32) @ np.asarray(B, np.float32)


class _Cand(Candidate):
    """A candidate whose device timer reports a scripted sequence.

    The sequence is what makes a tie testable: `measured_arbitrate` now redoes
    the device measurement `device_repeats` times, so a candidate can be given
    a genuinely noisy series rather than one number.
    """

    op = OP_MATMUL

    def __init__(self, name, target, series, tier=Tier.EMITTED):
        self.name = name
        self.target = target
        self.tier = tier
        self._series = list(series)
        self._i = 0

    def run(self, region, A, B, *a, **k):
        return region.reference(A, B), "fake"

    def measure_device_latency(self, region, *inputs, reps=100, warmup=10):
        v = self._series[self._i % len(self._series)]
        self._i += 1
        return v


def _mm():
    rng = np.random.default_rng(0)
    return (rng.standard_normal((4, 4)).astype(np.float32),
            rng.standard_normal((4, 4)).astype(np.float32))


def _arbitrate(target, cache, **kw):
    A, B = _mm()
    return AT.measured_arbitrate(
        _Region(), OP_MATMUL, target, A, B, dims=(4, 4, 4), dtype="bfloat16",
        cache=cache, device="fakedev", timing=AT.TIMING_DEVICE, **kw)


def test_a_noisy_race_is_recorded_as_unseparated():
    """Two lanes whose spreads swamp an 18% gap -- the 256^3 shape, in a test."""
    tgt = "sep_noisy_target"
    register_candidate(_Cand("noisy_a", tgt, [0.0100, 0.0170, 0.0130]))
    register_candidate(_Cand("noisy_b", tgt, [0.0070, 0.0150, 0.0106]))
    cache = AT.MeasureCache()
    assert _arbitrate(tgt, cache) is not None
    rec = AT.MeasureRecord.from_json(cache.to_dict()["records"][0])
    assert rec.is_separated() is False
    assert rec.separation["noise"] > 0.2


def test_a_clean_race_is_recorded_as_separated():
    tgt = "sep_clean_target"
    register_candidate(_Cand("clean_slow", tgt, [2.450, 2.452, 2.454]))
    register_candidate(_Cand("clean_fast", tgt, [1.475, 1.476, 1.474]))
    cache = AT.MeasureCache()
    win = _arbitrate(tgt, cache)
    assert win.name == "clean_fast"
    rec = AT.MeasureRecord.from_json(cache.to_dict()["records"][0])
    assert rec.is_separated() is True


def test_an_unseparated_rerace_keeps_the_incumbent_instead_of_flipping():
    """A tie must not thrash the cached selection.

    Re-picking by noise flips the winner between runs and invalidates whatever
    downstream keyed off it, for no measured reason -- the selection would churn
    precisely where the measurement says there is nothing to choose.

    The re-race is provoked the way it happens in practice: an existing record
    that did not race the whole live field (here it never saw `inc_b`) is
    refused by `_record_raced_the_live_field`, so the arbiter re-times. `inc_b`
    then samples nominally faster, but both lanes are noisy enough that the
    margin cannot clear the floor -- so the incumbent stands.
    """
    tgt = "sep_incumbent_target"
    a = _Cand("inc_a", tgt, [0.0100, 0.0170, 0.0130])
    b = _Cand("inc_b", tgt, [0.0070, 0.0150, 0.0106])
    register_candidate(a)
    register_candidate(b)

    cache = AT.MeasureCache()
    key = ("fakedev", tgt, OP_MATMUL, AT.bucket_key((4, 4, 4), AT.SpecPolicy.BUCKET),
           "bfloat16", AT.TIMING_DEVICE)
    cache.put(key, AT.MeasureRecord(
        winner="inc_a", latency_ms=0.0100,
        candidates={"inc_a": 0.0100},      # never raced inc_b -> forces a re-race
        unmeasured={}))

    win = _arbitrate(tgt, cache)
    rec = AT.MeasureRecord.from_json(cache.to_dict()["records"][0])
    assert rec.is_separated() is False, "the fixture must produce a tie"
    assert win.name == "inc_a", (
        "an unseparated re-race must keep the incumbent, not adopt the "
        f"nominally-faster {rec.separation['runner_up']!r} by noise")
    assert rec.winner == "inc_a"


def test_a_separated_rerace_does_replace_the_incumbent():
    """The incumbent rule must not become a moat: a real win still displaces."""
    tgt = "sep_displace_target"
    register_candidate(_Cand("dis_a", tgt, [2.450, 2.452, 2.454]))
    register_candidate(_Cand("dis_b", tgt, [1.475, 1.476, 1.474]))
    cache = AT.MeasureCache()
    key = ("fakedev", tgt, OP_MATMUL, AT.bucket_key((4, 4, 4), AT.SpecPolicy.BUCKET),
           "bfloat16", AT.TIMING_DEVICE)
    cache.put(key, AT.MeasureRecord(winner="dis_a", latency_ms=2.450,
                                    candidates={"dis_a": 2.450}, unmeasured={}))
    win = _arbitrate(tgt, cache)
    assert win.name == "dis_b"
    assert AT.MeasureRecord.from_json(
        cache.to_dict()["records"][0]).is_separated() is True


# --------------------------------------------------------------------------
# The consumer. A verdict nothing reads is a declaration, not a check.
# --------------------------------------------------------------------------

def test_corpus_winner_refuses_an_unseparated_verdict():
    """`separation` was recorded, documented -- and read by nothing.

    Review on #670 caught it: `corpus_winner` validated that a record raced the
    live field (#655) and never asked whether the verdict was supported, so
    `run_arbitrated` kept dispatching on rows the corpus itself marks as noise.
    The committed sm_120 corpus holds a float16 device row whose 2.16% margin
    sits under 148.55% noise; before this check that row still changed a
    production route.

    Decision #29 names this exactly: a declaration with no consumer is worse
    than a missing one, because it reads as a closed contract in review.
    """
    tgt = "consumer_unsep_target"
    register_candidate(_Cand("cu_a", tgt, [1.0, 1.0, 1.0]))
    register_candidate(_Cand("cu_b", tgt, [1.2, 1.2, 1.2]))
    cache = AT.MeasureCache()
    key = ("fakedev", tgt, OP_MATMUL,
           AT.bucket_key((4, 4, 4), AT.SpecPolicy.BUCKET), "bfloat16",
           AT.TIMING_END_TO_END)
    rec = lambda sep: AT.MeasureRecord(                       # noqa: E731
        winner="cu_a", latency_ms=1.0,
        candidates={"cu_a": 1.0, "cu_b": 1.2}, unmeasured={}, separation=sep)

    A, B = _mm()
    ask = lambda: AT.corpus_winner(                            # noqa: E731
        _Region(), OP_MATMUL, tgt, A, B, dims=(4, 4, 4),
        dtype="bfloat16", cache=cache, device="fakedev")

    cache.put(key, rec({"separated": True, "margin": 0.4, "noise": 0.01}))
    assert ask() == "cu_a", "a supported verdict must still be usable"

    cache.put(key, rec({"separated": False, "margin": 0.02, "noise": 1.48}))
    assert ask() is None, (
        "a verdict the measurement says is noise must never become a dispatch "
        "hint")


def test_corpus_winner_refuses_a_selector_ineligible_row():
    """The finalizer's own signal, reached by a different mechanism.

    `finalize_test5_corpus` sets `selector_eligible` False when two independent
    runs disagree on the winner. That is the same conclusion `separation`
    reaches from a single run's spread, and it must bind the consumer too.
    """
    tgt = "consumer_ineligible_target"
    register_candidate(_Cand("ci_a", tgt, [1.0, 1.0, 1.0]))
    register_candidate(_Cand("ci_b", tgt, [1.2, 1.2, 1.2]))
    cache = AT.MeasureCache()
    key = ("fakedev", tgt, OP_MATMUL,
           AT.bucket_key((4, 4, 4), AT.SpecPolicy.BUCKET), "bfloat16",
           AT.TIMING_END_TO_END)
    cache.put(key, AT.MeasureRecord(
        winner="ci_a", latency_ms=1.0,
        candidates={"ci_a": 1.0, "ci_b": 1.2}, unmeasured={},
        separation={"separated": True, "margin": 0.4, "noise": 0.01},
        evidence={"selector_eligible": False, "stable_winner": False}))
    A, B = _mm()
    assert AT.corpus_winner(
        _Region(), OP_MATMUL, tgt, A, B, dims=(4, 4, 4), dtype="bfloat16",
        cache=cache, device="fakedev") is None


def test_a_legacy_row_without_a_verdict_is_still_usable():
    """`None` is allowed where `False` is refused, and the asymmetry is the
    point.

    None means the row predates the field and was never asked -- the state
    every row was in before #663. Rejecting it would silently deactivate most
    of the committed corpus as a side effect of adding a check. A row that is
    KNOWN unsupported is strictly worse than one that is merely unproven, and
    only the first is a regression to allow.
    """
    tgt = "consumer_legacy_target"
    register_candidate(_Cand("lg_a", tgt, [1.0, 1.0, 1.0]))
    register_candidate(_Cand("lg_b", tgt, [1.2, 1.2, 1.2]))
    cache = AT.MeasureCache()
    cache.put(("fakedev", tgt, OP_MATMUL,
               AT.bucket_key((4, 4, 4), AT.SpecPolicy.BUCKET), "bfloat16",
               AT.TIMING_END_TO_END),
              AT.MeasureRecord(winner="lg_a", latency_ms=1.0,
                               candidates={"lg_a": 1.0, "lg_b": 1.2},
                               unmeasured={}))
    A, B = _mm()
    assert AT.corpus_winner(
        _Region(), OP_MATMUL, tgt, A, B, dims=(4, 4, 4), dtype="bfloat16",
        cache=cache, device="fakedev") == "lg_a"
