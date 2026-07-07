"""Workstream D1 (core) — candidate registry + accuracy-budgeted arbiter.

Host-free: the registry, tier-priority selection, F4-gate reuse, ``force`` escape
hatch (plan E3), the ``measure`` seam (plan D2), and the honest reference fallback
are all exercised with fake candidates + a real ``FusedRegion`` oracle — no device.
"""
from __future__ import annotations

import numpy as np
import pytest

import tessera.compiler.fusion as F
from tessera.compiler.emit import candidate as C
from tessera.compiler.emit.candidate import (
    OP_FUSED_REGION, ArbiterError, Candidate, Tier,
)

_TGT = "faketarget"


class _Correct(Candidate):
    """Returns the exact reference under a real tag — always F4-passes."""
    op = OP_FUSED_REGION

    def __init__(self, name, tier, *, applies=True, avail=True, atol=None):
        self.name, self.tier, self.target = name, tier, _TGT
        self._applies, self._avail, self.accuracy_atol = applies, avail, atol

    def available(self):
        return self._avail

    def applies_to(self, region):
        return self._applies

    def run(self, region, A, B, bias=None, *a, **k):
        return region.reference(A, B, bias), f"{self.name}_tag"


class _Wrong(_Correct):
    """Runs a real kernel that returns garbage — must be F4-rejected."""
    def run(self, region, A, B, bias=None, *a, **k):
        return np.full_like(region.reference(A, B, bias), 9.0), f"{self.name}_tag"


@pytest.fixture(autouse=True)
def _clean_registry():
    # Isolate each test: snapshot + restore the module-level registry and the F4
    # verdict cache so fakes never leak across tests (or into real backends).
    saved = {k: list(v) for k, v in C._CANDIDATES.items()}
    C._CANDIDATES.clear()
    F.clear_verification_cache()
    yield
    C._CANDIDATES.clear()
    C._CANDIDATES.update(saved)
    F.clear_verification_cache()


def _region():
    return F.FusedRegion(epilogue=("relu",))


def test_register_and_enumerate_in_order():
    a = _Correct("a", Tier.SYNTHESIZED)
    b = _Correct("b", Tier.HAND_TUNED)
    C.register_candidate(a)
    C.register_candidate(b)
    names = [c.name for c in C.candidates_for(_TGT, OP_FUSED_REGION)]
    assert names == ["a", "b"]


def test_reregister_same_name_replaces():
    C.register_candidate(_Correct("a", Tier.SYNTHESIZED))
    C.register_candidate(_Correct("a", Tier.HAND_TUNED))   # same name, new tier
    got = C.candidates_for(_TGT, OP_FUSED_REGION)
    assert len(got) == 1 and got[0].tier is Tier.HAND_TUNED


def test_default_prefers_highest_tier():
    C.register_candidate(_Correct("synth", Tier.SYNTHESIZED))
    C.register_candidate(_Correct("hand", Tier.HAND_TUNED))
    win = C.arbitrate(_region(), OP_FUSED_REGION, _TGT)
    assert win is not None and win.name == "hand"   # lead-safety (Decision #28)


def test_applies_and_available_filter_out():
    C.register_candidate(_Correct("hand", Tier.HAND_TUNED, applies=False))
    C.register_candidate(_Correct("emit", Tier.EMITTED, avail=False))
    C.register_candidate(_Correct("synth", Tier.SYNTHESIZED))
    win = C.arbitrate(_region(), OP_FUSED_REGION, _TGT)
    assert win is not None and win.name == "synth"  # only one left standing


def test_f4_gate_rejects_wrong_candidate():
    # The crown-jewel-tier candidate is a miscompiler; the arbiter must drop it and
    # fall to the correct lower tier — a faster-but-wrong kernel is never selected.
    C.register_candidate(_Wrong("hand", Tier.HAND_TUNED))
    C.register_candidate(_Correct("synth", Tier.SYNTHESIZED))
    win = C.arbitrate(_region(), OP_FUSED_REGION, _TGT)
    assert win is not None and win.name == "synth"


def test_verify_false_keeps_wrong_candidate():
    # With verification off, tier priority alone selects (used by the measured
    # loop once D2 owns correctness separately) — proves the gate is what refuses.
    C.register_candidate(_Wrong("hand", Tier.HAND_TUNED))
    win = C.arbitrate(_region(), OP_FUSED_REGION, _TGT, verify=False)
    assert win is not None and win.name == "hand"


def test_force_escape_hatch_selects_by_name():
    C.register_candidate(_Correct("synth", Tier.SYNTHESIZED))
    C.register_candidate(_Correct("hand", Tier.HAND_TUNED))
    win = C.arbitrate(_region(), OP_FUSED_REGION, _TGT, force="synth")
    assert win is not None and win.name == "synth"   # forced past the higher tier


def test_force_unknown_raises():
    C.register_candidate(_Correct("synth", Tier.SYNTHESIZED))
    with pytest.raises(ArbiterError, match="nope"):
        C.arbitrate(_region(), OP_FUSED_REGION, _TGT, force="nope")


def test_force_unavailable_raises():
    C.register_candidate(_Correct("hand", Tier.HAND_TUNED, avail=False))
    with pytest.raises(ArbiterError, match="hand"):
        C.arbitrate(_region(), OP_FUSED_REGION, _TGT, force="hand")


def test_measure_hook_overrides_tier():
    # D2 seam: a latency callback (lowest wins) overrides tier priority, so the
    # generic lane *can* beat the crown jewel when it measures faster + in budget.
    C.register_candidate(_Correct("synth", Tier.SYNTHESIZED))
    C.register_candidate(_Correct("hand", Tier.HAND_TUNED))
    latency = {"synth": 1.0, "hand": 9.0}
    win = C.arbitrate(_region(), OP_FUSED_REGION, _TGT,
                      measure=lambda c: latency[c.name])
    assert win is not None and win.name == "synth"


def test_run_arbitrated_falls_back_to_reference():
    # No applicable candidate → honest numpy reference, tagged "reference".
    C.register_candidate(_Correct("hand", Tier.HAND_TUNED, applies=False))
    region = _region()
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 12)).astype(np.float32)
    B = rng.standard_normal((12, 16)).astype(np.float32)
    out, tag = C.run_arbitrated(region, OP_FUSED_REGION, _TGT, A, B, None)
    assert tag == "reference"
    assert np.allclose(out, region.reference(A, B, None), atol=1e-6)


def test_run_arbitrated_runs_winner():
    C.register_candidate(_Correct("hand", Tier.HAND_TUNED))
    region = _region()
    rng = np.random.default_rng(1)
    A = rng.standard_normal((8, 12)).astype(np.float32)
    B = rng.standard_normal((12, 16)).astype(np.float32)
    out, tag = C.run_arbitrated(region, OP_FUSED_REGION, _TGT, A, B, None)
    assert tag == "hand_tag"
    assert np.allclose(out, region.reference(A, B, None), atol=1e-6)


def test_accuracy_budget_admits_f16_grade_candidate():
    # A candidate a hair past the 1e-3 oracle default but inside its declared 5e-3
    # budget must F4-pass (the accuracy-budgeted arbiter, Decision #28 / D2).

    class _F16(_Correct):
        def run(self, region, A, B, bias=None, *a, **k):
            return region.reference(A, B, bias) + np.float32(3e-3), "f16_tag"

    within = _F16("hand", Tier.HAND_TUNED, atol=5e-3)
    nobudget = _F16("synth", Tier.SYNTHESIZED, atol=None)
    assert C.verify_candidate(within, _region()) is True
    assert C.verify_candidate(nobudget, _region()) is False
