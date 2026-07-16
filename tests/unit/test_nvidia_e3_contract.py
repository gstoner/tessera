"""Host-free E3 arbiter contract."""
from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.emit.candidate import OP_MATMUL, ArbiterError, Candidate, Tier, arbitrate, register_candidate, run_arbitrated


class _FakeReg:
    dtype = "bfloat16"
    def reference(self, a, b): return np.asarray(a, np.float32) @ np.asarray(b, np.float32)


class _FakeCand(Candidate):
    op = OP_MATMUL
    target = "e3_faketarget"
    def __init__(self, name, tier, tag): self.name, self.tier, self._tag = name, tier, tag
    def run(self, region, a, b, *args, **kwargs): return region.reference(a, b), self._tag


def test_e3_hand_tuned_wins_by_default_and_is_forceable():
    hand = _FakeCand("e3_handtuned", Tier.HAND_TUNED, "handtuned_real"); synth = _FakeCand("e3_synth", Tier.SYNTHESIZED, "synth_real")
    register_candidate(synth); register_candidate(hand); region = _FakeReg(); a = np.zeros((2, 2), np.float32); b = np.zeros((2, 2), np.float32)
    assert arbitrate(region, OP_MATMUL, "e3_faketarget", verify=False).name == "e3_handtuned"
    assert run_arbitrated(region, OP_MATMUL, "e3_faketarget", a, b, verify=False, force="e3_handtuned")[1] == "handtuned_real"
    assert run_arbitrated(region, OP_MATMUL, "e3_faketarget", a, b, verify=False, force="e3_synth")[1] == "synth_real"
    with pytest.raises(ArbiterError, match="not available"): arbitrate(region, OP_MATMUL, "e3_faketarget", verify=False, force="e3_nonexistent")
