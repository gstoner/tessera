"""Phase A — F4 codegen-gated oracle for the pointwise-DAG synthesizer.

`verify_synthesized_pointwise` brings the pointwise path to parity with the
matmul-epilogue / gated / attention region kinds: before the runtime trusts a
synthesized pointwise kernel, it probes it against the unfused numpy reference
and refuses a divergent synthesizer (the reward-hack-rejection contract). This
gate is what makes lane-by-lane numpy displacement (Phases C/D) safe.

Cross-platform: on a non-Darwin host no Metal kernel runs, so the oracle is
trusted by construction (returns True) and these tests still exercise the
caching + plumbing.
"""

from __future__ import annotations

import numpy as np

from tessera.compiler import fusion as F

_RNG = np.random.default_rng(20260617)


def _gelu(v):
    t = np.clip(0.7978845608028654 * (v + 0.044715 * v**3), -30.0, 30.0)
    return 0.5 * v * (1.0 + np.tanh(t))


def _region():
    # mul(x,a) -> add(_,b) -> gelu  (the canonical 3-op DAG used elsewhere).
    return F.PointwiseGraphRegion(
        ops=(("mul", ("x", "a"), "m"), ("add", ("m", "b"), "s"),
             ("gelu", ("s",), "o")),
        inputs=("x", "a", "b"), output="o")


def test_correct_region_passes_oracle():
    assert F.verify_synthesized_pointwise(_region(), force=True) is True


def test_oracle_verdict_is_cached():
    region = _region()
    F._VERIFY_CACHE.pop(("P", region.ops, len(region.inputs)), None)
    first = F.verify_synthesized_pointwise(region, force=True)
    # Second call (no force) must hit the cache and agree.
    assert F.verify_synthesized_pointwise(region) is first
    assert ("P", region.ops, len(region.inputs)) in F._VERIFY_CACHE


def test_divergent_synthesizer_is_rejected(monkeypatch):
    """A synthesizer that returns a wrong-but-metal_runtime result must be
    refused by the oracle (verdict False), so the caller falls back."""
    region = _region()
    F._VERIFY_CACHE.pop(("P", region.ops, len(region.inputs)), None)

    def _bad_run(_region, probes):
        # Pretend the GPU ran and produced garbage.
        return np.zeros_like(np.asarray(probes[0])) + 999.0, "metal_runtime"

    monkeypatch.setattr("tessera.compiler.emit.apple_msl.run_pointwise_graph",_bad_run)
    assert F.verify_synthesized_pointwise(region, force=True) is False


def test_reference_only_host_is_trusted(monkeypatch):
    """When no synthesized kernel runs (reference path), the oracle trusts it."""
    region = _region()
    F._VERIFY_CACHE.pop(("P", region.ops, len(region.inputs)), None)

    def _ref_run(rgn, probes):
        return rgn.reference(*probes), "reference"

    monkeypatch.setattr("tessera.compiler.emit.apple_msl.run_pointwise_graph",_ref_run)
    assert F.verify_synthesized_pointwise(region, force=True) is True
