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


def _pkey(region):
    # B3: verify verdicts are keyed by backend identity too. The default path
    # resolves the registered active runner (Apple here).
    from tessera.compiler.emit.kernel_emitter import active_runner
    return (active_runner().target, "P", region.ops, len(region.inputs))


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
    F.clear_verification_cache()
    first = F.verify_synthesized_pointwise(region, force=True)
    # Second call (no force) must hit the cache and agree.
    assert F.verify_synthesized_pointwise(region) is first
    assert _pkey(region) in F._VERIFY_CACHE


def test_divergent_synthesizer_is_rejected(monkeypatch):
    """A synthesizer that returns a wrong-but-metal_runtime result must be
    refused by the oracle (verdict False), so the caller falls back."""
    region = _region()
    F.clear_verification_cache()

    def _bad_run(_region, probes):
        # Pretend the GPU ran and produced garbage.
        return np.zeros_like(np.asarray(probes[0])) + 999.0, "metal_runtime"

    monkeypatch.setattr("tessera.compiler.emit.apple_msl.run_pointwise_graph",_bad_run)
    assert F.verify_synthesized_pointwise(region, force=True) is False


def test_reference_only_host_is_trusted(monkeypatch):
    """When no synthesized kernel runs (reference path), the oracle trusts it."""
    region = _region()
    F.clear_verification_cache()

    def _ref_run(rgn, probes):
        return rgn.reference(*probes), "reference"

    monkeypatch.setattr("tessera.compiler.emit.apple_msl.run_pointwise_graph",_ref_run)
    assert F.verify_synthesized_pointwise(region, force=True) is True


# ── the declared relative budget reaches the matmul + pointwise verifiers ────
#
# `_effective_rtol` exists to consume a candidate's `accuracy_rtol`, and the
# region/attention/gated verifiers all pass it. These two dropped it, so a
# candidate that declares a relative budget was judged at numpy's default
# rtol=1e-5 and a numerically correct low-precision GEMM could be rejected as a
# miscompile and starved by the arbiter.


class _BudgetRunner:
    """Candidate adapter that returns a result off by a fixed RELATIVE amount,
    with a declared relative budget and no absolute one."""

    target = "budget_probe"
    accuracy_atol = None

    def __init__(self, rel_error, accuracy_rtol=1e-2):
        self.rel_error = rel_error
        self.accuracy_rtol = accuracy_rtol

    def run_matmul(self, region, A, B):
        return region.reference(A, B) * (1.0 + self.rel_error), "budget_probe"

    def run_pointwise_graph(self, region, probes):
        return region.reference(*probes) * (1.0 + self.rel_error), "budget_probe"


def test_matmul_oracle_honors_declared_relative_budget():
    F.clear_verification_cache()
    # 5e-3 relative on a probe whose entries are ~O(1) is far outside the
    # default atol=1e-3, so only the declared rtol can accept it.
    assert F.verify_synthesized_matmul(
        F.MatmulRegion(), force=True, runner=_BudgetRunner(5e-3)) is True


def test_matmul_oracle_still_rejects_out_of_budget_error():
    F.clear_verification_cache()
    assert F.verify_synthesized_matmul(
        F.MatmulRegion(), force=True, runner=_BudgetRunner(0.5)) is False


def test_pointwise_oracle_honors_declared_relative_budget():
    F.clear_verification_cache()
    assert F.verify_synthesized_pointwise(
        _region(), force=True, runner=_BudgetRunner(5e-3)) is True


def test_pointwise_oracle_still_rejects_out_of_budget_error():
    F.clear_verification_cache()
    assert F.verify_synthesized_pointwise(
        _region(), force=True, runner=_BudgetRunner(0.5)) is False
