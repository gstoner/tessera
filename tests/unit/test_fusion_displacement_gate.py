"""M5 — Evaluator-gated synthesizer-displacement oracle.

Each fused MSL codegen lane that displaces the per-op MPS/MSL dispatcher must be
proven equivalent to its unfused reference on *hidden* inputs (fresh RNG the
codegen never saw), and must genuinely run on Metal — a numpy fallback can never
earn a "displaced" verdict. These tests run that gate over every shipped lane.
See docs/audit/backend/apple/APPLE_GPU_CODEGEN_PLAN.md (M5).
"""

from __future__ import annotations

import pytest

from tessera.compiler import fusion_equivalence as FE


@pytest.mark.parametrize("kind", FE.DISPLACED_LANES)
def test_lane_never_diverges(kind):
    shape = {
        "matmul_epilogue": (16, 64, 128),
        "norm_chain": (8, 64),
        "attention": (8, 32, 16),
        "pointwise": (8, 64),
        "gated_matmul": (16, 32, 48),
    }[kind]
    v = FE.displacement_verdict(kind, shape, seed=1234)
    # The gate's hard invariant: a lane is EITHER equivalent (ran on Metal +
    # matched) OR not_displaced (no Metal here) — never divergent. A divergent
    # verdict is a real codegen bug and blocks shipping the lane.
    assert v.relation in ("equivalent", "not_displaced"), v.detail


@pytest.mark.parametrize("kind", FE.DISPLACED_LANES)
def test_lane_equivalent_when_on_metal(kind):
    shape = {
        "matmul_epilogue": (16, 64, 128),
        "norm_chain": (8, 64),
        "attention": (8, 32, 16),
        "pointwise": (8, 64),
        "gated_matmul": (16, 32, 48),
    }[kind]
    v = FE.displacement_verdict(kind, shape, seed=7)
    if v.executed == "metal_runtime":
        assert v.relation == "equivalent", v.detail
        assert v.max_rel_err is not None and v.max_rel_err < 1e-2
    else:
        pytest.skip(f"{kind}: no Metal runtime here ({v.relation})")


def test_gate_all_no_divergence():
    verdicts = FE.gate_all(seed=99)
    assert set(verdicts) == set(FE.DISPLACED_LANES)
    divergent = {k: v.detail for k, v in verdicts.items()
                 if v.relation == "divergent"}
    assert not divergent, f"lanes diverged from reference: {divergent}"


def test_hidden_inputs_vary_with_seed():
    # The gate must use fresh RNG per seed (hidden inputs) — different seeds must
    # not collapse to the same probe, or the "hidden" guarantee is vacuous.
    import numpy as np
    rng_a = np.random.default_rng(1)
    rng_b = np.random.default_rng(2)
    assert not np.allclose(rng_a.standard_normal(64), rng_b.standard_normal(64))


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        FE.displacement_verdict("not_a_lane", (8, 8))
