"""Workstream #16 — opt-level checksum oracle (E2, DESIL checksum-across-opt-levels).

A correct compiler produces the same result at any opt level; DESIL compiles one
program at two levels and compares a stable checksum. This closes the checksum
half of EVALUATOR_PLAN §9.5's "opt-level checksum + MLIRod" follow-on (the MLIRod
grammar-generator remains a documented larger follow-on).
"""

from __future__ import annotations

import numpy as np

from tessera.compiler.evaluator import (
    opt_level_checksum, opt_level_equivalence, OptLevelVerdict)


# ── pure checksum: stable under benign noise, sensitive to real divergence ───


def test_checksum_identical_for_equal_arrays():
    a = np.array([1.0, 2.0, 3.5], np.float32)
    assert opt_level_checksum(a) == opt_level_checksum(a.copy())


def test_checksum_robust_to_subtolerance_noise():
    a = np.array([1.0, 2.0, 3.5], np.float64)
    b = a + 1e-7   # below the 4-decimal rounding floor
    assert opt_level_checksum(a) == opt_level_checksum(b)


def test_checksum_differs_for_real_divergence():
    a = np.array([1.0, 2.0, 3.5], np.float64)
    b = a.copy(); b[1] += 0.5   # a real difference
    assert opt_level_checksum(a) != opt_level_checksum(b)


def test_checksum_nonfinite_is_unstable():
    assert opt_level_checksum(np.array([1.0, np.inf])) == -1
    assert opt_level_checksum(np.array([np.nan, 2.0])) == -1


# ── equivalence verdict ───────────────────────────────────────────────────────


def test_equivalence_inconclusive_with_one_variant():
    class _Dead:
        def runtime_artifact(self):
            raise RuntimeError("does not run")
    v = opt_level_equivalence([("o0", _Dead())], (np.zeros((2, 2), np.float32),))
    assert v.relation == "inconclusive"
    assert not v.is_stable


def test_equivalence_stable_for_two_lowerings_of_same_matmul():
    import tessera
    # Two algebraically-identical lowerings of the same computation. If they run
    # natively, their checksums must agree (stable); if neither runs native here,
    # the verdict is honestly inconclusive — never a false "stable".
    @tessera.jit(target="apple_gpu")
    def v_fused(a, b):
        return tessera.ops.matmul(a, b)

    @tessera.jit(target="apple_gpu")
    def v_plain(a, b):
        return tessera.ops.matmul(a, b)

    rng = np.random.default_rng(0)
    a = rng.standard_normal((8, 8)).astype(np.float32)
    b = rng.standard_normal((8, 8)).astype(np.float32)
    verdict = opt_level_equivalence([("fused", v_fused), ("plain", v_plain)], (a, b))
    assert isinstance(verdict, OptLevelVerdict)
    assert verdict.relation in {"stable", "inconclusive"}  # never divergent here
    if verdict.relation == "stable":
        assert len(set(verdict.checksums)) == 1
