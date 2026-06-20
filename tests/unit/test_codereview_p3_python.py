"""P3 Python regressions from the full-source code review.

  * evaluator.evaluate — a *successful* native run that returns NaN/Inf or a
    wrong-shaped output was scored "unproven" (match=None); it must be a
    correctness "fail".  [FIX]
  * conformance_evaluator._JIT_CACHE — unbounded dict; now a bounded LRU.  [FIX]

Doc-only honesty fixes (data.prefetch/interleave, tokenizer_sentencepiece_compat,
cli/mlir._load_symbol) are verified by import, not asserted here.
"""

from __future__ import annotations

import numpy as np

from tessera.compiler import conformance_evaluator, evaluator


def test_evaluate_nan_output_is_failure_not_unproven(monkeypatch):
    def fake_launch(artifact, args):
        return {
            "execution_kind": "native_cpu",
            "runtime_status": "success",
            "output": np.array([np.nan, 1.0]),
        }

    monkeypatch.setattr("tessera.runtime.launch", fake_launch)

    class FakeFn:
        def runtime_artifact(self):
            return object()

    v = evaluator.evaluate("apple_cpu", FakeFn(), (), np.array([0.0, 1.0]))
    assert v.correctness == "fail", v


def test_evaluate_wrong_shape_is_failure(monkeypatch):
    def fake_launch(artifact, args):
        return {
            "execution_kind": "native_cpu",
            "runtime_status": "success",
            "output": np.zeros((3,)),  # oracle is (2,)
        }

    monkeypatch.setattr("tessera.runtime.launch", fake_launch)

    class FakeFn:
        def runtime_artifact(self):
            return object()

    v = evaluator.evaluate("apple_cpu", FakeFn(), (), np.zeros((2,)))
    assert v.correctness == "fail", v


def test_conformance_jit_cache_is_bounded(monkeypatch):
    # Stub out the (expensive) real jit so we exercise only the cache eviction.
    monkeypatch.setattr(conformance_evaluator.ts, "jit", lambda **_k: (lambda fn: fn))
    conformance_evaluator._JIT_CACHE.clear()
    cap = conformance_evaluator._JIT_CACHE_MAX
    for i in range(cap + 50):
        conformance_evaluator._jitted("matmul", f"target_{i}")
    assert len(conformance_evaluator._JIT_CACHE) <= cap
