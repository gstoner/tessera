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


def test_vjp_moe_expert_segmentation_matches_the_per_token_reference():
    """P3: `vjp_moe` accumulated the backward one token at a time — a (D,)@(D,E)
    matvec plus an `np.outer` per token, so 2*T tiny BLAS dispatches. Routing is
    a fixed integer partition, so the same arithmetic is one batched GEMM pair
    per expert. This pins that the restructuring is arithmetically the same
    thing, against the per-token form it replaced.
    """
    from tessera.autodiff.vjp import _VJPS

    def per_token_reference(dout, x, experts, route):
        experts = np.asarray(experts, dtype=np.float64)
        tokens = np.asarray(x, dtype=np.float64).reshape(-1, x.shape[-1])
        dout = np.asarray(dout, dtype=np.float64).reshape(
            tokens.shape[0], experts.shape[2])
        dx = np.zeros_like(tokens)
        dE = np.zeros_like(experts)
        for i in range(tokens.shape[0]):
            e = int(route[i])
            dx[i] = dout[i] @ experts[e].T
            dE[e] += np.outer(tokens[i], dout[i])
        return dx.reshape(x.shape), dE

    rng = np.random.default_rng(0)
    for n_tokens, d_model, d_out, n_experts in [(64, 16, 12, 4), (33, 8, 8, 5),
                                                (7, 4, 6, 3)]:
        x = rng.standard_normal((n_tokens, d_model))
        experts = rng.standard_normal((n_experts, d_model, d_out))
        dout = rng.standard_normal((n_tokens, d_out))
        # Includes an expert with no tokens routed to it — an empty segment
        # must contribute a zero block, not be skipped into stale memory.
        route = rng.integers(0, n_experts - 1, size=n_tokens)

        dx, dE = _VJPS["moe"](dout, x, experts, route=route)
        want_dx, want_dE = per_token_reference(dout, x, experts, route)
        np.testing.assert_allclose(dx, want_dx, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(dE, want_dE, rtol=1e-12, atol=1e-12)
        assert np.all(dE[n_experts - 1] == 0.0), "unrouted expert must be zero"
