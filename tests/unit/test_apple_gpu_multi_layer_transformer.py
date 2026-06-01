"""Multi-layer transformer chain — structural / correctness guard.

Companion test to ``benchmarks/apple_gpu/benchmark_multi_layer_transformer.py``.
The benchmark measures wall-clock; this test validates the chain's
*correctness* + the single-cb invariant under multi-layer scale.

Specifically:

* **Single command buffer for N layers** — a 4-layer transformer
  step (~50 user ops) commits exactly 1 cb when run under
  ``@auto_batch`` with ``ResidentWeights``.
* **Output is finite + correctly shaped** — every layer's bmm /
  rope / flash_attn / norm produces well-formed output (no NaN,
  no inf, no shape drift).
* **ResidentWeights handle stability** — weight DeviceTensor
  handles stay constant across multiple decode steps (no
  re-upload per step).
* **Steady-state determinism** — running the same step twice
  produces the same output.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.apple_gpu_batched import session_available, session_commit_count
from tessera.apple_gpu_resident import ResidentWeights
import tessera.apple_gpu_ops as agpu


def test_four_layer_transformer_commits_one_cb_per_step():
    if not session_available():
        pytest.skip("encode-session unavailable")
    B, S, D = 1, 16, 32
    FFD = 2 * D
    N = 4  # number of layers
    scale = 1.0 / np.sqrt(D)
    eps = 1e-5

    rng = np.random.default_rng(0xDEC0DEFF)
    X = rng.standard_normal((B * S, D), dtype=np.float32) * 0.1

    cache = ResidentWeights()
    try:
        for i in range(N):
            cache.weight(f"L{i}_gamma_a",
                          rng.standard_normal((D,), dtype=np.float32))
            cache.weight(f"L{i}_Wq",
                          rng.standard_normal((1, D, D),
                                               dtype=np.float32) * 0.05)
            cache.weight(f"L{i}_Wk",
                          rng.standard_normal((1, D, D),
                                               dtype=np.float32) * 0.05)
            cache.weight(f"L{i}_Wv",
                          rng.standard_normal((1, D, D),
                                               dtype=np.float32) * 0.05)
            cache.weight(f"L{i}_Wo",
                          rng.standard_normal((1, D, D),
                                               dtype=np.float32) * 0.05)
            cache.weight(f"L{i}_Theta",
                          (np.arange(B * S * D, dtype=np.float32) * 0.001
                           * (1 + i)).reshape(B * S, D))
            cache.weight(f"L{i}_gamma_m",
                          rng.standard_normal((D,), dtype=np.float32))
            cache.weight(f"L{i}_Wgate",
                          rng.standard_normal((1, D, FFD),
                                               dtype=np.float32) * 0.05)
            cache.weight(f"L{i}_Wdown",
                          rng.standard_normal((1, FFD, D),
                                               dtype=np.float32) * 0.05)

        @agpu.auto_batch
        def step(x):
            x_t = x
            for i in range(N):
                # Attention sub-block — 8 ops.
                n = agpu.rmsnorm(x_t, cache[f"L{i}_gamma_a"],
                                  rows=B * S, cols=D, eps=eps)
                q = agpu.bmm(n, cache[f"L{i}_Wq"],
                              batch=1, M=B * S, N=D, K=D)
                k = agpu.bmm(n, cache[f"L{i}_Wk"],
                              batch=1, M=B * S, N=D, K=D)
                v = agpu.bmm(n, cache[f"L{i}_Wv"],
                              batch=1, M=B * S, N=D, K=D)
                q_r = agpu.rope(q, cache[f"L{i}_Theta"], M=B * S, K=D)
                k_r = agpu.rope(k, cache[f"L{i}_Theta"], M=B * S, K=D)
                a = agpu.flash_attn(q_r, k_r, v,
                                     B=B, Sq=S, Sk=S, D=D, scale=scale)
                x_t = agpu.bmm(a, cache[f"L{i}_Wo"],
                                batch=1, M=B * S, N=D, K=D)
                # Simplified MLP sub-block — 4 ops.
                m_n = agpu.rmsnorm(x_t, cache[f"L{i}_gamma_m"],
                                    rows=B * S, cols=D, eps=eps)
                gate = agpu.bmm(m_n, cache[f"L{i}_Wgate"],
                                 batch=1, M=B * S, N=FFD, K=D)
                act = agpu.silu(gate, n=B * S * FFD)
                x_t = agpu.bmm(act, cache[f"L{i}_Wdown"],
                                batch=1, M=B * S, N=D, K=FFD)
            return x_t

        x_dev = cache.activation("x", X)
        before = session_commit_count()
        out = step(x_dev)
        after = session_commit_count()
        # ONE command buffer for N=4 layers × 12 ops = 48 user ops.
        assert (after - before) == 1, (
            f"4-layer transformer step expected 1 cb commit, got "
            f"delta={after - before}")
        arr = out.download(np.float32, (1, B * S, D))
        out.free()
        assert arr.shape == (1, B * S, D)
        assert np.isfinite(arr).all(), (
            f"output contains NaN/inf — {(~np.isfinite(arr)).sum()} bad "
            f"values out of {arr.size}")
    finally:
        cache.free()


def test_resident_weights_handles_stable_across_decode_steps():
    """Run 3 decode steps; weight DeviceTensor handles must stay
    constant (no re-upload). This is the steady-state contract that
    matters for real LLM decode."""
    if not session_available():
        pytest.skip("encode-session unavailable")
    B, S, D = 1, 8, 16
    FFD = 2 * D
    N = 2
    scale = 1.0 / np.sqrt(D)
    eps = 1e-5

    rng = np.random.default_rng(0xDEC0DEAA)
    Xs = [rng.standard_normal((B * S, D), dtype=np.float32) * 0.1
          for _ in range(3)]

    cache = ResidentWeights()
    try:
        for i in range(N):
            cache.weight(f"L{i}_gamma_a",
                          rng.standard_normal((D,), dtype=np.float32))
            cache.weight(f"L{i}_Wq",
                          rng.standard_normal((1, D, D),
                                               dtype=np.float32) * 0.05)
            cache.weight(f"L{i}_Wk",
                          rng.standard_normal((1, D, D),
                                               dtype=np.float32) * 0.05)
            cache.weight(f"L{i}_Wv",
                          rng.standard_normal((1, D, D),
                                               dtype=np.float32) * 0.05)
            cache.weight(f"L{i}_Wo",
                          rng.standard_normal((1, D, D),
                                               dtype=np.float32) * 0.05)
            cache.weight(f"L{i}_Theta",
                          (np.arange(B * S * D, dtype=np.float32) * 0.001
                           * (1 + i)).reshape(B * S, D))

        # Snapshot weight handles before any step.
        weight_handles_pre = {
            name: cache[name].handle for name in cache.weight_names()}

        @agpu.auto_batch
        def step(x):
            x_t = x
            for i in range(N):
                n = agpu.rmsnorm(x_t, cache[f"L{i}_gamma_a"],
                                  rows=B * S, cols=D, eps=eps)
                q = agpu.bmm(n, cache[f"L{i}_Wq"],
                              batch=1, M=B * S, N=D, K=D)
                k = agpu.bmm(n, cache[f"L{i}_Wk"],
                              batch=1, M=B * S, N=D, K=D)
                v = agpu.bmm(n, cache[f"L{i}_Wv"],
                              batch=1, M=B * S, N=D, K=D)
                q_r = agpu.rope(q, cache[f"L{i}_Theta"], M=B * S, K=D)
                k_r = agpu.rope(k, cache[f"L{i}_Theta"], M=B * S, K=D)
                a = agpu.flash_attn(q_r, k_r, v,
                                     B=B, Sq=S, Sk=S, D=D, scale=scale)
                x_t = agpu.bmm(a, cache[f"L{i}_Wo"],
                                batch=1, M=B * S, N=D, K=D)
            return x_t

        for X in Xs:
            x_dev = cache.activation("x", X)
            out = step(x_dev)
            arr = out.download(np.float32, (1, B * S, D))
            out.free()
            assert np.isfinite(arr).all()
            # Weight handles must be identical to the pre-step snapshot.
            for name, h in weight_handles_pre.items():
                assert cache[name].handle == h, (
                    f"weight {name!r} handle changed across decode "
                    f"steps — re-upload happened (regression)")
    finally:
        cache.free()


def test_multi_layer_steady_state_is_deterministic():
    """Same X in twice → same output out twice."""
    if not session_available():
        pytest.skip("encode-session unavailable")
    B, S, D = 1, 8, 16
    N = 2
    scale = 1.0 / np.sqrt(D)
    eps = 1e-5

    rng = np.random.default_rng(0xDE7E70)
    X = rng.standard_normal((B * S, D), dtype=np.float32) * 0.1
    cache = ResidentWeights()
    try:
        for i in range(N):
            cache.weight(f"g{i}",
                          rng.standard_normal((D,), dtype=np.float32))
            cache.weight(f"Wq{i}",
                          rng.standard_normal((1, D, D),
                                               dtype=np.float32) * 0.05)
            cache.weight(f"Wo{i}",
                          rng.standard_normal((1, D, D),
                                               dtype=np.float32) * 0.05)

        @agpu.auto_batch
        def step(x):
            x_t = x
            for i in range(N):
                n = agpu.rmsnorm(x_t, cache[f"g{i}"],
                                  rows=B * S, cols=D, eps=eps)
                proj = agpu.bmm(n, cache[f"Wq{i}"],
                                 batch=1, M=B * S, N=D, K=D)
                x_t = agpu.bmm(proj, cache[f"Wo{i}"],
                                batch=1, M=B * S, N=D, K=D)
            return x_t

        out1 = step(cache.activation("x", X)).download(
            np.float32, (1, B * S, D))
        # Re-upload SAME X.
        out2 = step(cache.activation("x", X)).download(
            np.float32, (1, B * S, D))
        np.testing.assert_array_equal(out1, out2)
    finally:
        cache.free()
