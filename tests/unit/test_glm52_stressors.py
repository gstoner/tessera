"""Low-cost stressors for the GLM-5.2 contract slice."""

from __future__ import annotations

import numpy as np

from benchmarks.rl.benchmark_glm52_serving_pressure import run_scaled_glm52_serving_pressure
from tessera.models import glm5
from tessera.models import moe_transformer as mt
from tessera.speculative import rejection_verify_chain


def test_deterministic_topk_stress_ties_are_stable():
    rng = np.random.default_rng(10)
    scores = rng.integers(0, 5, size=(64, 256)).astype(np.float64)
    idx = mt.deterministic_topk_indices(scores, 16)
    assert idx.shape == (64, 16)
    for row, selected in zip(scores, idx):
        sorted_idx = sorted(range(row.shape[0]), key=lambda i: (-row[i], i))[:16]
        np.testing.assert_array_equal(selected, np.asarray(sorted_idx))


def test_glm52_scaled_all_layers_build_under_indexshare():
    cfg = glm5.scaled_config()
    modes = []
    for layer in range(cfg.num_layers):
        graph = mt.build_block(cfg, layer_index=layer)
        modes.append(graph.find("deepseek_sparse_attention").attrs["indexer_mode"])
    assert modes == list(cfg.indexer_types)


def test_rejection_sampling_longer_chain_distribution_stress():
    rng = np.random.default_rng(11)
    q = np.tile(np.array([[0.55, 0.30, 0.15]]), (4, 1))
    p = np.tile(np.array([[0.20, 0.50, 0.30]]), (5, 1))
    counts = np.zeros(3, dtype=np.int64)
    for _ in range(3000):
        draft = np.asarray([rng.choice(3, p=row) for row in q], dtype=np.int64)
        res = rejection_verify_chain(draft, q, p, rng=rng)
        counts[res.new_tokens[0]] += 1
    np.testing.assert_allclose(counts / counts.sum(), p[0], atol=0.04)


def test_scaled_serving_pressure_stress_is_stable():
    out = run_scaled_glm52_serving_pressure(tokens=128, seed=12)
    assert 0.0 <= out["mean_accepted_length"] <= out["mtp_steps"]
    assert 0.0 <= out["cache_hit_ratio"] <= 1.0
    assert out["mean_tv"] >= 0.0
    assert out["mean_target_entropy"] > 0.0
