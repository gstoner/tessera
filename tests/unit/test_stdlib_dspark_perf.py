import time

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.stdlib import dspark


def _weights(vocab=64, hidden=32, seed=0):
    rng = np.random.default_rng(seed)
    return dspark.DSparkWeights(
        token_embedding=rng.standard_normal((vocab, hidden)).astype(np.float32) * 0.05,
        hidden_proj=rng.standard_normal((hidden, hidden)).astype(np.float32) * 0.03,
        token_proj=rng.standard_normal((hidden, hidden)).astype(np.float32) * 0.03,
        out_proj=rng.standard_normal((hidden, vocab)).astype(np.float32) * 0.04,
        confidence_proj=rng.standard_normal(hidden).astype(np.float32) * 0.02,
        markov=rng.standard_normal((vocab, hidden)).astype(np.float32) * 0.01,
    )


def _median_ms(fn, reps=7):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


@pytest.mark.performance
def test_ds1_reference_draft_block_has_cpu_perf_budget():
    cfg = dspark.DSparkConfig(num_anchors=4, block_size=8, vocab_size=64)
    rng = np.random.default_rng(21)
    target_hidden = rng.standard_normal((4, 64, 32)).astype(np.float32)
    prev_tokens = rng.integers(0, cfg.vocab_size, size=(4,), dtype=np.int64)
    anchors = np.array([0, 12, 28, 48], dtype=np.int64)
    weights = _weights()

    ms = _median_ms(
        lambda: dspark.draft_block_forward(target_hidden, prev_tokens, anchors, weights, cfg),
        reps=9,
    )
    # Broad CI-safe ratchet: catches accidental quadratic Python work over vocab
    # or sequence while leaving room for slower shared runners.
    assert ms < 80.0


@pytest.mark.performance
def test_ds2_runtime_launch_overhead_is_bounded_against_ds1_oracle():
    cfg = dspark.DSparkConfig(num_anchors=2, block_size=4, vocab_size=64)
    rng = np.random.default_rng(22)
    target_hidden = rng.standard_normal((2, 32, 32)).astype(np.float32)
    prev_tokens = rng.integers(0, cfg.vocab_size, size=(2,), dtype=np.int64)
    anchors = np.array([0, 16], dtype=np.int64)
    weights = _weights(seed=1)
    art = rt.RuntimeArtifact(metadata={
        "target": "rocm",
        "compiler_path": "rocm_dspark_draft_block_compiled",
        "executable": True,
        "arg_names": ["target_hidden", "prev_tokens", "anchors", "weights"],
        "dspark_config": {
            "num_anchors": cfg.num_anchors,
            "block_size": cfg.block_size,
            "vocab_size": cfg.vocab_size,
        },
        "ops": [{"op_name": "tessera.dspark.draft_block"}],
    })

    direct_ms = _median_ms(
        lambda: dspark.draft_block_forward(target_hidden, prev_tokens, anchors, weights, cfg),
        reps=9,
    )
    launch_ms = _median_ms(
        lambda: rt.launch(art, (target_hidden, prev_tokens, anchors, weights)),
        reps=9,
    )
    # Hardware-free runs fall back to the DS1 oracle; native ROCm runs still
    # need to keep launch overhead bounded for these small draft-block shapes.
    assert launch_ms < max(75.0, direct_ms * 4.0)
