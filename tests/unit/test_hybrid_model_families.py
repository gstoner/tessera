"""Track L — named hybrid model families (Qwen3.6 / Nemotron-3 / LFM2.5).

Each `models/*` factory wires the `stdlib.hybrid` full-model-block stack into a
named config.  Full `config()` is a shape-level artifact (verified structurally,
not built — the weights are gigabytes); `scaled_config()` is a Mac-executable
shrink gated against the recompute reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.models import lfm2_5, nemotron3, qwen3_6
from tessera.stdlib import hybrid as hy

_MODELS = [qwen3_6, nemotron3, lfm2_5]


def test_full_configs_match_published_shapes():
    q = qwen3_6.config()
    assert q.schedule.num_layers == 40 and q.linear_mixer == "delta"
    assert q.schedule.counts() == {"linear": 30, "full": 10}   # [delta×3, attn]×10
    assert q.ffn == "moe" and q.num_experts == 256 and q.top_k == 8 and q.shared_expert

    n = nemotron3.config()
    assert n.schedule.num_layers == 88 and n.linear_mixer == "ssm"
    assert n.schedule.counts()["full"] == 11                   # sparse anchors
    assert n.ffn == "moe"

    lf = lfm2_5.config()
    assert lf.schedule.num_layers == 24 and lf.linear_mixer == "liv"
    assert lf.schedule.counts() == {"linear": 18, "full": 6}   # 18 LIV : 6 GQA
    assert lf.conv_kernel == 3 and lf.num_experts == 32 and lf.top_k == 4


@pytest.mark.parametrize("mod", _MODELS, ids=lambda m: m.__name__.split(".")[-1])
def test_scaled_config_decode_equals_recompute(mod):
    """Each scaled model runs end-to-end: streaming dual-cache decode ≡ full
    recompute (the full-block contract: mixer + MoE FFN, dual cache)."""
    cfg = mod.scaled_config()
    rng = np.random.default_rng(0)
    w = hy.synth_weights(cfg, rng)
    x = rng.standard_normal((2, 9, cfg.d_model))
    np.testing.assert_allclose(hy.hybrid_decode(x, w, cfg, prefill=2),
                               hy.hybrid_forward(x, w, cfg), rtol=1e-9, atol=1e-9)


def test_qwen36_scaled_with_mtp_is_lossless():
    """A named model + the MTP draft head → lossless self-speculation."""
    cfg = qwen3_6.scaled_config()
    lm = hy.synth_lm_weights(cfg, vocab_size=20, rng=np.random.default_rng(1))
    prompt = np.array([[3, 7, 1]])
    ar = hy.greedy_generate(lm, prompt, n=6)
    spec, accepted = hy.mtp_speculative_generate(lm, prompt, n=6)
    np.testing.assert_array_equal(spec, ar)
    assert accepted >= 0
