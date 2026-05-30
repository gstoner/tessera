"""Regression coverage for the examples/advanced/gumiho port.

Locks the Gumiho hybrid-speculative-decoding demo: the draft + Full Tree
Attention verification math runs on the Apple compiler backend and matches a
float64 numpy reference, across all three target paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_EX = Path(__file__).resolve().parents[2] / "examples" / "advanced" / "gumiho"


@pytest.fixture(scope="module")
def gumiho_mod():
    if not _EX.exists():
        pytest.skip("gumiho example not present")
    if str(_EX) not in sys.path:
        sys.path.insert(0, str(_EX))
    import gumiho  # noqa: E402
    return gumiho


@pytest.mark.parametrize("target", ["numpy", "apple_cpu", "apple_gpu"])
def test_gumiho_backend_matches_reference(gumiho_mod, target):
    s = gumiho_mod.run_gumiho_demo(gumiho_mod.tiny_config(), seed=0, target=target)
    assert s.validated
    assert s.backend_matches_reference
    assert s.max_logprob_abs_err <= 1e-3


def test_gumiho_hybrid_structure(gumiho_mod):
    s = gumiho_mod.run_gumiho_demo(gumiho_mod.tiny_config(), seed=0, target="numpy")
    # 2 serial + 5 parallel = 7 draft tokens; FTA keeps top-8 paths.
    assert s.serial_tokens == 2
    assert s.parallel_heads == 5
    assert s.total_draft_tokens == 7
    assert s.num_paths == 8
    # The 8 length-7 paths share prefixes, so the trie has fewer than 8*7 nodes.
    assert s.num_tree_nodes < 8 * 7


def test_gumiho_advances_kv_by_accepted_length(gumiho_mod):
    s = gumiho_mod.run_gumiho_demo(gumiho_mod.tiny_config(), seed=0, target="numpy")
    assert 0 <= s.accepted_length <= s.total_draft_tokens
    assert s.kv_advanced_to == s.kv_pre_seq + s.accepted_length


def test_gumiho_deterministic_across_seeds(gumiho_mod):
    a = gumiho_mod.run_gumiho_demo(gumiho_mod.tiny_config(), seed=3, target="numpy")
    b = gumiho_mod.run_gumiho_demo(gumiho_mod.tiny_config(), seed=3, target="numpy")
    assert a.accepted_prefix == b.accepted_prefix


# ── distillation + multi-step decode (the speculative-decoding win) ──────────
def test_distillation_lifts_accepted_length(gumiho_mod):
    before, after = gumiho_mod.run_training_demo(
        gumiho_mod.tiny_config(), seed=0, target="numpy",
        num_prompts=6, horizon=20, max_new_tokens=16, train_steps=400)
    # Untrained accepts ~<1 token/step; distillation should clear it by a wide
    # margin and beat vanilla (1.0 token / target pass) decisively.
    assert after.mean_accepted_length > before.mean_accepted_length + 1.0
    assert after.speedup_vs_vanilla > 2.0
    assert after.trained and not before.trained


def test_gpu_mask_add_matches_numpy(gumiho_mod):
    """The FTA tree-attention mask add runs on the GPU `add` op (MPSGraph binary
    op 0), bit-exact against numpy — so the mask is no longer host glue."""
    from gumiho.backend import make_backend

    be = make_backend("apple_gpu")
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((2, 5, 5)).astype(np.float32)
    mask = np.triu(np.full((5, 5), -1e30, np.float32), 1)
    got = be.add(scores, np.broadcast_to(mask, (2, 5, 5)))
    np.testing.assert_array_equal(got, scores + mask[None, :, :])
    assert be.name in ("metal", "numpy")


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_half_precision_draft(gumiho_mod, dtype):
    s = gumiho_mod.run_precision_demo(gumiho_mod.tiny_config(), seed=0, dtype=dtype)
    # Half precision keeps fp32 accumulation -> serial argmax tokens match f32
    # and the target logits stay close; off Metal it falls back to f32 (err 0).
    assert s.serial_tokens_match
    assert s.draft_paths > 0
    tol = 0.05 if dtype == "f16" else 0.2
    assert s.max_logit_abs_err < tol


def test_prefix_sharing_matches_and_saves(gumiho_mod):
    from gumiho.model import make_weights

    cfg = gumiho_mod.tiny_config()
    weights = make_weights(cfg, seed=0)
    rng = np.random.default_rng(0)
    prompts = rng.integers(0, cfg.vocab, size=(3, cfg.context_len), dtype=np.int64)
    s = gumiho_mod.run_prefix_sharing_demo(cfg, weights, prompts=prompts,
                                           max_new_tokens=16, seed=0)
    # Prefix-shared verify reproduces the naive target log-probs...
    assert s.verify_matches_recompute
    assert s.max_target_logprob_err < 1e-4
    # ...while computing strictly fewer K/V rows across the decode.
    assert s.shared_kv_rows < s.naive_kv_rows
    assert s.kv_rows_saved > 0


def test_prefix_shared_single_verify_equals_build_draft(gumiho_mod):
    import numpy as _np
    from gumiho.backend import NumpyBackend
    from gumiho.draft import build_draft
    from gumiho.model import ParallelHeads, SerialHead, TargetModel, make_weights
    from gumiho.prefix_shared import PrefixSharedVerifier

    cfg = gumiho_mod.tiny_config()
    w = make_weights(cfg, seed=1)
    be = NumpyBackend(eps=cfg.rmsnorm_eps)
    ctx = _np.random.default_rng(1).integers(0, cfg.vocab, size=cfg.context_len,
                                             dtype=_np.int64)
    tgt = TargetModel(w, cfg)
    lh, _ = tgt.forward(be, ctx)
    bundle = build_draft(be, cfg, tgt, SerialHead(w, cfg), ParallelHeads(w, cfg),
                         context_tokens=ctx, last_hidden=lh[-1])
    res = PrefixSharedVerifier(w, cfg).verify_tree(ctx, bundle)
    assert _np.max(_np.abs(res.target_log_probs - bundle.target_log_probs)) < 1e-5
    assert res.kv_rows_per_verify < res.kv_rows_full_recompute


def test_onchip_step_accept_matches_host(gumiho_mod):
    """Compose the on-device speculative step: draft + verify (MPSGraph) feed the
    Rung-3 MSL accept kernel. The on-device accept must match the host reference,
    and a distilled draft must accept more than the untrained one."""
    cfg = gumiho_mod.tiny_config()
    untrained = gumiho_mod.run_onchip_step_demo(cfg, seed=0, target="numpy")
    assert untrained.matches_host
    assert 0 <= untrained.accepted_length <= cfg.total_draft_tokens
    assert untrained.num_paths == cfg.fta_top_paths

    trained = gumiho_mod.run_onchip_step_demo(cfg, seed=0, target="numpy",
                                              distill_steps=250)
    assert trained.matches_host
    assert trained.accepted_length >= untrained.accepted_length


def test_serial_draft_forloop_matches_host(gumiho_mod):
    """Phase-G Rung 1: the serial draft lowered into one MPSGraph control-flow
    executable reproduces the host SerialHead token-for-token, in one dispatch."""
    from gumiho.model import make_weights

    cfg = gumiho_mod.tiny_config()
    weights = make_weights(cfg, seed=0)
    r = gumiho_mod.validate_serial_forloop(cfg, weights, seed=0)
    assert r.matches_host
    assert r.backend in ("metal", "numpy")
    if r.backend == "metal":
        assert r.dispatches == 1                 # the whole loop is one graph
        assert r.host_dispatch_equiv > 1
        assert r.max_hidden_err < 1e-3


def test_resident_serial_draft_matches_host(gumiho_mod):
    from gumiho.model import make_weights

    cfg = gumiho_mod.tiny_config()
    weights = make_weights(cfg, seed=0)
    r = gumiho_mod.validate_resident_draft(cfg, weights, seed=0)
    # The resident command-buffer draft must reproduce the host SerialHead
    # token-for-token, and collapse the per-op dispatches into 1 buffer/token.
    assert r.matches_host
    assert r.backend in ("metal", "numpy")
    if r.backend == "metal":
        assert r.command_buffers == cfg.serial_tokens
        assert r.host_dispatch_equiv > r.command_buffers
        assert r.max_logit_abs_err < 1e-3


def test_multistep_decode_accounting(gumiho_mod):
    import numpy as np
    from gumiho.model import make_weights

    cfg = gumiho_mod.tiny_config()
    weights = make_weights(cfg, seed=0)
    prompts = np.zeros((2, cfg.context_len), dtype=np.int64)
    m = gumiho_mod.run_multistep_decode(
        cfg, weights, prompts=prompts, max_new_tokens=12, target="numpy")
    # tokens/step == mean_accepted + 1 bonus, and we generated >= the request.
    assert abs(m.tokens_per_step - (m.mean_accepted_length + 1.0)) < 1e-9
    assert m.tokens_generated >= 12
    assert m.speedup_vs_vanilla == m.tokens_per_step
