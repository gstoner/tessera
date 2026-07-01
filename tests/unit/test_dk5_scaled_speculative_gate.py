import numpy as np

from tessera.cache import KVCacheHandle
from tessera.speculative import (
    dk5_scaled_speculative_decode_gate,
    dk5_select_lanes,
)
from tessera.stdlib import dspark


def _weights(vocab=9, hidden=6, seed=61):
    rng = np.random.default_rng(seed)
    return dspark.DSparkWeights(
        token_embedding=(rng.standard_normal((vocab, hidden)) * 0.2).astype(np.float32),
        hidden_proj=(rng.standard_normal((hidden, hidden)) * 0.08).astype(np.float32),
        token_proj=(rng.standard_normal((hidden, hidden)) * 0.08).astype(np.float32),
        out_proj=(rng.standard_normal((hidden, vocab)) * 0.15).astype(np.float32),
        confidence_proj=np.ones((hidden,), dtype=np.float32) * 0.01,
        markov=(rng.standard_normal((vocab, hidden)) * 0.05).astype(np.float32),
    )


def test_dk5_gate_composes_ds2_verify_accept_and_cache_rollback():
    cfg = dspark.DSparkConfig(
        num_anchors=1,
        block_size=3,
        vocab_size=9,
        confidence_threshold=0.0,
    )
    weights = _weights()
    rng = np.random.default_rng(62)
    target_hidden = (rng.standard_normal((1, 5, 6)) * 0.2).astype(np.float32)
    prev_tokens = np.array([2], dtype=np.int64)
    anchors = np.array([1], dtype=np.int64)

    cache = KVCacheHandle(num_heads=1, head_dim=2, max_seq=16)
    cache.append(np.zeros((4, 1, 2), np.float32), np.zeros((4, 1, 2), np.float32))
    pre_seq = cache.current_seq
    spec_k = np.ones((cfg.block_size, 1, 2), np.float32)
    spec_v = np.ones((cfg.block_size, 1, 2), np.float32)

    def target_verify(draft_tokens, proposal, draft_output):
        target = np.empty((1, cfg.block_size + 1), dtype=np.int64)
        target[0, :2] = draft_tokens[0, :2]
        target[0, 2] = (int(draft_tokens[0, 2]) + 1) % cfg.vocab_size
        target[0, 3] = 7
        return target

    result = dk5_scaled_speculative_decode_gate(
        target_hidden=target_hidden,
        prev_tokens=prev_tokens,
        anchors=anchors,
        dspark_weights=weights,
        dspark_config=cfg,
        target_verify=target_verify,
        model_config={
            "uses_mla": True,
            "sparse": "msa",
            "num_experts": 4,
            "weight_dtype": "int4",
        },
        cache_handles=(cache,),
        cache_pre_seq=pre_seq,
        speculative_cache_chunks=((spec_k, spec_v),),
    )

    assert result.accepted_lengths.tolist() == [2]
    assert result.emitted_tokens[0][:2] == tuple(result.proposal.tokens[0, :2])
    assert result.emitted_tokens[0][2] == result.target_tokens[0, 2]
    assert cache.current_seq == pre_seq + 2
    assert result.cache_final_seq == (pre_seq + 2,)
    np.testing.assert_allclose(cache.keys[pre_seq:pre_seq + 2], 1.0)
    np.testing.assert_allclose(cache.keys[pre_seq + 2:pre_seq + cfg.block_size], 0.0)
    assert result.draft_execution_kind in {"native_gpu", "reference_cpu"}
    assert result.lanes.as_dict() == {
        "draft": "rocm_dspark_draft_block_compiled",
        "accept": "spec_accept",
        "cache_effects": "cache_commit/cache_rollback",
        "mla": "rocm_exotic_attn_compiled",
        "sparse_attention": "rocm_sparse_attn_compiled",
        "moe": "rocm_moe_transport_compiled",
        "dequant_gemm": "rocm_dequant_gemm_compiled",
    }


def test_dk5_lane_selection_is_structural_for_model_config():
    class Config:
        uses_mla = True
        sparse = "dsa"
        num_experts = 0
        quantization = None

    lanes = dk5_select_lanes(Config()).as_dict()
    assert lanes["mla"] == "rocm_exotic_attn_compiled"
    assert lanes["sparse_attention"] == "rocm_sparse_attn_compiled"
    assert "moe" not in lanes
    assert "dequant_gemm" not in lanes

