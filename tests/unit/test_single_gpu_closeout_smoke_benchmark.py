from benchmarks import single_gpu_closeout_smoke as smoke


def test_closeout_smoke_rows_are_complete_and_runnable():
    expected = {
        "attn_compressed_blocks",
        "attn_local_window_2d",
        "attn_top_k_blocks",
        "linear_attn_state",
        "lookahead_sparse_attention",
        "msa_sparse_attention",
        "memory_index_score",
        "msa_index_scores",
        "varlen_sdpa",
        "score_combine",
        "dynamic_slice",
        "masked_categorical",
        "slice",
        "cast",
        "chunk",
        "rope_split",
        "split",
        "unpack",
        "dequant_matmul",
        "kv_cache_read",
        "complex_abs",
        "complex_arg",
        "complex_conjugate",
        "complex_div",
        "complex_exp",
        "complex_log",
        "complex_mul",
        "complex_pow",
        "complex_sqrt",
        "mobius",
        "stereographic",
    }
    assert set(smoke.CLOSEOUT_SMOKE_OPS) == expected

    rows = smoke.run_smoke(reps=1)
    assert {row["op"] for row in rows} == expected
    assert all(row["ok"] for row in rows)
    assert all(row["latency_ms"] >= 0.0 for row in rows)
