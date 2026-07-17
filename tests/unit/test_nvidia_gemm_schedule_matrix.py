from benchmarks.nvidia.record_gemm_schedule_matrix import near_winner_consensus


def test_schedule_matrix_requires_cross_run_near_winner_consensus() -> None:
    assert near_winner_consensus(
        {"direct": 1.0, "shared": 1.02},
        {"direct": 1.01, "shared": 1.0}, .03) == ["direct", "shared"]
    assert near_winner_consensus(
        {"direct": 1.0, "shared": 1.2},
        {"direct": 1.2, "shared": 1.0}, .03) == []
