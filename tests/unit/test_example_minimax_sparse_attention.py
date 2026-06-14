"""Regression coverage for examples/attention/minimax_sparse_attention.py (MSA Phase 5).

Locks the MSA example: the dense-equivalence anchor (MSA with
``top_k == num_blocks`` reproduces dense GQA), the selected-block stats, and the
theoretical attention-compute reduction sweep. CPU reference only.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_EX = Path(__file__).resolve().parents[2] / "examples" / "attention"


@pytest.fixture(scope="module")
def msa_example():
    if not _EX.exists():
        pytest.skip("MSA example not present")
    if str(_EX) not in sys.path:
        sys.path.insert(0, str(_EX))
    import minimax_sparse_attention as mod  # noqa: E402
    return mod


def test_dense_equivalence_anchor(msa_example):
    c = msa_example.compare(seq_len=64, block_size=8, top_k=2, causal=True)
    # MSA with top_k == num_blocks == dense GQA, bit-for-bit (fp64).
    assert c.dense_equiv_max_abs_err < 1e-9
    # Forced-local guarantees every row selects its own block.
    assert c.local_block_hit == 1.0
    assert 0.0 < c.coverage <= 1.0
    assert c.num_blocks == 8 and c.top_k == 2


def test_flop_reduction_is_real_and_bounded(msa_example):
    f = msa_example.attention_flops(
        seq_len=8192, head_dim=128, block_size=128, top_k=7, Hq=64, Hkv=8
    )
    assert f["num_blocks"] == 64
    assert f["selected_tokens"] == 7 * 128
    # Sparse attends fewer tokens → strictly less compute, but the index branch
    # keeps the reduction below the naive num_blocks/top_k ceiling.
    assert f["msa_flops_per_token"] < f["dense_flops_per_token"]
    assert 1.0 < f["reduction_factor"] <= 64 / 7


def test_long_context_table_monotone_and_labeled_compute(msa_example):
    rows = msa_example.long_context_table(seq_lens=(8192, 131072, 1048576))
    assert [r["seq_len"] for r in rows] == [8192, 131072, 1048576]
    for r in rows:
        assert r["reduction_factor"] > 1.0
        assert r["top_k"] == max(1, math.ceil(0.1 * r["num_blocks"]))  # default sparsity 0.1


def test_main_runs_clean(msa_example, capsys):
    msa_example.main()
    out = capsys.readouterr().out
    assert "MiniMax Sparse Attention" in out
    assert "dense-equivalence error" in out
    # Honesty: the example must NOT claim a wall-clock speedup.
    assert "NOT wall-clock speedup" in out
