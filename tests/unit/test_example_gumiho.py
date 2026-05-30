"""Regression coverage for the examples/advanced/gumiho port.

Locks the Gumiho hybrid-speculative-decoding demo: the draft + Full Tree
Attention verification math runs on the Apple compiler backend and matches a
float64 numpy reference, across all three target paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
