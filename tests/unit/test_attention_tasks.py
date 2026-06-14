"""Phase E3 — structure-keyed attention task family (EVALUATOR_PLAN.md §9)."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from tessera.compiler import attention_tasks as A
from tessera.compiler.compiler_grader import grade, task_names


# ── portable ─────────────────────────────────────────────────────────────────

def test_matrix_is_structure_keyed_by_mask_and_seqlen():
    names = A.attention_task_names()
    assert len(names) == 6                        # 3 seqlens × {full, causal}
    assert "attention/causal/s256" in names
    # importing the module registered them into the grader
    assert "attention/full/s64" in task_names()


def test_causal_reference_differs_from_full():
    rng = np.random.default_rng(0)
    q = rng.standard_normal((1, 1, 8, 4)).astype(np.float32)
    k = rng.standard_normal((1, 1, 8, 4)).astype(np.float32)
    v = rng.standard_normal((1, 1, 8, 4)).astype(np.float32)
    full = A._attn_reference(q, k, v, causal=False)
    causal = A._attn_reference(q, k, v, causal=True)
    assert not np.allclose(full, causal)          # the mask actually matters
    # row 0 (only attends to itself) under causal == just v[...,0,:]
    assert np.allclose(causal[0, 0, 0], v[0, 0, 0], atol=1e-5)


# ── Darwin: grade every (mask × seqlen) cell ─────────────────────────────────

@pytest.mark.skipif(sys.platform != "darwin", reason="flash_attn executes on Metal.")
def test_every_attention_cell_grades_pass():
    rng = np.random.default_rng(20260612)
    for name in A.attention_task_names():
        g = grade(name, rng)
        assert g.passed, f"{name}: {[c.detail for c in g.failures]}"


# ── structured masks: sliding-window + block-sparse (RULER structure axis) ───

def test_structured_matrix_is_keyed_by_mask_seqlen_and_param():
    names = A.structured_attention_task_names()
    assert len(names) == 4                        # {sliding, block} × 2 (seqlen,param)
    assert "attention/sliding/s128/p32" in names
    assert "attention/block/s256/p64" in names
    # importing the module registered them into the grader
    assert "attention/sliding/s128/p32" in task_names()


@pytest.mark.parametrize("mask,param", [("sliding", 4), ("block", 4)])
def test_structured_mask_changes_the_reference(mask, param):
    rng = np.random.default_rng(7)
    q = rng.standard_normal((1, 1, 16, 4)).astype(np.float32)
    k = rng.standard_normal((1, 1, 16, 4)).astype(np.float32)
    v = rng.standard_normal((1, 1, 16, 4)).astype(np.float32)
    full = A._attn_reference(q, k, v, causal=False)
    bias = A.structured_mask_bias(mask, 16, param)
    masked = A._attn_reference_biased(q, k, v, bias)
    assert not np.allclose(full, masked)          # the structured mask matters
    # block 0 (queries 0..param-1) only attend within their own block
    if mask == "block":
        # query 0 attends only keys [0, param) → independent of keys >= param
        v2 = v.copy()
        v2[0, 0, param:] += 5.0
        masked2 = A._attn_reference_biased(q, k, v2, bias)
        assert np.allclose(masked[0, 0, 0], masked2[0, 0, 0], atol=1e-5)


@pytest.mark.skipif(sys.platform != "darwin", reason="flash_attn executes on Metal.")
def test_every_structured_cell_grades_pass():
    rng = np.random.default_rng(20260614)
    for name in A.structured_attention_task_names():
        g = grade(name, rng)
        assert g.passed, f"{name}: {[c.detail for c in g.failures]}"
