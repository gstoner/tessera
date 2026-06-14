"""Task adapters for real long-memory benchmarks decomposed onto the memory
contract (LongMemEval / MemoryArena / LongBench-v2).

The instances are synthetic and oracle-checkable: accuracy is ~1.0 by
construction, so these tests validate that the *decomposition* onto
``memory_read``/``memory_write`` is faithful and runnable — and that real
dataset records can flow through the ``from_records`` file-format hooks.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from benchmarks.common import ExecutionKind, RuntimeStatus  # noqa: E402
from benchmarks.long_memory_core import (  # noqa: E402
    AMABenchAdapter,
    AdapterResult,
    LongBenchV2Adapter,
    LongMemEvalAdapter,
    MemoryArenaAdapter,
    adapter_report,
    adapter_rows,
    run_all_adapters,
)


# ── LongMemEval: five abilities ──────────────────────────────────────────────


def test_longmemeval_all_abilities_solve():
    results = LongMemEvalAdapter(seed=3).run()
    by = {r.ability: r for r in results}
    assert set(by) == set(LongMemEvalAdapter.ABILITIES)
    for ability, r in by.items():
        assert r.passed, f"{ability} acc={r.accuracy}"


def test_longmemeval_knowledge_update_needs_recency():
    # the knowledge-update ability must return the NEWEST value
    a = LongMemEvalAdapter(seed=11)
    assert a.knowledge_update().accuracy == 1.0


def test_longmemeval_temporal_uses_version_column():
    a = LongMemEvalAdapter(seed=12)
    assert a.temporal_reasoning().accuracy == 1.0


def test_longmemeval_abstention_distinguishes_absent():
    a = LongMemEvalAdapter(seed=13)
    assert a.abstention().accuracy == 1.0


def test_longmemeval_from_records_normalizes_schema():
    recs = [{"question_id": "q7", "question_type": "abstention",
             "answer": "n/a", "haystack_sessions": [["a"], ["b"]]}]
    out = LongMemEvalAdapter.from_records(recs)
    assert out[0] == {"id": "q7", "ability": "abstention",
                      "answer": "n/a", "sessions": [["a"], ["b"]]}


# ── MemoryArena: interdependent action loop ──────────────────────────────────


def test_memory_arena_action_loop_beats_guessing():
    res = MemoryArenaAdapter(n_episodes=20, n_actions=4, seed=5).run()[0]
    assert res.accuracy == 1.0                       # memory-guided is exact
    assert res.accuracy > 1.0 / 4                     # vs. the guessing baseline


# ── LongBench-v2: multi-doc MCQ ──────────────────────────────────────────────


def test_longbench_v2_multi_doc_mcq_retrieves_answer():
    res = LongBenchV2Adapter(n_docs=24, n_choices=4, n_questions=20, seed=9).run()[0]
    assert res.accuracy == 1.0


def test_longbench_v2_from_records_normalizes_choices():
    recs = [{"_id": "x", "question": "?", "choice_A": "a", "choice_B": "b",
             "choice_C": "c", "choice_D": "d", "answer": "C"}]
    out = LongBenchV2Adapter.from_records(recs)
    assert out[0]["choices"] == ["a", "b", "c", "d"]
    assert out[0]["answer"] == "C"


# ── AMA-Bench: long-horizon trajectory memory ────────────────────────────────


def test_ama_bench_recalls_across_a_long_horizon():
    # a fact observed anywhere in a long trajectory must drive the action now
    res = {r.ability: r for r in AMABenchAdapter(horizon=512, seed=4).run()}
    assert res["long_horizon_recall"].accuracy == 1.0
    assert res["multi_domain"].accuracy == 1.0
    assert res["abstain_on_unobserved"].accuracy == 1.0


def test_ama_bench_recall_holds_as_horizon_grows():
    # arbitrary-horizon: residency requirement — accuracy is horizon-invariant
    for h in (64, 256, 1024):
        r = AMABenchAdapter(horizon=h, seed=h).long_horizon_recall()
        assert r.accuracy == 1.0, f"horizon {h} dropped recall"


def test_ama_bench_from_records_normalizes_trajectory_schema():
    recs = [{"trajectory_id": "t1", "domain": "swe",
             "trajectory": [{"tool": "grep"}], "question": "?", "answer": "x"}]
    out = AMABenchAdapter.from_records(recs)
    assert out[0]["id"] == "t1" and out[0]["domain"] == "swe"
    assert out[0]["steps"] == [{"tool": "grep"}]


# ── aggregate ────────────────────────────────────────────────────────────────


def test_run_all_adapters_report():
    results = run_all_adapters(seed=1)
    assert all(isinstance(r, AdapterResult) for r in results)
    report = adapter_report(results)
    assert report["all_pass"] is True
    assert set(report["benchmarks"]) == {"longmemeval", "memory_arena",
                                          "longbench_v2", "ama_bench"}
    assert report["abilities_graded"] == len(results)


def test_adapter_rows_use_shared_schema():
    rows = adapter_rows(run_all_adapters(seed=2))
    assert rows and all(r.execution_kind is ExecutionKind.REFERENCE for r in rows)
    assert all(r.runtime_status is RuntimeStatus.EXECUTABLE for r in rows)
    assert all("accuracy" in r.metrics for r in rows)


def test_adapters_are_deterministic():
    a = adapter_report(run_all_adapters(seed=42))
    b = adapter_report(run_all_adapters(seed=42))
    assert a == b
