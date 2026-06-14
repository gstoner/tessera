"""Direct contract tests for the long-memory primitives + the long_memory_core
benchmark (RULER / LongMemEval / MemoryArena lens).

Two layers:

* ``tessera.memory`` contract — append↔read roundtrip, eviction, deterministic
  top-k, and the new ``abstain_below=`` abstention contract.  These promote the
  memory primitives from ``structural_only`` toward ``needs_direct_test`` closure.
* ``long_memory_core`` proof rows — the reference scenarios pass against their
  oracles and the gap rows are honestly ``missing_backend``.
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

from tessera.memory import MemoryTable, memory_evict, memory_read, memory_write  # noqa: E402

from benchmarks.common import ExecutionKind, RuntimeStatus  # noqa: E402
from benchmarks.long_memory_core import (  # noqa: E402
    LANDED_MEMORY_PRIMITIVES,
    MEMORY_PRIMITIVE_GAPS,
    PARTIAL_MEMORY_PRIMITIVES,
    LongMemoryConfig,
    abstention_scenario,
    build_report,
    resident_decode_telemetry,
    ruler_multihop_scenario,
    ruler_needle_scenario,
    run_core,
    telemetry,
    version_update_scenario,
)
from benchmarks.long_memory_core import core as LM  # noqa: E402


# ── tessera.memory contract ──────────────────────────────────────────────────


def _unit_bank(seed: int, n: int = 8, kd: int = 4, vd: int = 3):
    rng = np.random.default_rng(seed)
    keys = rng.standard_normal((n, kd)).astype(np.float32)
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    vals = rng.standard_normal((n, vd)).astype(np.float32)
    return MemoryTable(keys=keys, values=vals), keys, vals


def test_append_then_read_roundtrip():
    bank, keys, vals = _unit_bank(0)
    res = memory_read(bank, keys[2], top_k=1)
    assert np.allclose(res.values, vals[2], atol=1e-5)
    # a freshly appended fact is recoverable by its own key
    new_k = (keys[0] + keys[1])
    new_k = new_k / np.linalg.norm(new_k)
    new_v = np.array([9.0, 9.0, 9.0], np.float32)
    grown = memory_write(bank, new_k, new_v)
    assert grown.size == bank.size + 1
    assert np.allclose(memory_read(grown, new_k, top_k=1).values, new_v, atol=1e-5)


@pytest.mark.parametrize("kind", ["keep_last", "indices", "max_entries"])
def test_evict_removes_entries(kind):
    bank, _keys, _vals = _unit_bank(1, n=8)
    if kind == "keep_last":
        out = memory_evict(bank, keep_last=5)
        assert out.size == 5
    elif kind == "max_entries":
        out = memory_evict(bank, max_entries=3)
        assert out.size == 3
    else:
        out = memory_evict(bank, indices=[0, 1])
        assert out.size == 6


def test_evict_requires_exactly_one_selector():
    bank, _k, _v = _unit_bank(2)
    with pytest.raises(ValueError):
        memory_evict(bank, keep_last=3, max_entries=4)


def test_topk_is_deterministic():
    bank, keys, _vals = _unit_bank(3)
    a = memory_read(bank, keys[5], top_k=3).indices
    b = memory_read(bank, keys[5], top_k=3).indices
    assert np.array_equal(a, b)


def test_abstention_defaults_off_and_field_present():
    bank, keys, vals = _unit_bank(4)
    res = memory_read(bank, keys[1], top_k=1)            # no abstain_below
    assert res.abstained is False                        # default, no claim
    assert np.allclose(res.values, vals[1], atol=1e-5)


def test_abstention_present_hits_absent_abstains():
    # orthonormal bank so similarity is unambiguous
    d = 6
    keys = np.eye(d, dtype=np.float32)
    vals = np.arange(d, dtype=np.float32)[:, None].repeat(2, axis=1)
    bank = MemoryTable(keys=keys, values=vals)

    present = keys[2]                                    # cosine 1.0 with entry 2
    hit = memory_read(bank, present, top_k=1, abstain_below=0.5)
    assert hit.abstained is False
    assert np.allclose(hit.values, vals[2], atol=1e-5)

    absent = np.full((d,), 1.0 / np.sqrt(d), np.float32)  # cosine 1/√d ≈ 0.41 < 0.5
    miss = memory_read(bank, absent, top_k=1, abstain_below=0.5)
    assert miss.abstained is True
    assert np.all(np.isnan(miss.values))                # stale hit can't masquerade


def test_prefer_recent_breaks_exact_key_tie_toward_newest():
    # same key written twice; similarity ties, recency must pick the newer value
    key = np.array([[1.0, 0.0, 0.0, 0.0]], np.float32)
    v1 = np.array([[1.0, 1.0]], np.float32)
    v2 = np.array([[2.0, 2.0]], np.float32)
    bank = MemoryTable(keys=np.concatenate([key, key]),
                       values=np.concatenate([v1, v2]))
    recent = memory_read(bank, key[0], top_k=1, prefer_recent=True)
    assert np.allclose(recent.values, v2[0], atol=1e-5)        # newest wins


def test_recency_key_uses_named_metadata_column():
    key = np.array([[1.0, 0.0, 0.0, 0.0]], np.float32)
    v_old = np.array([[1.0, 1.0]], np.float32)
    v_new = np.array([[2.0, 2.0]], np.float32)
    # insertion order says row 1 is newest, but the version column says row 0 is
    bank = MemoryTable(keys=np.concatenate([key, key]),
                       values=np.concatenate([v_old, v_new]),
                       metadata={"version": np.array([9, 3])})
    got = memory_read(bank, key[0], top_k=1, recency_key="version")
    assert np.allclose(got.values, v_old[0], atol=1e-5)        # version 9 wins


def test_recency_does_not_reorder_distinct_scores():
    bank, keys, _vals = _unit_bank(9, n=6)
    top = memory_read(bank, keys[4], top_k=3, prefer_recent=True).indices
    assert top[0] == 4                                          # score still dominates


def test_recency_key_must_exist():
    bank, keys, _vals = _unit_bank(10)
    with pytest.raises(ValueError):
        memory_read(bank, keys[0], recency_key="missing")


def test_abstention_batched_is_per_query():
    d = 6
    eye = np.eye(d, dtype=np.float32)
    bank = MemoryTable(keys=eye,
                       values=np.arange(d, dtype=np.float32)[:, None].repeat(2, axis=1))
    q = np.stack([eye[0],                                    # present
                  np.full((d,), 1.0 / np.sqrt(d), np.float32)])  # absent
    res = memory_read(bank, q, top_k=1, abstain_below=0.5)
    assert res.abstained.tolist() == [False, True]
    assert not np.any(np.isnan(res.values[0]))
    assert np.all(np.isnan(res.values[1]))


# ── long_memory_core scenarios ───────────────────────────────────────────────


def test_ruler_needle_recall_is_exact():
    cfg = LongMemoryConfig(bank_size=64, seed=11)
    bank, queries, oracle = ruler_needle_scenario(cfg)
    got = memory_read(bank, queries, top_k=1).values
    assert np.allclose(got, oracle, atol=1e-4)


def test_ruler_multihop_chain_recovers_payload():
    cfg = LongMemoryConfig(bank_size=64, seed=12)
    bank, q0, payload = ruler_multihop_scenario(cfg)
    hop1 = memory_read(bank, q0, top_k=1)
    hop2 = memory_read(bank, np.asarray(hop1.values), top_k=1)
    assert np.allclose(hop2.values, payload, atol=1e-4)


def test_abstention_scenario_is_deterministic():
    cfg = LongMemoryConfig(key_dim=16)
    bank, present, absent = abstention_scenario(cfg)
    hit = memory_read(bank, present, top_k=1, abstain_below=cfg.abstain_floor)
    miss = memory_read(bank, absent, top_k=1, abstain_below=cfg.abstain_floor)
    assert hit.abstained is False and miss.abstained is True


def test_version_update_writes_are_ordered():
    cfg = LongMemoryConfig(seed=13)
    bank, _key, latest = version_update_scenario(cfg)
    assert bank.size == 2                                # both versions retained
    # the newest write is the last row (recency lives in order, not similarity)
    assert np.allclose(bank.values[-1], latest, atol=1e-5)


def test_resident_decode_is_metamorphically_equivalent_and_cheaper():
    cfg = LongMemoryConfig(key_dim=16, value_dim=4, decode_steps=16, seed=15)
    match, tele = resident_decode_telemetry(cfg)
    assert match is True                                 # cached ≡ uncached (oracle)
    # resident appends T entries; recompute rebuilds Σt = T(T+1)/2
    assert tele["resident_append_entries"] == 16
    assert tele["recompute_build_entries"] == 16 * 17 // 2
    assert tele["build_reduction_x"] > 1.0               # residency processes less


# ── long_memory_core proof rows ──────────────────────────────────────────────


def test_run_core_proof_levels():
    rows = run_core(LongMemoryConfig(bank_size=64, seed=14))
    ref = [r for r in rows if r.execution_kind is ExecutionKind.REFERENCE]
    gap = [r for r in rows if r.runtime_status is RuntimeStatus.MISSING_BACKEND]
    assert len(ref) == 5 and all(r.correctness.passed for r in ref)
    assert len(gap) == 1 and all(r.correctness.passed is None for r in gap)


def test_gap_rows_name_declared_gaps():
    rows = run_core(LongMemoryConfig(bank_size=64))
    for r in rows:
        if r.runtime_status is RuntimeStatus.MISSING_BACKEND:
            assert r.reason.startswith("gap: ")
            assert r.reason.removeprefix("gap: ") in MEMORY_PRIMITIVE_GAPS


def test_gap_row_rejects_undeclared_gap():
    with pytest.raises(ValueError):
        LM._gap_row("bogus", "shape", gap="not_a_real_gap")


def test_build_report_summary():
    rows = run_core(LongMemoryConfig(bank_size=64))
    report = build_report(rows)
    assert report["reference_passed"] == 5
    assert report["missing_backend"] == 1
    assert set(report["open_gaps"]).issubset(set(MEMORY_PRIMITIVE_GAPS))
    assert "abstention_read_threshold" in report["landed_primitives"]
    assert "metadata_time_version_filter" in report["landed_primitives"]


@pytest.mark.parametrize("name", ["abstention_read_threshold",
                                  "metadata_time_version_filter",
                                  "memory_index_score_gpu"])
def test_landed_contracts_are_not_also_gaps(name):
    # a contract we shipped must not also be advertised as an open gap
    assert name in LANDED_MEMORY_PRIMITIVES
    assert name not in MEMORY_PRIMITIVE_GAPS


def test_segmented_topk_is_partial_kernel_landed_frontend_pending():
    # the Metal TopK kernel is landed+verified but not yet @jit-reachable —
    # it must be in the PARTIAL bucket, not GAPS and not (fully) LANDED.
    assert "segmented_topk_gpu" in PARTIAL_MEMORY_PRIMITIVES
    assert "segmented_topk_gpu" not in MEMORY_PRIMITIVE_GAPS
    assert "segmented_topk_gpu" not in LANDED_MEMORY_PRIMITIVES


def test_buckets_are_disjoint():
    gaps = set(MEMORY_PRIMITIVE_GAPS)
    landed = set(LANDED_MEMORY_PRIMITIVES)
    partial = set(PARTIAL_MEMORY_PRIMITIVES)
    assert gaps.isdisjoint(landed)
    assert gaps.isdisjoint(partial)
    assert landed.isdisjoint(partial)


def test_telemetry_one_event_per_row():
    rows = run_core(LongMemoryConfig(bank_size=64))
    events = telemetry(rows)
    assert len(events) == len(rows)
    assert all(isinstance(e, dict) for e in events)
