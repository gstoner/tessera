from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).parents[2] / "benchmarks/nvidia/finalize_test5_corpus.py"
SPEC = importlib.util.spec_from_file_location("finalize_test5", PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(mod)


def _row(winner: str):
    return {"device": "nvidia:sm_120", "target": "nvidia", "op": "matmul",
            "bucket": [127, 259, 63], "dtype": "float16", "timing": "device",
            "winner": winner, "latency_ms": 1.0, "candidates": {winner: 1.0}}


def test_two_matching_runs_become_selector_eligible_with_resources():
    payload = {"version": 3, "records": [_row("direct")]}
    out = mod.merge({"version": 3, "records": []}, payload, payload,
                    {"routes": {"direct": ["sha256:r"]}})
    evidence = out["records"][0]["evidence"]
    assert evidence["stable_winner"] is True
    assert evidence["selector_eligible"] is True


def test_disagreeing_runs_remain_ineligible():
    out = mod.merge({"version": 3, "records": []},
                    {"records": [_row("direct")]},
                    {"records": [_row("shared")]},
                    {"routes": {"shared": ["sha256:r"]}})
    assert out["records"][0]["evidence"]["selector_eligible"] is False


def test_run_winner_jitter_can_converge_inside_declared_noise_band():
    left = _row("direct"); left["candidates"] = {"direct": 1.0, "shipped": 1.001}
    right = _row("shipped"); right["candidates"] = {"direct": 1.08, "shipped": 1.0}
    out = mod.merge({"version": 3, "records": []}, {"records": [left]},
                    {"records": [right]},
                    {"routes": {"shipped": ["sha256:r"]}},
                    noise_fraction=.03)
    row = out["records"][0]
    assert row["winner"] == "shipped"
    assert row["evidence"]["stable_winner"] is True
    assert row["evidence"]["near_winner_consensus"] == ["shipped"]


def test_unstable_fresh_evidence_never_restamps_a_stale_prior_row():
    """Decision #11: an unstable fresh pair used to copy its evidence -- now
    carrying today's toolchain_digest -- onto the prior committed row, which
    would serve a measurement from another (or no) toolchain as current."""
    from tessera.compiler.emit.autotune import CORPUS_VERSION

    stale_prior = _row("old_winner")            # pre-v4: no toolchain identity
    left = _row("direct"); left["evidence"] = {"toolchain_digest": "sha256:now"}
    right = _row("shared"); right["evidence"] = {"toolchain_digest": "sha256:now"}
    out = mod.merge({"version": 3, "records": [stale_prior]},
                    {"records": [left]}, {"records": [right]},
                    {"routes": {"shared": ["sha256:r"]}})
    row, = out["records"]
    assert out["version"] == CORPUS_VERSION
    assert row["winner"] == "shared"             # the fresh row, not the stale one
    assert row["evidence"]["toolchain_digest"] == "sha256:now"
    assert row["evidence"]["selector_eligible"] is False


def test_unstable_fresh_evidence_keeps_a_same_toolchain_prior_row():
    prior = _row("prior"); prior["evidence"] = {"toolchain_digest": "sha256:now"}
    left = _row("direct"); left["evidence"] = {"toolchain_digest": "sha256:now"}
    right = _row("shared"); right["evidence"] = {"toolchain_digest": "sha256:now"}
    out = mod.merge({"version": 4, "records": [prior]},
                    {"records": [left]}, {"records": [right]},
                    {"routes": {"shared": ["sha256:r"]}})
    row, = out["records"]
    assert row["winner"] == "prior"
    assert row["evidence"]["selector_eligible"] is False


def test_unstable_fresh_evidence_never_transplants_identities_onto_a_prior_row():
    """AUTOTUNE-EMITTED-IDENTITY-2026-09-27: a prior row measured under the
    same toolchain but stamped with other (or no) artifact identities must not
    receive the fresh run's identities -- that would backfill a code identity
    onto a measurement of code nobody verified. The fresh row replaces it."""
    ids = {"direct": {"source_sha256": "new"}, "shared": {"source_sha256": "new"}}
    prior = _row("prior")
    prior["evidence"] = {"toolchain_digest": "sha256:now",
                         "delegate_identities": {"shipped": {"abi_digest": "x"}}}
    left = _row("direct")
    left["evidence"] = {"toolchain_digest": "sha256:now", "delegate_identities": ids}
    right = _row("shared")
    right["evidence"] = {"toolchain_digest": "sha256:now", "delegate_identities": ids}
    out = mod.merge({"version": 4, "records": [prior]},
                    {"records": [left]}, {"records": [right]},
                    {"routes": {"shared": ["sha256:r"]}})
    row, = out["records"]
    assert row["winner"] == "shared"
    assert row["evidence"]["delegate_identities"] == ids
    assert row["evidence"]["selector_eligible"] is False



@pytest.mark.parametrize("field,first,second", [
    ("delegate_identities",
     {"direct": {"source_sha256": "run1"}}, {"direct": {"source_sha256": "run2"}}),
    ("delegate_identities", {"direct": {"source_sha256": "run1"}}, None),
    ("toolchain_digest", "sha256:before", "sha256:after"),
])
def test_runs_that_timed_different_code_are_refused(field, first, second):
    """AUTOTUNE-EMITTED-IDENTITY-2026-09-27 review: the merged row takes its
    stamps from the second run and its winner from both. Two runs that timed
    different code (a rebuild or emitter change between them) would publish
    run 2's identities over a consensus that partly measured run 1's code."""
    left, right = _row("direct"), _row("direct")
    left["evidence"] = {field: first}
    right["evidence"] = {field: second} if second is not None else {}
    with pytest.raises(ValueError, match=field):
        mod.merge({"version": 4, "records": []}, {"records": [left]},
                  {"records": [right]}, {"routes": {"direct": ["sha256:r"]}})
