"""Ordinary dispatch finds the gated rows (AUTOTUNE-GATED-INFER-DIMS).

Sync key ``SM120-AUTOTUNE-FOLLOWUPS-2026-09-27``. The NVIDIA recorder keys its
``gated_matmul`` rows on explicit ``(M, H, K)`` dims, but
``autotune._infer_dims`` had no gated rule, so ``corpus_winner`` asked the way
``run_arbitrated`` asks (no dims) returned ``None`` for every one of them.
These tests pin the rule and the production lookup that depends on it:

* the inferred dims are exactly the recorder's ``(M, H, K)`` for
  ``A (M,K), Wg (K,H), Wu (K,H)``, and malformed operand sets stay anonymous;
* for every committed ``gated_matmul`` row, operands built from its recorded
  workload shape land in the row's own bucket -- so the rule matches how the
  rows were written, not just a spelling of it;
* a stamped gated row is served by ``corpus_winner`` without explicit dims.

Host-independent (no GPU, no compiler).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tessera.compiler import fusion_core as F
from tessera.compiler.emit import autotune as AT
from tessera.compiler.emit import candidate as C
from tessera.compiler.emit.candidate import OP_GATED_MATMUL, Candidate, Tier

_CORPUS = Path(__file__).resolve().parents[2] / "benchmarks" / "baselines" / "autotune_corpus.json"


def _gated(m: int, h: int, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return (rng.standard_normal((m, k)).astype(np.float32),
            rng.standard_normal((k, h)).astype(np.float32),
            rng.standard_normal((k, h)).astype(np.float32))


def test_gated_dims_are_m_h_k():
    assert AT._infer_dims(OP_GATED_MATMUL, _gated(64, 256, 128)) == (64, 256, 128)
    # H and K differ, so an (M, K, H) mix-up cannot pass.
    assert AT._infer_dims(OP_GATED_MATMUL, _gated(7, 3, 5)) == (7, 3, 5)


@pytest.mark.parametrize("operands", [
    # contraction mismatch: A is (M, K) but Wg is (K', H)
    (np.zeros((4, 8)), np.zeros((9, 16)), np.zeros((9, 16))),
    # the up projection is not the gate projection's shape
    (np.zeros((4, 8)), np.zeros((8, 16)), np.zeros((8, 15))),
    # not a matrix
    (np.zeros((4, 8, 2)), np.zeros((8, 16)), np.zeros((8, 16))),
    # too few operands
    (np.zeros((4, 8)), np.zeros((8, 16))),
])
def test_malformed_gated_operands_stay_shape_anonymous(operands):
    assert AT._infer_dims(OP_GATED_MATMUL, operands) is None


def test_every_committed_gated_row_is_reachable_from_its_operands():
    rows = [r for r in json.loads(_CORPUS.read_text())["records"]
            if r["op"] == OP_GATED_MATMUL]
    assert rows, "the committed corpus holds gated_matmul rows"
    for row in rows:
        m, h, k = row["evidence"]["workload_shape"]
        dims = AT._infer_dims(OP_GATED_MATMUL, _gated(m, h, k))
        assert dims == (m, h, k)
        assert list(AT.bucket_key(dims, AT.SpecPolicy.BUCKET)) == row["bucket"], row


_TARGET = "gated_infer_dims_private"


class _StubGated(Candidate):
    """One live gated candidate under a private target, so the live field is
    exactly this lane on every host."""

    name = "stub_gated"
    tier = Tier.SYNTHESIZED
    target = _TARGET
    op = OP_GATED_MATMUL

    def artifact_identity(self, region, *inputs):
        return {"kind": "stub", "digest": "sha256:" + "0" * 64}

    def run(self, region, A, Wg, Wu, *a, **k):
        return region.reference(A, Wg, Wu), "stub"


@pytest.fixture
def stub_gated():
    cand = _StubGated()
    C.register_candidate(cand)
    yield cand
    C.unregister_candidate(cand)


@pytest.mark.parametrize("timing", [AT.TIMING_END_TO_END, AT.TIMING_DEVICE])
def test_production_lookup_serves_a_gated_row_without_dims(stub_gated, timing):
    region = F.GatedMatmulRegion(gate_act="silu", storage_dtype="f32")
    ins = _gated(128, 512, 512)
    key = ("dev:gated", _TARGET, OP_GATED_MATMUL,
           AT.bucket_key((128, 512, 512), AT.SpecPolicy.BUCKET), "f32", timing)
    cache = AT.MeasureCache()
    stamped = AT._delegate_identities({stub_gated.name: stub_gated}, region, ins)
    cache.put(key, AT.MeasureRecord(
        winner=stub_gated.name, latency_ms=1.0,
        candidates={stub_gated.name: 1.0}, unmeasured={},
        evidence={"delegate_identities": stamped}), fresh=True)

    served = AT.corpus_winner(region, OP_GATED_MATMUL, _TARGET, *ins,
                              dtype="f32", cache=cache, device="dev:gated",
                              timing=timing)
    assert served == stub_gated.name
    # A workload in another bucket does not borrow the row.
    other = AT.corpus_winner(region, OP_GATED_MATMUL, _TARGET, *_gated(4, 8, 8),
                             dtype="f32", cache=cache, device="dev:gated",
                             timing=timing)
    assert other is None
