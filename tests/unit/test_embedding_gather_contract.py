"""AR-DECODE-PROGRAM-1 S1c: native sm_120 embedding gather lowering contract (no GPU).

Needs a production ``tessera-opt``; a host without one cannot evaluate this
lane and skips, naming why.
"""

from dataclasses import replace

import pytest

from tessera.compiler import scheduled_embedding_gather as seg
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

NAMES = ("embed", "token", "x")


@pytest.fixture(scope="module")
def artifact():
    if find_tessera_opt() is None:
        pytest.skip("no production tessera-opt on this host; embedding gather lowering cannot be evaluated")
    return seg.lower_scheduled_embedding_gather(NAMES, vocab=512, hidden=64, batch=3)


def test_replay_carries_the_semantic_contract(artifact):
    artifact.validate()
    assert (artifact.vocab, artifact.hidden, artifact.batch) == (512, 64, 3)
    assert artifact.entry == f"tessera_tile_embedding_gather_f32_{artifact.schedule_digest[:10]}"
    tile = artifact.tile_ir
    assert 'index_bounds = "kernel_checked_status_word"' in tile
    assert 'schedule = "cta_128_per_token_row"' in tile
    assert "func.func @embedding_gather" not in tile  # the Graph entry is consumed


def test_tampered_tile_ir_is_refused(artifact):
    bad = replace(artifact, tile_ir=artifact.tile_ir.replace(
        '"kernel_checked_status_word"', '"unchecked"', 1))
    with pytest.raises(ValueError, match="disagrees"):
        bad.validate()


def test_relabelled_artifact_is_refused(artifact):
    with pytest.raises(ValueError, match="disagree"):
        replace(artifact, vocab=511).validate()
    with pytest.raises(ValueError, match="disagree"):
        replace(artifact, batch=4).validate()
    with pytest.raises(ValueError, match="disagree"):
        replace(artifact, names=("token", "embed", "x")).validate()


def test_schedule_hash_changes_with_every_extent(artifact):
    digests = {artifact.schedule_digest}
    for extents in ({"vocab": 511, "hidden": 64, "batch": 3},
                    {"vocab": 512, "hidden": 63, "batch": 3},
                    {"vocab": 512, "hidden": 64, "batch": 2}):
        digests.add(seg.lower_scheduled_embedding_gather(NAMES, **extents).schedule_digest)
    assert len(digests) == 4


@pytest.mark.parametrize("names,extents,needle", [
    (("t", "t", "o"), (8, 4, 2), "unique identifiers"),
    (("t", "i", "status"), (8, 4, 2), "none 'status'"),
    (("t", "i"), (8, 4, 2), "unique identifiers"),
    (NAMES, (0, 4, 2), "positive integers"),
    (NAMES, (8, 4, 2**31), "launch ABI"),
])
def test_lowering_refuses_outside_its_envelope(names, extents, needle):
    vocab, hidden, batch = extents
    with pytest.raises(ValueError, match=needle):
        seg.lower_scheduled_embedding_gather(names, vocab=vocab, hidden=hidden, batch=batch)


def test_non_sm120_module_is_left_to_its_own_owner(artifact):
    graph = artifact.graph_ir.replace('"nvidia_sm120"', '"rocm_gfx1201"').replace('"sm_120"', '"gfx1201"')
    schedule = run_tessera_opt(find_tessera_opt(), graph, "--tessera-graph-to-schedule")
    assert "schedule.embedding_gather" not in schedule
