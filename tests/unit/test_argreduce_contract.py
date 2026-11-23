"""AR-DECODE-PROGRAM-1 S1b: native sm_120 argreduce lowering contract (no GPU).

Needs a production ``tessera-opt``; a host without one cannot evaluate this
lane and skips, naming why.
"""

from dataclasses import replace

import pytest

from tessera.compiler import scheduled_argreduce as sar
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

NAMES = ("logits", "token")


@pytest.fixture(scope="module")
def artifact():
    if find_tessera_opt() is None:
        pytest.skip("no production tessera-opt on this host; argreduce lowering cannot be evaluated")
    return sar.lower_scheduled_argreduce("argmax", NAMES, (1, 256))


def test_replay_carries_the_semantic_contract(artifact):
    artifact.validate()
    assert (artifact.rows, artifact.columns, artifact.result_shape) == (1, 256, (1,))
    assert artifact.entry == f"tessera_tile_argreduce_argmax_f32_{artifact.schedule_digest[:10]}"
    tile = artifact.tile_ir
    assert 'tie_break = "first"' in tile and 'nan_mode = "first_nan"' in tile
    assert 'schedule = "cta_128_warp_shuffle"' in tile
    assert "func.func @argreduce" not in tile  # the Graph entry is consumed


@pytest.mark.parametrize("kind,shape,axis,keepdims,rows,columns,result", [
    ("argmin", (4, 3, 7), -1, False, 12, 7, (4, 3)),
    ("argmax", (4, 3, 7), 2, True, 12, 7, (4, 3, 1)),
    ("argmax", (4, 3, 7), None, False, 1, 84, ()),
    ("argmin", (4, 3, 7), None, True, 1, 84, (1, 1, 1)),
])
def test_launch_extents_follow_numpy_axis_rules(kind, shape, axis, keepdims, rows, columns, result):
    if find_tessera_opt() is None:
        pytest.skip("no production tessera-opt on this host")
    a = sar.lower_scheduled_argreduce(kind, ("x", "i"), shape, axis=axis, keepdims=keepdims)
    assert (a.rows, a.columns, a.result_shape) == (rows, columns, result)
    assert a.axis_mode == ("flatten" if axis is None else "last")


def test_tampered_tile_ir_is_refused(artifact):
    bad = replace(artifact, tile_ir=artifact.tile_ir.replace('"first_nan"', '"ignore"', 1))
    with pytest.raises(ValueError, match="disagrees"):
        bad.validate()


def test_relabelled_artifact_is_refused(artifact):
    with pytest.raises(ValueError, match="disagree"):
        replace(artifact, columns=128).validate()
    with pytest.raises(ValueError, match="disagree"):
        replace(artifact, kind="argmin").validate()
    with pytest.raises(ValueError, match="disagree"):
        replace(artifact, names=("token", "logits")).validate()


def test_schedule_hash_changes_with_kind_and_shape(artifact):
    other_shape = sar.lower_scheduled_argreduce("argmax", NAMES, (1, 255))
    other_kind = sar.lower_scheduled_argreduce("argmin", NAMES, (1, 256))
    assert len({artifact.schedule_digest, other_shape.schedule_digest, other_kind.schedule_digest}) == 3


@pytest.mark.parametrize("kind,names,shape,axis,needle", [
    ("argsort", NAMES, (1, 8), -1, "kind must be one of"),
    ("argmax", ("x", "x"), (1, 8), -1, "unique identifiers"),
    ("argmax", NAMES, (), -1, "rank >= 1"),
    ("argmax", NAMES, (1, 0), -1, "positive integers"),
    ("argmax", NAMES, (4, 8), 0, "only the last axis"),
    ("argmax", NAMES, (65536, 32768), -1, "i32 index ABI"),
])
def test_lowering_refuses_outside_its_envelope(kind, names, shape, axis, needle):
    with pytest.raises(ValueError, match=needle):
        sar.lower_scheduled_argreduce(kind, names, shape, axis=axis)


def test_non_sm120_module_is_left_to_its_own_owner(artifact):
    graph = artifact.graph_ir.replace('"nvidia_sm120"', '"rocm_gfx1201"').replace('"sm_120"', '"gfx1201"')
    schedule = run_tessera_opt(find_tessera_opt(), graph, "--tessera-graph-to-schedule")
    assert "schedule.argreduce" not in schedule
