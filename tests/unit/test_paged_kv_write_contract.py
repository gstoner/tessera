"""AR-DECODE-PROGRAM-1 S1a: native paged-write lowering contract (no GPU).

Needs a production ``tessera-opt``; a host without one cannot evaluate this
lane and skips, naming why.
"""

from dataclasses import replace

import pytest

from tessera.compiler import scheduled_paged_kv as spk
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

NAMES = ("pages", "page_table", "values", "position")
DIMS = (4, 4, 16, 2, 16, 1)


@pytest.fixture(scope="module")
def artifact():
    if find_tessera_opt() is None:
        pytest.skip("no production tessera-opt on this host; paged-write lowering cannot be evaluated")
    return spk.lower_scheduled_paged_kv_write(NAMES, DIMS)


def test_replay_carries_the_in_place_contract(artifact):
    artifact.validate()
    assert artifact.entry == spk.PAGED_KV_WRITE_ENTRY
    tile = artifact.tile_ir
    assert 'page_ownership = "in_place"' in tile
    assert 'position_bounds = "kernel_checked_device_position"' in tile
    assert 'refusal = "i32_status_word"' in tile
    assert "func.func @paged_write" not in tile  # the Graph entry is consumed


def test_tampered_tile_ir_is_refused(artifact):
    bad = replace(artifact, tile_ir=artifact.tile_ir.replace('"in_place"', '"read_only_borrow"', 1))
    with pytest.raises(ValueError, match="disagrees"):
        bad.validate()


def test_relabelled_artifact_is_refused(artifact):
    with pytest.raises(ValueError, match="disagree"):
        replace(artifact, dims=(4, 4, 16, 2, 16, 2)).validate()
    with pytest.raises(ValueError, match="disagree"):
        replace(artifact, names=("pages", "page_table", "position", "values")).validate()


def test_schedule_hash_changes_with_the_shape(artifact):
    other = spk.lower_scheduled_paged_kv_write(NAMES, (4, 4, 16, 2, 16, 3))
    assert other.schedule_digest != artifact.schedule_digest


@pytest.mark.parametrize("names,dims,needle", [
    (NAMES, (4, 4, 16, 2, 16), "six positive"),
    (NAMES, (4, 4, 16, 2, 16, 0), "six positive"),
    (NAMES, (4, 2, 16, 2, 16, 33), "exceeds the logical page capacity"),
    (("pages", "pages", "values", "position"), DIMS, "unique identifiers"),
])
def test_lowering_refuses_outside_its_envelope(names, dims, needle):
    with pytest.raises(ValueError, match=needle):
        spk.lower_scheduled_paged_kv_write(names, dims)


def test_rocm_module_is_refused_at_schedule(artifact):
    graph = artifact.graph_ir.replace('"nvidia_sm120"', '"rocm_gfx1201"').replace('"sm_120"', '"gfx1201"')
    with pytest.raises(Exception, match="only sm_120 has one"):
        run_tessera_opt(find_tessera_opt(), graph, "--tessera-graph-to-schedule")
