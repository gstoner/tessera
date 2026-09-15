import pytest
from types import SimpleNamespace
from tessera.compiler.apple_threadgroup import TILED_SCORES, tiled_threadgroup_length


def test_tiled_msl_slot_matches_existing_native_encoder_index():
    assert TILED_SCORES.declaration() == 'threadgroup float* tg_scores [[threadgroup(0)]]'
    assert tiled_threadgroup_length(SimpleNamespace(reduction=None), 8192, 32768) == 32768


@pytest.mark.parametrize('columns,reduction', [(8192, 'softmax'), (8193, None), (True, None)])
def test_dynamic_storage_does_not_ignore_static_scratch_or_the_device_limit(columns, reduction):
    with pytest.raises(ValueError):
        tiled_threadgroup_length(SimpleNamespace(reduction=reduction), columns, 32768)


def test_ragged_extent_rounds_up_to_metal_alignment_instead_of_refusing():
    # 1025 floats = 4100 B; Metal wants a 16 B multiple. The encoders round up
    # (2026-09-15), so the contract reports the rounded length rather than
    # rejecting every ragged width -- which had silently sent ragged wide
    # reductions to the reference path since 2026-09-05.
    assert tiled_threadgroup_length(SimpleNamespace(reduction=None), 1025, 32768) == 4112
    assert tiled_threadgroup_length(SimpleNamespace(reduction="softmax"), 2049, 32768) == 8208  # static scratch is limit-checked, not returned
