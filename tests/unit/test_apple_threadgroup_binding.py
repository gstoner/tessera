import pytest
from types import SimpleNamespace
from tessera.compiler.apple_threadgroup import TILED_SCORES, tiled_threadgroup_length


def test_tiled_msl_slot_matches_existing_native_encoder_index():
    assert TILED_SCORES.declaration() == 'threadgroup float* tg_scores [[threadgroup(0)]]'
    assert tiled_threadgroup_length(SimpleNamespace(reduction=None), 8192, 32768) == 32768


@pytest.mark.parametrize('columns,reduction', [(8192, 'softmax'), (8193, None), (1025, None), (True, None)])
def test_dynamic_storage_does_not_ignore_alignment_or_static_scratch(columns, reduction):
    with pytest.raises(ValueError):
        tiled_threadgroup_length(SimpleNamespace(reduction=reduction), columns, 32768)
