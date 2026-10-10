"""Host-free invariants for the gfx1201 matmul evidence recorder."""

import pytest

from benchmarks.rocm.record_gfx1201_matmul_shape_key import _expected_cache_states


@pytest.mark.parametrize(
    ("num_shapes", "expected"),
    [
        (1, ["cold"]),
        (3, ["cold", "warm_cache", "warm_cache"]),
        (6, ["cold", "warm_cache", "warm_cache", "warm_cache", "warm_cache", "warm_cache"]),
    ],
)
def test_expected_cache_states_scale_with_shape_matrix(num_shapes, expected):
    assert _expected_cache_states(num_shapes) == expected


def test_expected_cache_states_reject_empty_matrix():
    with pytest.raises(ValueError, match="at least one shape"):
        _expected_cache_states(0)
