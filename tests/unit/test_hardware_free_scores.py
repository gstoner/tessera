from __future__ import annotations

import pytest

from tessera.compiler.hardware_free_scores import (
    GFX1151_BANK_GEOMETRY,
    bank_conflict_degrees,
    step_distance_histogram,
)


def test_gfx1151_constants_are_wave32_one_phase_and_32_bank_domain():
    geometry = GFX1151_BANK_GEOMETRY
    assert geometry.wave_size == 32
    assert geometry.bank_count == 32
    assert geometry.bank_width_bytes == 4
    assert geometry.phases == (tuple(range(32)),)


def test_gfx1151_conflict_degree_distinguishes_unit_stride_and_broadcast():
    assert bank_conflict_degrees([lane * 4 for lane in range(32)]) == (1,)
    assert bank_conflict_degrees([0] * 32) == (32,)
    assert bank_conflict_degrees([(lane % 8) * 4 for lane in range(32)]) == (4,)


def test_step_distance_histogram_uses_manhattan_survey_buckets():
    histogram = step_distance_histogram([(0, 0), (0, 1), (4, 5), (40, 5)])
    assert histogram.sequential == 1
    assert histogram.cache_line_jump == 1
    assert histogram.large_jump == 1
    assert histogram.locality_score == pytest.approx(0.5)


def test_step_distance_rejects_mixed_coordinate_rank():
    with pytest.raises(ValueError, match="equal rank"):
        step_distance_histogram([(0, 0), (1,)])
