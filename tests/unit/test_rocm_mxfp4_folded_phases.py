"""Host-free checks for same-CTA folded prefill diagnostic slots."""
from __future__ import annotations

import numpy as np
import pytest

from benchmarks.rocm.measure_gfx1201_folded_phases import _phase_summary
from tessera.compiler.rocm_mxfp4_folded import emit_mxfp4_folded_prefill_hip


def test_phase_trace_is_compile_time_opt_in_and_non_selecting() -> None:
    source = emit_mxfp4_folded_prefill_hip()
    assert "#ifdef TESSERA_FOLDED_PHASE_TRACE" in source
    assert "Trace[slot + 2] = copy_ticks;" in source
    assert "Trace[slot + 3] = compute_ticks;" in source


def test_phase_slots_use_same_cta_deltas_only() -> None:
    slots = np.array([[100, 200, 40, 50], [300, 420, 60, 40]], dtype=np.uint64)
    result = _phase_summary(slots)
    assert result["cta_slots"] == 2
    assert result["copy_ticks_median"] == 50
    assert result["compute_ticks_median"] == 45
    assert result["clock_scope"] == "same_cta_delta_only_cross_cu_unvalidated"


@pytest.mark.parametrize(
    "slots",
    [
        [[0, 100, 20, 20]],
        [[100, 90, 20, 20]],
        [[100, 200, 0, 20]],
        [[100, 200, 70, 50]],
    ],
)
def test_phase_slots_refuse_invalid_measurements(slots: list[list[int]]) -> None:
    with pytest.raises(RuntimeError):
        _phase_summary(np.asarray(slots, dtype=np.uint64))
