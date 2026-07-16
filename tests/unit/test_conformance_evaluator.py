"""Phase E1c — the Evaluator corroborates conformance ``complete`` cells.

Two layers:
  * a **portable coverage gate** (runs in CI everywhere, no Metal): every
    executable complete cell must have an Evaluator program builder, so the
    corroboration set tracks the registry and cannot silently shrink;
  * a **Darwin corroboration** that independently re-derives each complete
    cell on real hardware and requires HARDWARE_VERIFIED — "derive validates
    declare", catching any cell the registry calls complete that the generic
    Evaluator cannot reproduce.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from tessera.compiler import conformance_evaluator as CE
from tessera.compiler.evaluator import Rung


# ── portable: coverage gate over the registry's complete cells ───────────────

def test_every_executable_complete_cell_has_an_evaluator_builder():
    """No executable complete cell may lack a corroboration builder — otherwise
    a 'complete' claim exists that the Evaluator can't reproduce."""
    uncovered = CE.uncovered_complete_cells()
    assert uncovered == [], (
        f"complete cells on an executable backend with no Evaluator program "
        f"builder (add one to conformance_evaluator._BASE_FN): {uncovered}"
    )


def test_there_are_corroboratable_complete_cells():
    cells = CE.corroboratable_complete_cells()
    assert cells, "expected at least one executable complete cell to corroborate"
    # The known executable complete surface on this tree.
    assert ("matmul", "apple_gpu") in cells
    assert ("flash_attn", "apple_gpu") in cells
    assert ("matmul_relu", "apple_gpu") in cells
    assert ("matmul", "apple_cpu") in cells


def test_apple_cpu_stateful_kv_builder_is_numerically_corroborated():
    verdict = CE.corroborate(
        "kv_cache_read", "apple_cpu", np.random.default_rng(20260712)
    )
    assert verdict.rung is Rung.HARDWARE_VERIFIED
    assert verdict.provenance_ok and verdict.correctness == "pass"


# ── Darwin: independently reproduce each complete cell on real hardware ───────

@pytest.mark.hardware_apple_gpu
def test_complete_cells_are_evaluator_corroborated_on_darwin():
    rng = np.random.default_rng(20260611)
    results = CE.corroborate_complete(rng)
    assert results, "no complete cells corroborated"
    for op, target, v in results:
        assert v.rung is Rung.HARDWARE_VERIFIED, (
            f"conformance marks {op!r}@{target!r} complete, but the Evaluator "
            f"could not independently reproduce it: rung={v.rung.name}, "
            f"kind={v.execution_kind!r}, status={v.runtime_status!r} — {v.detail}"
        )
        assert v.provenance_ok and v.correctness == "pass"
