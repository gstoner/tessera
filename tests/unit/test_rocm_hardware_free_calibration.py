from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/rocm/calibrate_hardware_free_scores.py"
)
SPEC = importlib.util.spec_from_file_location("rocm_calibration", SCRIPT)
assert SPEC and SPEC.loader
CALIBRATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CALIBRATION)


def test_spearman_handles_ties_and_directions():
    assert CALIBRATION._spearman(
        [1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert CALIBRATION._spearman(
        [1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)
    assert CALIBRATION._spearman([1, 1, 1], [10, 20, 30]) is None


def test_f32_locality_follows_emitted_operand_load_order():
    score = CALIBRATION._f32_locality([32, 32, 17], [2, 4])
    assert score["a"]["sequential"] > 0
    assert score["b"]["sequential"] > score["a"]["sequential"]
    assert 0 <= score["score"] <= 1


def test_committed_gfx1151_corpus_triggers_terminal_no_retuning_verdict():
    result = CALIBRATION.analyze_committed(
        CALIBRATION.DEFAULT_RETUNE, CALIBRATION.DEFAULT_HOT_PATHS)
    assert result["quality"]["candidate_rows"] == 54
    assert result["quality"]["winner_reproduction_count"] == 0
    assert result["gate"]["positive_fraction"] == 0.0
    assert result["verdict"]["locality"] == "reject"
    assert result["verdict"]["terminal"] is True
