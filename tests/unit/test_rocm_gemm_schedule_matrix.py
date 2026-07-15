from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks" / "rocm" / "benchmark_rocm_gemm_schedule_matrix.py"
BASELINE = (ROOT / "benchmarks" / "baselines" /
            "rocm_gfx1151_gemm_schedule_matrix.json")


def _load():
    spec = importlib.util.spec_from_file_location("rocm_gemm_schedule_matrix", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load()


def test_full_matrix_covers_required_families_and_deduplicates():
    cases = bench.matrix_cases()
    families = {case.family for case in cases}
    assert families == {"square", "rectangular", "ragged", "dtype", "epilogue"}
    identities = {(c.family, c.m, c.n, c.k, c.dtype, c.bias, c.activation)
                  for c in cases}
    assert len(identities) == len(cases)
    assert any((c.m, c.n, c.k) == (2049, 4093, 2051) for c in cases)
    assert any((c.m, c.n, c.k) == (128, 4096, 4096) for c in cases)
    assert any((c.m, c.n, c.k) == (4096, 11008, 4096) for c in cases)
    assert {c.dtype for c in cases if c.family == "dtype"} == {
        "f16", "bf16", "int8", "int4"}
    assert {(c.bias, c.activation) for c in cases if c.family == "epilogue"} >= {
        (True, "none"), (False, "relu"), (True, "relu"),
        (True, "gelu"), (True, "silu")}


def test_quick_matrix_is_small_but_representative():
    cases = bench.matrix_cases(quick=True)
    assert len(cases) == 5
    assert {case.family for case in cases} == {
        "square", "rectangular", "ragged", "dtype", "epilogue"}


def test_integer_correctness_requires_exact_result():
    case = bench.Case("dtype", 2, 2, 2, "int8")
    assert bench._correct(case, {"mismatch_count": 0})
    assert not bench._correct(case, {"mismatch_count": 1})


def test_float_correctness_accepts_scaled_accumulation_drift():
    case = bench.Case("square", 2, 2, 4096)
    assert bench._correct(case, {"max_abs_error": 0.2,
                                 "normalized_error": 1e-3})
    assert not bench._correct(case, {"max_abs_error": 0.2,
                                     "normalized_error": 3e-3})


def test_large_integer_oracle_is_sampled_exact():
    case = bench.Case("dtype", 4096, 4096, 4096, "int8")
    a = bench.np.ones((2, 3), dtype=bench.np.int8)
    b = bench.np.ones((3, 2), dtype=bench.np.int8)
    actual = bench.np.full((2, 2), 3, dtype=bench.np.int32)
    assert bench._reference(case, a, b, None) is None
    metrics = bench._sampled_integer_error(actual, a, b)
    assert metrics["mismatch_count"] == 0
    assert metrics["sample_count"] == 4


def test_winner_stability_uses_paired_interleaved_trials():
    case = bench.Case("square", 8, 8, 8)
    common = {"case": case.key, "correct": True,
              "tflops_or_tops": 1.0, "relative_mad": 0.2,
              "resources": {}}
    rows = [
        {**common, "tile": [2, 4], "median_ms": 1.0,
         "trials_ms": [1.0, 1.2, 0.9, 1.1]},
        {**common, "tile": [4, 4], "median_ms": 1.2,
         "trials_ms": [1.2, 1.4, 1.1, 1.3]},
    ]
    winner = bench.summarize_winners(rows, [case])[0]
    assert winner["tile"] == [2, 4]
    assert winner["paired_win_rate"] == 1.0
    assert winner["paired_median_speedup"] > 1.03
    assert winner["stable"] is True


def test_committed_schedule_ratchet_is_correct_and_resource_backed():
    doc = json.loads(BASELINE.read_text())
    assert doc["schema"] == "tessera.rocm.gemm_schedule_ratchet.v1"
    assert doc["evidence_arch"] == "gfx1151"
    assert doc["method"]["row_count"] == 185
    assert doc["method"]["all_rows_correct"] is True
    resources = {tuple(row["tile"]): row
                 for row in doc["code_object_resources_f16_plain"]}
    assert resources[(1, 1)]["spill_count"] == 0
    assert resources[(2, 2)]["spill_count"] == 0
    assert resources[(2, 4)]["spill_count"] > 0
    assert resources[(4, 4)]["spill_count"] > resources[(3, 4)]["spill_count"]
    assert all(row["paired_speedup_min"] >= 1.03
               for row in doc["promoted_rows"])
    assert len(doc["retained_non_promotions"]) == 3


def test_resource_parser_is_honest_when_tools_cannot_decode(monkeypatch):
    monkeypatch.setattr(bench, "_tool", lambda _name: None)
    row = bench.code_object_resources(b"not an object")
    assert row["vgpr_count"] is None
    assert row["lds_bytes"] is None
    assert row["spills"] is False
    assert row["spill_count"] is None


def test_resource_parser_reads_amdgpu_metadata(monkeypatch):
    metadata = """
    .group_segment_fixed_size: 0
    .max_flat_workgroup_size: 256
    .private_segment_fixed_size: 736
    .sgpr_count: 107
    .sgpr_spill_count: 107
    .vgpr_count: 256
    .vgpr_spill_count: 285
    .wavefront_size: 32
    """

    class Result:
        stdout = metadata
        stderr = ""

    monkeypatch.setattr(bench, "_tool", lambda name: name)
    monkeypatch.setattr(bench.subprocess, "run", lambda *args, **kwargs: Result())
    row = bench.code_object_resources(b"object")
    assert row["vgpr_count"] == 256
    assert row["sgpr_count"] == 107
    assert row["lds_bytes"] == 0
    assert row["scratch_bytes"] == 736
    assert row["spill_count"] == 392
    assert row["spills"] is True
    assert row["vgpr_limited_waves_per_simd"] == 6
