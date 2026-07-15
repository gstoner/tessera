"""Static coverage/schema pins for the gfx1151 compiler retune ratchets."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    path = ROOT / "benchmarks" / "rocm" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_f32_retune_covers_square_rectangular_and_ragged():
    bench = _load("benchmark_rocm_f32_retune")
    assert (256, 256, 256) in bench.SHAPES
    assert (128, 512, 256) in bench.SHAPES
    assert any(m % 16 or n % 16 or k % 16 for m, n, k in bench.SHAPES)
    assert {(2, 2), (4, 4), (6, 4)} <= set(bench.TILES)


def test_grouped_and_swiglu_retunes_cover_transition_and_model_rows():
    grouped = _load("benchmark_rocm_grouped_gemm_retune")
    names = {case[0] for case in grouped.CASES}
    assert {"balanced_small", "transition_64k", "transition_64k_high_k",
            "balanced_model", "ragged_model", "wide_ffn",
            "narrow_down"} <= names
    swiglu = _load("benchmark_rocm_swiglu_retune")
    assert len(swiglu.CASES) >= 2


def test_transport_and_attention_ratchets_pin_required_matrix():
    transport = _load("benchmark_rocm_transport_retune")
    assert {(16, 128), (128, 1024), (16, 4096)} <= set(transport.CASES)
    g6b = _load("benchmark_rocm_g6b_two_wave")
    assert (1, 16, 1024, 128, False) in g6b.CASES
    assert (1, 16, 1009, 128, True) in g6b.CASES
    g6c = _load("benchmark_rocm_g6c_split_reduced")
    assert any(h != g for _, h, g, _, _, _ in g6c.CASES)
    assert {causal for *_, causal in g6c.CASES} == {False, True}


def test_consolidated_retune_baseline_records_all_decisions():
    path = (ROOT / "benchmarks" / "baselines" /
            "rocm_gfx1151_compiler_retune_2026_07_15.json")
    data = json.loads(path.read_text())
    assert data["schema"] == "tessera.rocm.compiler_retune.v1"
    assert data["device"] == "gfx1151"
    assert {"f32_gemm", "grouped_gemm", "grouped_swiglu",
            "kv_moe_transport", "g6b", "g6c"} <= data.keys()
    assert "promote" in data["g6b"]["decision"]
    assert "reject" in data["g6c"]["decision"]
    assert data["g6b"]["resources"]["two_wave_d128"]["vgpr_spills"] == 0
