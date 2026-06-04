"""Sprint 11 — honesty guards for the apple_gpu reasoning-attention benchmark.

The benchmark example (`benchmarks/apple_gpu/benchmark_reasoning_attention.py`)
reports route / target / executor / correctness / timing as SEPARATE fields and
promotes exactly one strict executable envelope. These tests lock the honesty
contract so the example can never drift into over-claiming:

  * Every row carries all five separate fields (no conflation).
  * The executable/non-executable split is grounded in runtime envelopes, not
    hard-coded — `mla_decode_fused` / `lightning_attention` / etc. are
    compiler-visible only (executor None, executable False), while the MLA-style
    block's `tessera.matmul` primitive and the native-sparse value lane are
    executable only when their runtime probes say so.
  * Compiler-visible-only rows NEVER carry a correctness number or a timing —
    we don't fabricate numbers for ops we didn't run.
  * When Metal is active (Darwin), the executable envelope actually ran:
    correctness is tight (fp32) and timing is positive.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_BENCH = (Path(__file__).resolve().parents[2] / "benchmarks" / "apple_gpu"
          / "benchmark_reasoning_attention.py")


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("bench_reasoning", _BENCH)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def report(bench):
    return bench.build_report([(8, 16, 2), (16, 32, 4)], reps=3)


_REQUIRED_FIELDS = {
    "name", "variant_kind", "shape", "route", "target", "executor",
    "executable", "correctness_max_rel_err", "timing_ms", "skip_reason",
}


def test_every_row_has_all_separate_fields(report):
    assert report["rows"], "report has no rows"
    for r in report["rows"]:
        assert _REQUIRED_FIELDS <= set(r), (
            f"row missing fields: {_REQUIRED_FIELDS - set(r)}")
        assert r["target"] == "apple_gpu"


def test_executable_split_is_grounded_in_driver_envelope(bench):
    """The executable flag must come from the driver runtime envelope, not a
    literal in the benchmark."""
    from tessera.compiler import driver as d
    # The MLA-style block is built from matmul → executable.
    assert bench._is_executable("tessera.matmul") is True
    assert "tessera.matmul" in d._APPLE_GPU_RUNTIME_OPS
    # Every compiler-visible-only reasoning op must be ABSENT from the runtime
    # envelope (honestly not executable).
    for v in bench.COMPILER_VISIBLE_ONLY:
        op = v["graph_op"]
        assert op not in d._APPLE_GPU_RUNTIME_OPS, (
            f"{op} unexpectedly in runtime envelope — benchmark would "
            f"over-claim it as executable")
        assert bench._is_executable(op) is False
        assert bench._executor_for(op) is None


def test_native_sparse_claim_is_grounded_in_value_runtime_probe(bench):
    from tessera import runtime

    symbol = "tessera_apple_gpu_native_sparse_attn_f32"
    assert symbol in runtime._APPLE_VALUE_GPU_SYMBOLS
    assert bench._native_sparse_value_available() == (
        runtime._apple_gpu_native_sparse_attn_f32() is not None)


def test_compiler_visible_rows_never_fabricate_numbers(report):
    cv = [r for r in report["rows"]
          if r["variant_kind"] == "compiler_visible_only"]
    assert cv, "expected compiler-visible-only rows"
    for r in cv:
        assert r["executable"] is False
        assert r["executor"] is None
        assert r["correctness_max_rel_err"] is None
        assert r["timing_ms"] is None
        assert r["skip_reason"]  # must say why it wasn't executed


def test_executable_envelope_rows_classification(report):
    ex = [r for r in report["rows"]
          if r["variant_kind"] == "executable_envelope"]
    assert ex, "expected executable-envelope rows"
    mla = [r for r in ex if r["name"] == "mla_style_attention"]
    assert mla, "expected MLA-style executable-envelope rows"
    for r in mla:
        assert r["executable"] is True
        assert r["executor"] == "apple_gpu_mps"

    nsa = [r for r in ex if r["name"] == "deepseek_native_sparse_attn"]
    assert nsa, "expected native-sparse executable-envelope rows"
    for r in nsa:
        if r["executable"]:
            assert r["executor"] == "apple_gpu_value_target_ir"
            assert r["skip_reason"] is None
        else:
            assert r["executor"] is None
            assert r["correctness_max_rel_err"] is None
            assert r["timing_ms"] is None
            assert r["skip_reason"]


@pytest.mark.skipif(sys.platform != "darwin",
                    reason="executable envelope only runs on Darwin/Metal")
def test_executable_envelope_actually_ran_when_metal_active(bench, report):
    if not report["metal_active"]:
        pytest.skip("metal runtime inactive on this host")
    ex = [r for r in report["rows"]
          if r["variant_kind"] == "executable_envelope"]
    ran = [r for r in ex if r["skip_reason"] is None]
    assert ran, "metal active but executable envelope did not run"
    for r in ran:
        # fp32 fused matmul→softmax→matmul + MPS projections: tight.
        assert r["correctness_max_rel_err"] is not None
        assert r["correctness_max_rel_err"] < 1e-4
        assert r["timing_ms"] is not None and r["timing_ms"] > 0.0
