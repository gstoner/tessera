"""
test_benchmark_gemm.py — GEMMBenchmark tests (Phase 6)
"""
from __future__ import annotations

import json
import os
import tempfile
import pytest
from benchmarks.benchmark_gemm import GEMMBenchmark, GEMMConfig, GEMMResult


# ---------------------------------------------------------------------------
# GEMMConfig
# ---------------------------------------------------------------------------

class TestGEMMConfig:
    def test_flops_2mnk(self):
        cfg = GEMMConfig(M=2, N=2, K=2)
        assert cfg.flops() == 2 * 2 * 2 * 2  # 16

    def test_flops_large(self):
        cfg = GEMMConfig(M=4096, N=4096, K=4096)
        assert cfg.flops() == 2 * 4096 ** 3

    def test_bytes_bf16(self):
        cfg = GEMMConfig(M=1024, N=1024, K=1024, dtype="bf16")
        expected = 2 * (1024 * 1024 + 1024 * 1024 + 1024 * 1024)
        assert cfg.bytes_accessed() == expected

    def test_bytes_fp32_double(self):
        cfg_bf16 = GEMMConfig(M=128, N=128, K=128, dtype="bf16")
        cfg_fp32 = GEMMConfig(M=128, N=128, K=128, dtype="fp32")
        assert cfg_fp32.bytes_accessed() == 2 * cfg_bf16.bytes_accessed()

    def test_invalid_m_raises(self):
        with pytest.raises(ValueError):
            GEMMConfig(M=0, N=128, K=128)

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            GEMMConfig(M=128, N=-1, K=128)

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            GEMMConfig(M=128, N=128, K=0)


# ---------------------------------------------------------------------------
# GEMMBenchmark construction
# ---------------------------------------------------------------------------

class TestGEMMBenchmarkInit:
    def test_default_dtype(self, gemm_bench):
        assert gemm_bench.dtype == "bf16"

    def test_custom_peak_tflops(self):
        b = GEMMBenchmark(peak_tflops=1000.0)
        assert b.peak_tflops == 1000.0

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError):
            GEMMBenchmark(dtype="float128")

    def test_all_valid_dtypes(self):
        for dtype in ("bf16", "fp16", "fp32", "fp8"):
            b = GEMMBenchmark(dtype=dtype)
            assert b.dtype == dtype


# ---------------------------------------------------------------------------
# run / run_single
# ---------------------------------------------------------------------------

class TestGEMMRun:
    def test_run_default_sizes(self, gemm_bench):
        results = gemm_bench.run()
        assert len(results) == len(GEMMBenchmark.DEFAULT_SIZES)

    def test_run_custom_sizes(self, gemm_bench):
        results = gemm_bench.run(sizes=[(512, 512, 512), (1024, 512, 256)])
        assert len(results) == 2

    def test_run_single(self, gemm_bench):
        r = gemm_bench.run_single(256, 256, 256)
        assert isinstance(r, GEMMResult)

    def test_result_tflops_positive(self, gemm_bench):
        r = gemm_bench.run_single(1024, 1024, 1024)
        assert r.tflops > 0

    def test_result_latency_positive(self, gemm_bench):
        r = gemm_bench.run_single(1024, 1024, 1024)
        assert r.latency_ms > 0

    def test_result_memory_bw_positive(self, gemm_bench):
        r = gemm_bench.run_single(1024, 1024, 1024)
        assert r.memory_bw_gbps > 0

    def test_roofline_bound_is_compute_or_memory(self, gemm_bench):
        r = gemm_bench.run_single(4096, 4096, 4096)
        assert r.roofline_bound in ("compute", "memory")

    def test_large_gemm_compute_bound(self, gemm_bench):
        # Large square GEMMs are typically compute-bound
        r = gemm_bench.run_single(8192, 8192, 8192)
        assert r.roofline_bound == "compute"

    def test_small_gemm_may_be_memory_bound(self):
        # Very small GEMM with low arithmetic intensity is memory-bound
        b = GEMMBenchmark(peak_tflops=312.0, peak_membw_gbps=2000.0)
        r = b.run_single(64, 64, 64)
        # Just ensure it runs and classifies
        assert r.roofline_bound in ("compute", "memory")

    def test_tflops_below_peak(self, gemm_bench):
        r = gemm_bench.run_single(4096, 4096, 4096)
        assert r.tflops <= gemm_bench.peak_tflops * 1.01  # allow 1% float slack

    def test_mfu_in_0_1(self, gemm_bench):
        r = gemm_bench.run_single(4096, 4096, 4096)
        mfu = gemm_bench.mfu(r)
        assert 0.0 < mfu <= 1.0

    def test_result_config_stored(self, gemm_bench):
        r = gemm_bench.run_single(512, 256, 128)
        assert r.config.M == 512
        assert r.config.N == 256
        assert r.config.K == 128


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class TestGEMMReport:
    def test_report_is_string(self, gemm_bench):
        results = gemm_bench.run(sizes=[(1024, 1024, 1024)])
        text = gemm_bench.report(results)
        assert isinstance(text, str)

    def test_report_contains_header(self, gemm_bench):
        results = gemm_bench.run(sizes=[(1024, 1024, 1024)])
        text = gemm_bench.report(results)
        assert "TFLOPs" in text or "tflops" in text.lower()

    def test_report_contains_size(self, gemm_bench):
        results = gemm_bench.run(sizes=[(2048, 2048, 2048)])
        text = gemm_bench.report(results)
        assert "2048" in text


# ---------------------------------------------------------------------------
# to_json
# ---------------------------------------------------------------------------

class TestGEMMJson:
    def test_to_json_writes_file(self, gemm_bench, tmp_path):
        results = gemm_bench.run(sizes=[(1024, 1024, 1024)])
        path = str(tmp_path / "gemm.json")
        gemm_bench.to_json(results, path)
        assert os.path.exists(path)

    def test_to_json_valid_json(self, gemm_bench, tmp_path):
        results = gemm_bench.run(sizes=[(512, 512, 512)])
        path = str(tmp_path / "gemm.json")
        gemm_bench.to_json(results, path)
        with open(path) as f:
            data = json.load(f)
        assert data["benchmark"] == "gemm"
        assert len(data["results"]) == 1

    def test_to_json_fields_present(self, gemm_bench, tmp_path):
        results = gemm_bench.run(sizes=[(256, 256, 256)])
        path = str(tmp_path / "gemm.json")
        gemm_bench.to_json(results, path)
        with open(path) as f:
            data = json.load(f)
        r = data["results"][0]
        for key in ("M", "N", "K", "latency_ms", "tflops", "roofline_bound"):
            assert key in r
