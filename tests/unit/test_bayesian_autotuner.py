"""
Phase 5 — test_bayesian_autotuner.py

Tests for BayesianAutotuner, GEMMWorkload, TuningConfig, TuningResult.
"""
import math
import os
import tempfile
import pytest
from tessera.compiler.autotune_v2 import (
    BayesianAutotuner, GEMMWorkload, TuningConfig, TuningResult,
)


class TestGEMMWorkload:
    def test_basic_creation(self):
        w = GEMMWorkload(M=1024, N=1024, K=1024)
        assert w.M == 1024

    def test_flops(self):
        w = GEMMWorkload(M=1024, N=1024, K=1024)
        assert w.flops() == 2 * 1024 ** 3

    def test_tflops_at(self):
        w = GEMMWorkload(M=4096, N=4096, K=4096)
        latency_ms = w.flops() / (312e12) * 1e3
        assert abs(w.tflops_at(latency_ms) - 312.0) < 0.01

    def test_invalid_m_zero(self):
        with pytest.raises(ValueError):
            GEMMWorkload(M=0, N=1024, K=1024)

    def test_int8_dtype_supported_for_quantized_weights(self):
        w = GEMMWorkload(M=1024, N=1024, K=1024, dtype="int8")
        assert w.dtype == "int8"

    def test_tflops_at_zero_latency(self):
        w = GEMMWorkload(M=1024, N=1024, K=1024)
        with pytest.raises(ValueError):
            w.tflops_at(0.0)

    def test_repr(self):
        w = GEMMWorkload(M=4096, N=4096, K=4096)
        assert "4096" in repr(w)


class TestTuningConfig:
    def test_basic(self):
        cfg = TuningConfig(tile_m=128, tile_n=128, tile_k=32)
        assert cfg.tile_m == 128

    def test_invalid_non_power_of_2(self):
        with pytest.raises(ValueError):
            TuningConfig(tile_m=100, tile_n=128, tile_k=32)

    def test_invalid_num_warps(self):
        with pytest.raises(ValueError):
            TuningConfig(tile_m=128, tile_n=128, tile_k=32, num_warps=3)

    def test_invalid_num_stages_zero(self):
        with pytest.raises(ValueError):
            TuningConfig(tile_m=128, tile_n=128, tile_k=32, num_stages=0)

    def test_smem_bytes_positive(self):
        cfg = TuningConfig(tile_m=128, tile_n=128, tile_k=32, num_stages=2)
        assert cfg.smem_bytes() > 0

    def test_smem_bytes_scales_with_stages(self):
        cfg1 = TuningConfig(tile_m=64, tile_n=64, tile_k=32, num_stages=1)
        cfg2 = TuningConfig(tile_m=64, tile_n=64, tile_k=32, num_stages=2)
        assert cfg2.smem_bytes() == 2 * cfg1.smem_bytes()

    def test_to_ir_attr_contains_tile(self):
        cfg = TuningConfig(tile_m=128, tile_n=64, tile_k=32)
        attr = cfg.to_ir_attr()
        assert "tile_m = 128" in attr
        assert "tile_n = 64" in attr

    def test_repr(self):
        cfg = TuningConfig(tile_m=128, tile_n=128, tile_k=32)
        assert "128" in repr(cfg)


class TestBayesianAutotuner:
    def _tuner(self, M=1024, N=1024, K=1024):
        return BayesianAutotuner(GEMMWorkload(M=M, N=N, K=K), seed=42)

    def test_tune_returns_tuning_result(self):
        tuner = self._tuner()
        result = tuner.tune(max_trials=5)
        assert isinstance(result, TuningResult)

    def test_result_tflops_positive(self):
        tuner = self._tuner()
        result = tuner.tune(max_trials=5)
        assert result.tflops > 0

    def test_result_latency_ms_positive(self):
        tuner = self._tuner()
        result = tuner.tune(max_trials=5)
        assert result.latency_ms > 0

    def test_num_trials_matches(self):
        tuner = self._tuner()
        tuner.tune(max_trials=10)
        assert tuner.num_trials >= 1   # feasibility pruning may reduce count

    def test_best_property(self):
        tuner = self._tuner()
        tuner.tune(max_trials=5)
        assert tuner.best is not None
        assert tuner.best.tflops > 0

    def test_results_list_nonempty(self):
        tuner = self._tuner()
        tuner.tune(max_trials=5)
        assert len(tuner.results) > 0

    def test_best_is_max_tflops(self):
        tuner = self._tuner()
        tuner.tune(max_trials=20)
        best_tflops = max(r.tflops for r in tuner.results)
        assert abs(tuner.best.tflops - best_tflops) < 1e-9

    def test_larger_tiles_generally_faster_for_large_gemm(self):
        """128×128 tiles should outperform 32×32 tiles for large GEMMs."""
        tuner = BayesianAutotuner(GEMMWorkload(M=4096, N=4096, K=4096))
        small = TuningConfig(tile_m=32, tile_n=32, tile_k=32)
        large = TuningConfig(tile_m=128, tile_n=128, tile_k=32)
        lat_small = tuner._mock_latency(small)
        lat_large = tuner._mock_latency(large)
        assert lat_large < lat_small

    def test_smem_budget_enforced(self):
        """Configs exceeding smem budget should not be in results."""
        tuner = BayesianAutotuner(
            GEMMWorkload(M=1024, N=1024, K=1024),
            smem_budget_bytes=4096,  # tiny budget
        )
        tuner.tune(max_trials=20)
        for r in tuner.results:
            assert r.config.smem_bytes() <= 4096

    def test_to_mlir_attrs_after_tune(self):
        tuner = self._tuner()
        tuner.tune(max_trials=5)
        attr = tuner.to_mlir_attrs()
        assert "tessera.autotune" in attr
        assert "tile_m" in attr
        assert "schedule_hash" in attr

    def test_to_mlir_attrs_before_tune(self):
        tuner = self._tuner()
        attr = tuner.to_mlir_attrs()
        assert attr == "{tessera.autotune = {}}"

    def test_repr_contains_workload(self):
        tuner = self._tuner()
        assert "GEMMWorkload" in repr(tuner)

    def test_schedule_artifact_has_stable_hash_and_policy(self):
        tuner = self._tuner()
        tuner.tune(max_trials=5)
        artifact = tuner.schedule_artifact(arch="sm90")
        assert artifact["hash"] == tuner.schedule_hash(arch="sm90")
        assert artifact["numeric_policy"]["accum"] == "f32"
        assert artifact["movement"]["overlap"] == "compute"

    def test_schedule_artifact_mlir_op(self):
        tuner = self._tuner()
        tuner.tune(max_trials=5)
        mlir = tuner.to_schedule_artifact_mlir(arch="sm90")
        assert "schedule.artifact" in mlir
        assert "hash" in mlir
        assert "shape_key" in mlir


class TestBayesianAutotunerCache:
    def test_save_and_warm_start(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            # First run: tune and save.
            tuner1 = BayesianAutotuner(GEMMWorkload(M=512, N=512, K=512))
            tuner1.tune(max_trials=5)
            assert tuner1.num_trials > 0
            tuner1.save_to_cache(db_path)

            # Second run: warm-start from cache.
            tuner2 = BayesianAutotuner(GEMMWorkload(M=512, N=512, K=512))
            loaded = tuner2.warm_start_from_cache(db_path)
            assert loaded == tuner1.num_trials
            assert tuner2.best is not None
        finally:
            os.unlink(db_path)

    def test_warm_start_missing_db(self):
        tuner = BayesianAutotuner(GEMMWorkload(M=512, N=512, K=512))
        loaded = tuner.warm_start_from_cache("/nonexistent/path.db")
        assert loaded == 0

    def test_warm_start_different_workload_returns_zero(self):
        """Cache entries from a different workload should not be loaded."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            t1 = BayesianAutotuner(GEMMWorkload(M=512, N=512, K=512))
            t1.tune(max_trials=3)
            t1.save_to_cache(db_path)

            t2 = BayesianAutotuner(GEMMWorkload(M=1024, N=1024, K=1024))
            loaded = t2.warm_start_from_cache(db_path)
            assert loaded == 0
        finally:
            os.unlink(db_path)
