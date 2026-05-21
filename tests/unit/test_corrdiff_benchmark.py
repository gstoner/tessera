"""Sub-5 — CorrDiff-core benchmark library.

Forward-pass correctness + determinism + JSON-schema conformance for
``benchmarks/corrdiff``.  The benchmark integrates conv2d + 2D
local-window attention + deterministic Philox diffusion noise +
activation checkpointing + tile_field; this module locks each piece.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "python"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benchmarks.corrdiff import (  # noqa: E402
    CorrDiffBenchmark,
    CorrDiffConfig,
    CorrDiffModel,
    diffusion_noise_step,
    tile_field,
)
from tessera.rng import RNGKey  # noqa: E402


# --------------------------------------------------------------------------- #
# tile_field
# --------------------------------------------------------------------------- #


class TestTileField:
    def test_row_major_decomposition(self):
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        tiles = tile_field(arr, (2, 2))
        assert len(tiles) == 4
        # Top-left tile = [[0,1],[4,5]]
        np.testing.assert_array_equal(tiles[0], [[0, 1], [4, 5]])
        # Bottom-right tile = [[10,11],[14,15]]
        np.testing.assert_array_equal(tiles[3], [[10, 11], [14, 15]])

    def test_carries_leading_axes(self):
        arr = np.zeros((2, 3, 4, 4), dtype=np.float32)
        tiles = tile_field(arr, (2, 2))
        assert len(tiles) == 4
        for t in tiles:
            assert t.shape == (2, 3, 2, 2)

    def test_rejects_non_divisible_tile(self):
        with pytest.raises(ValueError, match="does not divide"):
            tile_field(np.zeros((5, 4)), (2, 2))


# --------------------------------------------------------------------------- #
# diffusion_noise_step
# --------------------------------------------------------------------------- #


class TestDiffusionNoise:
    def test_deterministic_with_same_key(self):
        x = np.zeros((2, 4, 4), dtype=np.float32)
        rng = RNGKey.from_seed(42)
        a = diffusion_noise_step(x, rng, 0.1)
        b = diffusion_noise_step(x, rng, 0.1)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_produces_different_noise(self):
        x = np.zeros((2, 4, 4), dtype=np.float32)
        a = diffusion_noise_step(x, RNGKey.from_seed(1), 0.1)
        b = diffusion_noise_step(x, RNGKey.from_seed(2), 0.1)
        assert not np.array_equal(a, b)

    def test_sigma_zero_is_identity(self):
        x = np.full((2, 4, 4), 3.0, dtype=np.float32)
        out = diffusion_noise_step(x, RNGKey.from_seed(0), 0.0)
        np.testing.assert_array_equal(out, x)


# --------------------------------------------------------------------------- #
# CorrDiffModel
# --------------------------------------------------------------------------- #


class TestCorrDiffModel:
    def _cfg(self, **kw) -> CorrDiffConfig:
        base = dict(B=2, H=16, W=16, C_in=4, C_hid=8, C_out=4,
                     heads=2, window=(1, 1), seed=0)
        base.update(kw)
        return CorrDiffConfig(**base)

    def test_output_shape(self):
        cfg = self._cfg()
        model = CorrDiffModel(cfg)
        x = np.zeros((cfg.B, cfg.H, cfg.W, cfg.C_in), dtype=np.float32)
        y = model(x, step=0)
        assert y.shape == (cfg.B, cfg.H, cfg.W, cfg.C_out)

    def test_forward_is_deterministic(self):
        """Two model instances with the same config produce bit-identical
        weights and bit-identical outputs."""
        cfg = self._cfg()
        m1 = CorrDiffModel(cfg)
        m2 = CorrDiffModel(cfg)
        np.testing.assert_array_equal(m1.W1, m2.W1)
        np.testing.assert_array_equal(m1.W2, m2.W2)
        np.testing.assert_array_equal(m1.W_out, m2.W_out)
        x = np.zeros((cfg.B, cfg.H, cfg.W, cfg.C_in), dtype=np.float32)
        np.testing.assert_array_equal(m1(x, step=5), m2(x, step=5))

    def test_different_seed_changes_weights(self):
        a = CorrDiffModel(self._cfg(seed=0))
        b = CorrDiffModel(self._cfg(seed=1))
        assert not np.array_equal(a.W1, b.W1)

    def test_rejects_C_hid_not_divisible_by_heads(self):
        with pytest.raises(ValueError, match="divisible"):
            CorrDiffModel(self._cfg(C_hid=7, heads=2))

    def test_rejects_wrong_input_shape(self):
        cfg = self._cfg()
        m = CorrDiffModel(cfg)
        bad = np.zeros((cfg.B, cfg.H + 1, cfg.W, cfg.C_in), dtype=np.float32)
        with pytest.raises(ValueError, match="does not match cfg"):
            m(bad)

    def test_asymmetric_window(self):
        cfg = self._cfg(window=(2, 1))
        m = CorrDiffModel(cfg)
        x = np.zeros((cfg.B, cfg.H, cfg.W, cfg.C_in), dtype=np.float32)
        y = m(x, step=0)
        assert y.shape == (cfg.B, cfg.H, cfg.W, cfg.C_out)


# --------------------------------------------------------------------------- #
# Benchmark harness
# --------------------------------------------------------------------------- #


class TestCorrDiffBenchmark:
    def test_run_one_emits_canonical_schema(self):
        cfg = CorrDiffConfig(B=1, H=8, W=8, C_in=4, C_hid=8, C_out=4,
                              heads=2, window=(1, 1), seed=0)
        bench = CorrDiffBenchmark(warmup=1, reps=2)
        res = bench.run_one(cfg)
        # Architecture Decision #12 fields must all be present.
        d = res.to_dict()
        for field in (
            "backend", "op", "shape", "dtype", "latency_ms",
            "throughput_msps", "memory_bw_gb_s", "device",
            "tessera_version", "determinism_ok",
        ):
            assert field in d
        assert d["backend"]   == "tessera-reference"
        assert d["op"]        == "corrdiff_forward"
        assert d["dtype"]     == "fp32"
        assert d["device"]    == "cpu"
        assert d["determinism_ok"] is True
        # Numeric fields must be finite and non-negative.
        assert d["latency_ms"] >= 0
        assert d["throughput_msps"] >= 0
        assert d["memory_bw_gb_s"] >= 0

    def test_to_json_roundtrips(self, tmp_path):
        cfg = CorrDiffConfig(B=1, H=8, W=8, C_in=4, C_hid=8, C_out=4,
                              heads=2, window=(1, 1), seed=0)
        bench = CorrDiffBenchmark(warmup=1, reps=2)
        results = bench.run([cfg, cfg])
        out = tmp_path / "smoke.json"
        bench.to_json(results, str(out))
        with open(out) as f:
            loaded = json.load(f)
        assert isinstance(loaded, list)
        assert len(loaded) == 2
        # Re-validate schema on loaded JSON.
        for row in loaded:
            for field in ("backend", "op", "shape", "latency_ms"):
                assert field in row
            assert row["shape"]["window"] == [1, 1]


# --------------------------------------------------------------------------- #
# Integration story — exercise all five pieces in one path
# --------------------------------------------------------------------------- #


def test_corrdiff_uses_each_of_the_five_pieces():
    """Cross-link test — names the five pieces this benchmark integrates
    and verifies each one is actually invoked in the model's forward."""
    import inspect
    from benchmarks.corrdiff import corrdiff_core as mod
    src = inspect.getsource(mod.CorrDiffModel.forward)
    # 1. conv2d (NHWC) — backbone
    assert "ts.ops.conv2d" in src
    # 2. attn_local_window_2d — spatial bias
    assert "ts.ops.attn_local_window_2d" in src
    # 3. deterministic diffusion noise via RNGKey
    assert "diffusion_noise_step" in src
    # 4. activation checkpointing
    assert "checkpoint(" in src
    # 5. tiled fields — separate helper, but tested above
    assert callable(tile_field)
