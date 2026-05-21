"""Clifford-core benchmark tests.

Sister suite to ``test_grid_ai_core_benchmark.py``.  The library benchmark
exists *above* Tessera; this file pins:

  * tile_multivectors structural correctness
  * RotorSampler determinism (same seed → same rotor lineage)
  * Forward composition matches an independent CPU oracle
  * Benchmark result schema is the canonical Architecture Decision #12 shape
  * IR fixture names all required compiler-visible Clifford ops
  * Forward determinism across two model instances with the same seed
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "python"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benchmarks.clifford_core import (  # noqa: E402
    CliffordCoreBenchmark,
    CliffordCoreConfig,
    CliffordCoreModel,
    RotorSampler,
    multivector_oracle_chain,
    tile_multivectors,
)
import tessera.ga as ga  # noqa: E402
from tessera.ga.multivector import Multivector  # noqa: E402
from tessera.rng import RNGKey  # noqa: E402


FIXTURE = (
    REPO_ROOT / "tests" / "tessera-ir" / "phase7"
    / "clifford_core_ir_visible.mlir"
)


# --------------------------------------------------------------------------- #
# Helpers (mirror test_grid_ai_core_benchmark conventions)
# --------------------------------------------------------------------------- #


def _find_tessera_opt() -> str | None:
    for c in (
        os.environ.get("TESSERA_OPT"),
        shutil.which("tessera-opt"),
        str(REPO_ROOT / "build" / "tools" / "tessera-opt" / "tessera-opt"),
    ):
        if c and Path(c).exists():
            return c
    return None


# --------------------------------------------------------------------------- #
# Library primitives
# --------------------------------------------------------------------------- #


class TestTileMultivectors:
    def test_row_major_split_along_batch(self) -> None:
        arr = np.arange(16 * 8, dtype=np.float32).reshape(16, 8)
        tiles = tile_multivectors(arr, 4)
        assert len(tiles) == 4
        np.testing.assert_array_equal(tiles[0], arr[0:4])
        np.testing.assert_array_equal(tiles[3], arr[12:16])

    def test_rejects_non_divisible_tile(self) -> None:
        with pytest.raises(ValueError, match="does not divide batch"):
            tile_multivectors(np.zeros((5, 8), dtype=np.float32), 2)


class TestRotorSampler:
    def test_same_key_produces_same_rotor_lineage(self) -> None:
        algebra = ga.Cl(3, 0)
        key = RNGKey.from_seed(42, name="rotor-test")
        s1 = RotorSampler(algebra, key)
        s2 = RotorSampler(algebra, key)
        for _ in range(3):
            r1 = s1.next_rotor()
            r2 = s2.next_rotor()
            np.testing.assert_array_equal(r1.coefficients, r2.coefficients)

    def test_advances_counter_between_calls(self) -> None:
        algebra = ga.Cl(3, 0)
        sampler = RotorSampler(algebra, RNGKey.from_seed(0))
        r0 = sampler.next_rotor()
        r1 = sampler.next_rotor()
        # Two sequential rotors must differ — counter is part of fold_in.
        assert not np.array_equal(r0.coefficients, r1.coefficients)


# --------------------------------------------------------------------------- #
# Model forward + oracle
# --------------------------------------------------------------------------- #


class TestCliffordCoreModel:
    def _cfg(self, **kw) -> CliffordCoreConfig:
        base = dict(B=4, algebra_signature=(3, 0, 0), tile=2, n_rotors=2, seed=0)
        base.update(kw)
        return CliffordCoreConfig(**base)

    def test_forward_shapes(self) -> None:
        cfg = self._cfg()
        m = CliffordCoreModel(cfg)
        x = np.random.default_rng(0).standard_normal((cfg.B, 8)).astype(np.float32)
        composed, bivec, nsq = m(x)
        assert composed.coefficients.shape == (cfg.B, 8)
        assert bivec.shape == (cfg.B, 8)
        assert nsq.shape == (cfg.B,)

    def test_forward_matches_independent_oracle(self) -> None:
        """The forward path must match a hand-rolled GA composition built
        in the same order — proving the *composition* contract (not per-op
        which the GA unit tests already cover)."""
        cfg = self._cfg()
        m = CliffordCoreModel(cfg)
        x_coeffs = np.random.default_rng(123).standard_normal(
            (cfg.B, 8)
        ).astype(np.float32)
        x_mv = Multivector(x_coeffs, m.algebra)
        # Re-build the same rotors from the same seed lineage.
        sampler_key = RNGKey.from_seed(cfg.seed, name="clifford_core").fold_in("sampler")
        sampler = RotorSampler(m.algebra, sampler_key)
        rotors = [sampler.next_rotor() for _ in range(cfg.n_rotors)]
        # Oracle.
        composed_oracle, bivec_oracle, nsq_oracle = multivector_oracle_chain(
            x_mv, rotors,
        )
        # Model output.
        composed, bivec, nsq = m(x_coeffs)
        np.testing.assert_allclose(
            composed.coefficients, composed_oracle.coefficients,
            rtol=1e-5, atol=1e-6,
        )
        np.testing.assert_allclose(bivec, bivec_oracle, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(nsq, nsq_oracle, rtol=1e-5, atol=1e-6)

    def test_forward_determinism_across_instances(self) -> None:
        cfg = self._cfg(seed=7)
        m1 = CliffordCoreModel(cfg)
        m2 = CliffordCoreModel(cfg)
        x = np.random.default_rng(99).standard_normal((cfg.B, 8)).astype(np.float32)
        c1, b1, n1 = m1(x)
        c2, b2, n2 = m2(x)
        np.testing.assert_array_equal(c1.coefficients, c2.coefficients)
        np.testing.assert_array_equal(b1, b2)
        np.testing.assert_array_equal(n1, n2)


# --------------------------------------------------------------------------- #
# Benchmark harness
# --------------------------------------------------------------------------- #


class TestCliffordCoreBenchmark:
    def test_result_schema(self) -> None:
        cfg = CliffordCoreConfig(B=4, tile=2, n_rotors=1, seed=0)
        bench = CliffordCoreBenchmark(warmup=1, reps=2)
        res = bench.run_one(cfg)
        d = res.to_dict()
        for field in (
            "backend", "op", "shape", "dtype", "latency_ms",
            "throughput_msps", "memory_bw_gb_s", "device",
            "tessera_version", "determinism_ok",
        ):
            assert field in d
        assert d["backend"] == "tessera-reference"
        assert d["op"] == "clifford_core_forward"
        assert d["dtype"] == "fp32"
        assert d["determinism_ok"] is True
        assert d["latency_ms"] >= 0

    def test_json_roundtrip(self, tmp_path) -> None:
        cfg = CliffordCoreConfig(B=4, tile=2, n_rotors=1, seed=0)
        bench = CliffordCoreBenchmark(warmup=1, reps=2)
        results = bench.run([cfg, cfg])
        out = tmp_path / "cliff.json"
        bench.to_json(results, str(out))
        loaded = json.loads(out.read_text())
        assert len(loaded) == 2
        for row in loaded:
            assert row["shape"]["algebra_signature"] == [3, 0, 0]


# --------------------------------------------------------------------------- #
# IR fixture — names all required compiler-visible Clifford ops
# --------------------------------------------------------------------------- #


def test_ir_fixture_names_required_clifford_ops() -> None:
    text = FIXTURE.read_text()
    for op in (
        "tessera_clifford.rotor_from_axis",
        "tessera_clifford.rotor_sandwich",
        "tessera_clifford.geometric_product",
        "tessera_clifford.grade_projection",
        "tessera_clifford.norm_squared",
    ):
        assert op in text, f"missing {op} from IR-visible fixture"
    # The Cl(3, 0) signature is documented as an attribute.
    assert "algebra_signature = [3, 0, 0]" in text


def test_ir_fixture_roundtrips_through_tessera_opt() -> None:
    binary = _find_tessera_opt()
    if binary is None:
        pytest.skip("tessera-opt not built")
    r = subprocess.run(
        [binary, "--allow-unregistered-dialect", str(FIXTURE)],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, r.stderr
    out = r.stdout
    # Every op survives the parse → print → parse roundtrip.
    for op in (
        "tessera_clifford.rotor_from_axis",
        "tessera_clifford.rotor_sandwich",
        "tessera_clifford.geometric_product",
        "tessera_clifford.grade_projection",
        "tessera_clifford.norm_squared",
    ):
        assert op in out
