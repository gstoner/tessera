"""Energy-core benchmark tests.

Sister suite to ``test_clifford_core_benchmark.py``.  Pins:
  * annealing_schedule structural correctness
  * energy_grid_oracle stable logsumexp match
  * Forward composition matches an independent CPU Langevin chain oracle
  * Benchmark result schema is canonical
  * IR fixture names all required compiler-visible EBM ops
  * Forward determinism across two model instances with the same seed
"""
from __future__ import annotations

import json
import math
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

from benchmarks.energy_core import (  # noqa: E402
    EnergyCoreBenchmark,
    EnergyCoreConfig,
    EnergyCoreModel,
    annealing_schedule,
    energy_grid_oracle,
    langevin_chain_oracle,
    quadratic_energy,
)
from tessera.rng import RNGKey  # noqa: E402


FIXTURE = (
    REPO_ROOT / "tests" / "tessera-ir" / "phase7"
    / "energy_core_ir_visible.mlir"
)


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
# Helpers
# --------------------------------------------------------------------------- #


class TestAnnealingSchedule:
    def test_linear_from_T_max_to_T_min(self) -> None:
        sched = annealing_schedule(5, T_max=1.0, T_min=0.1)
        assert sched.shape == (5,)
        np.testing.assert_allclose(sched[0], 1.0)
        np.testing.assert_allclose(sched[-1], 0.1)
        # Monotonically decreasing.
        assert np.all(np.diff(sched) < 0)

    def test_n_steps_one(self) -> None:
        sched = annealing_schedule(1, T_max=2.0, T_min=0.5)
        assert sched.shape == (1,)
        np.testing.assert_allclose(sched[0], 2.0)

    def test_rejects_invalid_temperatures(self) -> None:
        with pytest.raises(ValueError, match="T_max >= T_min"):
            annealing_schedule(3, T_max=0.5, T_min=1.0)
        with pytest.raises(ValueError, match="T_max >= T_min"):
            annealing_schedule(3, T_max=0.0, T_min=0.0)


class TestEnergyGridOracle:
    def test_matches_numpy_logsumexp_directly(self) -> None:
        rng = np.random.default_rng(0)
        states = rng.standard_normal((6, 4)).astype(np.float32)
        target = rng.standard_normal((4,)).astype(np.float32)
        Z = energy_grid_oracle(states, target, temperature=0.5)
        # Independent recomputation.
        diff = states - target[None, :]
        energies = 0.5 * np.sum(diff * diff, axis=-1)
        log_Z_ref = float(
            np.log(np.exp(-energies.astype(np.float64) / 0.5).sum())
        )
        np.testing.assert_allclose(math.log(Z), log_Z_ref, rtol=1e-5)


# --------------------------------------------------------------------------- #
# Model forward + Langevin chain oracle
# --------------------------------------------------------------------------- #


class TestEnergyCoreModel:
    def _cfg(self, **kw) -> EnergyCoreConfig:
        base = dict(B=4, D=4, n_steps=3, eta=0.05, T_max=1.0, T_min=0.1, seed=0)
        base.update(kw)
        return EnergyCoreConfig(**base)

    def test_forward_shapes(self) -> None:
        cfg = self._cfg()
        m = EnergyCoreModel(cfg)
        y, energies, log_Z = m()
        assert y.shape == (cfg.B, cfg.D)
        assert energies.shape == (cfg.B,)
        assert isinstance(log_Z, float) and not math.isnan(log_Z)

    def test_forward_matches_langevin_chain_oracle(self) -> None:
        """The Langevin chain inside EnergyCoreModel must produce
        bit-identical results to the inline analytic-gradient oracle
        when both share the same RNGKey lineage."""
        cfg = self._cfg(B=2, D=4, n_steps=3)
        m = EnergyCoreModel(cfg)
        y_model, _, _ = m()
        sched = annealing_schedule(cfg.n_steps, T_max=cfg.T_max, T_min=cfg.T_min)
        y_oracle = langevin_chain_oracle(
            m.initial_state, m.target,
            eta=cfg.eta, schedule=sched, base_key=m.chain_key,
        )
        np.testing.assert_allclose(y_model, y_oracle, rtol=1e-4, atol=1e-5)

    def test_forward_determinism_across_instances(self) -> None:
        cfg = self._cfg(seed=11)
        a = EnergyCoreModel(cfg)()
        b = EnergyCoreModel(cfg)()
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        assert a[2] == b[2]


# --------------------------------------------------------------------------- #
# Benchmark harness
# --------------------------------------------------------------------------- #


class TestEnergyCoreBenchmark:
    def test_result_schema(self) -> None:
        cfg = EnergyCoreConfig(B=2, D=4, n_steps=2, seed=0)
        bench = EnergyCoreBenchmark(warmup=1, reps=2)
        res = bench.run_one(cfg)
        d = res.to_dict()
        for field in (
            "backend", "op", "shape", "dtype", "latency_ms",
            "throughput_msps", "memory_bw_gb_s", "device",
            "tessera_version", "determinism_ok",
        ):
            assert field in d
        assert d["op"] == "energy_core_forward"
        assert d["dtype"] == "fp32"
        assert d["determinism_ok"] is True

    def test_json_roundtrip(self, tmp_path) -> None:
        cfg = EnergyCoreConfig(B=2, D=4, n_steps=2, seed=0)
        bench = EnergyCoreBenchmark(warmup=1, reps=2)
        results = bench.run([cfg, cfg])
        out = tmp_path / "energy.json"
        bench.to_json(results, str(out))
        loaded = json.loads(out.read_text())
        assert len(loaded) == 2
        for row in loaded:
            assert row["shape"]["B"] == 2
            assert row["shape"]["n_steps"] == 2


# --------------------------------------------------------------------------- #
# IR-visible fixture
# --------------------------------------------------------------------------- #


def test_ir_fixture_names_required_ebm_ops() -> None:
    text = FIXTURE.read_text()
    for op in (
        "tessera_ebm.energy_quadratic",
        "tessera_ebm.annealing_schedule",
        "tessera_ebm.langevin_step",
        "tessera_ebm.partition_exact",
        "tessera_ebm.logsumexp",
    ):
        assert op in text, f"missing {op} from IR-visible fixture"
    assert "eta = 5.000000e-02 : f32" in text
    assert "temperature = 1.000000e-01 : f32" in text


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
    for op in (
        "tessera_ebm.energy_quadratic",
        "tessera_ebm.annealing_schedule",
        "tessera_ebm.langevin_step",
        "tessera_ebm.partition_exact",
        "tessera_ebm.logsumexp",
    ):
        assert op in out
