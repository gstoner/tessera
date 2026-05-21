"""Visual-complex-core benchmark tests.

Cross-lane (GA × EBM) library benchmark — pins that the two compiler
surfaces co-exist in one module and that the cross-lane composition
produces results matching an independent oracle.

Mirrors ``test_clifford_core_benchmark.py`` and
``test_energy_core_benchmark.py``; the new content is the *cross-lane*
oracle comparison and the IR-visible fixture that names ops from both
lanes in one function.
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

from benchmarks.visual_complex_core import (  # noqa: E402
    VisualComplexCoreBenchmark,
    VisualComplexCoreConfig,
    VisualComplexCoreModel,
    clifford_energy,
    composition_oracle,
)
from benchmarks.energy_core.core import annealing_schedule  # noqa: E402
import tessera.ga as ga  # noqa: E402
from tessera.ga.multivector import Multivector  # noqa: E402


FIXTURE = (
    REPO_ROOT / "tests" / "tessera-ir" / "phase7"
    / "visual_complex_core_ir_visible.mlir"
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
# Cross-lane energy primitive
# --------------------------------------------------------------------------- #


class TestCliffordEnergy:
    def test_zero_state_against_zero_target_is_zero(self) -> None:
        state = np.zeros((3, 8), dtype=np.float32)
        target = np.zeros((8,), dtype=np.float32)
        energies = clifford_energy(state, target)
        np.testing.assert_array_equal(energies, np.zeros((3,), dtype=np.float32))

    def test_per_row_matches_half_squared_diff_norm(self) -> None:
        rng = np.random.default_rng(1)
        state = rng.standard_normal((5, 8)).astype(np.float32)
        target = rng.standard_normal((8,)).astype(np.float32)
        energies = clifford_energy(state, target)
        # Cl(3, 0) is Euclidean; norm_squared(mv) == Σ_i c_i².
        ref = 0.5 * np.sum((state - target[None, :]) ** 2, axis=-1).astype(np.float32)
        np.testing.assert_allclose(energies, ref, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------- #
# Cross-lane model + oracle
# --------------------------------------------------------------------------- #


class TestVisualComplexCoreModel:
    def _cfg(self, **kw) -> VisualComplexCoreConfig:
        base = dict(B=4, n_rotors=2, n_steps=3,
                     eta=0.05, T_max=1.0, T_min=0.1, seed=0)
        base.update(kw)
        return VisualComplexCoreConfig(**base)

    def test_forward_shapes(self) -> None:
        cfg = self._cfg()
        m = VisualComplexCoreModel(cfg)
        state, bivec, energies, Z = m()
        assert state.shape == (cfg.B, 8)
        assert bivec.shape == (cfg.B, 8)
        assert energies.shape == (cfg.B,)
        assert isinstance(Z, float) and Z >= 0.0

    def test_forward_matches_cross_lane_oracle(self) -> None:
        """The single most-valuable test in this file: the GA × EBM
        composition must agree with an independent inline re-derivation
        of the same chain.  Any drift here means *cross-lane*
        composition broke."""
        cfg = self._cfg(B=2, n_rotors=2, n_steps=3)
        m = VisualComplexCoreModel(cfg)
        sched = annealing_schedule(cfg.n_steps, T_max=cfg.T_max, T_min=cfg.T_min)
        oracle_state, oracle_bivec, oracle_energies, oracle_Z = composition_oracle(
            m.initial_coeffs, m.target_coeffs, m.rotors(),
            eta=cfg.eta, schedule=sched, base_key=m.chain_key,
        )
        model_state, model_bivec, model_energies, model_Z = m()
        np.testing.assert_allclose(model_state, oracle_state, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(model_bivec, oracle_bivec, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(model_energies, oracle_energies, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(model_Z, oracle_Z, rtol=1e-4, atol=1e-5)

    def test_forward_determinism_across_instances(self) -> None:
        cfg = self._cfg(seed=42)
        a = VisualComplexCoreModel(cfg)()
        b = VisualComplexCoreModel(cfg)()
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        np.testing.assert_array_equal(a[2], b[2])
        assert a[3] == b[3]

    def test_grade_2_projection_zeros_non_bivector_blades(self) -> None:
        """The Cl(3, 0) bivector grade lives at coefficient indices
        3 (e12), 5 (e13), 6 (e23); all other indices must be zero
        after grade_projection."""
        cfg = self._cfg(B=2, n_rotors=1, n_steps=1)
        m = VisualComplexCoreModel(cfg)
        _, bivec, _, _ = m()
        # Indices 0 (scalar), 1 (e1), 2 (e2), 4 (e3), 7 (e123) must be zero.
        for idx in (0, 1, 2, 4, 7):
            np.testing.assert_array_equal(
                bivec[:, idx], np.zeros((cfg.B,), dtype=np.float32),
                err_msg=f"non-bivector index {idx} not zero in projection",
            )


# --------------------------------------------------------------------------- #
# Benchmark harness
# --------------------------------------------------------------------------- #


class TestVisualComplexCoreBenchmark:
    def test_result_schema(self) -> None:
        cfg = VisualComplexCoreConfig(B=2, n_rotors=1, n_steps=2, seed=0)
        bench = VisualComplexCoreBenchmark(warmup=1, reps=2)
        res = bench.run_one(cfg)
        d = res.to_dict()
        for field in (
            "backend", "op", "shape", "dtype", "latency_ms",
            "throughput_msps", "memory_bw_gb_s", "device",
            "tessera_version", "determinism_ok",
        ):
            assert field in d
        assert d["op"] == "visual_complex_core_forward"
        assert d["shape"]["algebra"] == "Cl(3,0)"
        assert d["determinism_ok"] is True

    def test_json_roundtrip(self, tmp_path) -> None:
        cfg = VisualComplexCoreConfig(B=2, n_rotors=1, n_steps=2, seed=0)
        bench = VisualComplexCoreBenchmark(warmup=1, reps=2)
        results = bench.run([cfg, cfg])
        out = tmp_path / "vc.json"
        bench.to_json(results, str(out))
        loaded = json.loads(out.read_text())
        assert len(loaded) == 2
        for row in loaded:
            assert row["shape"]["n_rotors"] == 1
            assert row["shape"]["algebra"] == "Cl(3,0)"


# --------------------------------------------------------------------------- #
# IR fixture cross-lane
# --------------------------------------------------------------------------- #


def test_ir_fixture_names_both_lanes() -> None:
    text = FIXTURE.read_text()
    # GA lane ops.
    assert "tessera_clifford.rotor_sandwich" in text
    assert "tessera_clifford.grade_projection" in text
    # EBM lane ops.
    assert "tessera_ebm.energy_quadratic" in text
    assert "tessera_ebm.langevin_step" in text
    assert "tessera_ebm.partition_exact" in text
    assert "tessera_ebm.logsumexp" in text
    # Cross-lane signature attribute survives.
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
    for op in (
        "tessera_clifford.rotor_sandwich",
        "tessera_clifford.grade_projection",
        "tessera_ebm.energy_quadratic",
        "tessera_ebm.langevin_step",
        "tessera_ebm.partition_exact",
        "tessera_ebm.logsumexp",
    ):
        assert op in out


def test_ir_fixture_co_locates_ga_and_ebm_in_one_function() -> None:
    """The cross-lane contract: GA and EBM ops MUST share the same
    function scope so a single lowering pipeline (when one ships) can
    see both lanes' ops at once."""
    text = FIXTURE.read_text()
    func_start = text.find("func.func @visual_complex_block")
    func_end = text.find("\n}", func_start)
    body = text[func_start:func_end]
    assert "tessera_clifford" in body
    assert "tessera_ebm" in body
