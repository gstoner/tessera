"""Contrastive Gradient Guidance library contracts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

import tessera as ts
from tessera.diffusion_guidance import (
    ContrastivePair,
    ContrastiveScoreGuidance,
    DenoiseOutput,
    DiffusionSchedule,
    GuidanceError,
    GuidanceSafety,
    GuidedDiffusionSampler,
    ObjectiveGradientGuidance,
)
from tessera.compiler.op_catalog import get_op_spec


def _schedule():
    return DiffusionSchedule(alpha_bar=[0.99, 0.8, 0.55, 0.3])


class ConstantDenoiser:
    def __init__(self, value, *, model_id="constant", output="score"):
        self.value = np.asarray(value, dtype=np.float64)
        self.model_id = model_id
        self.output = output
        self.calls = 0

    def __call__(self, x_t, t, condition, schedule):
        self.calls += 1
        if self.output == "noise":
            return DenoiseOutput(noise_pred=self.value, prediction_type="noise", timestep=t, model_id=self.model_id)
        if self.output == "x0":
            return DenoiseOutput(x0_pred=self.value, prediction_type="x0", timestep=t, model_id=self.model_id)
        return DenoiseOutput(score=self.value, prediction_type="score", timestep=t, model_id=self.model_id)


def test_single_pair_cgg_algebra():
    sch = _schedule()
    x = np.zeros((2, 3))
    base = ConstantDenoiser(np.full((2, 3), 2.0), model_id="base")
    fav = ConstantDenoiser(np.full((2, 3), 5.0), model_id="fav")
    unfav = ConstantDenoiser(np.full((2, 3), 1.0), model_id="unfav")
    cgg = ContrastiveScoreGuidance(
        ContrastivePair(favored=fav, unfavored=unfav, gamma=0.25, name="quality")
    )
    out = cgg.apply(base, x, 2, None, sch)
    np.testing.assert_allclose(out.guided_score, np.full((2, 3), 3.0))
    assert out.scales == {"quality": 0.25}


def test_score_combine_reference_op_contract():
    base = np.array([[1.0, -2.0], [0.5, 4.0]])
    delta = np.array([[2.0, 1.0], [-1.0, 0.25]])
    out = ts.ops.score_combine(base, delta, gamma=0.5)
    np.testing.assert_allclose(out, base + 0.5 * delta)
    assert get_op_spec("score_combine").graph_name == "tessera.score_combine"
    with pytest.raises(ValueError, match="identical shapes"):
        ts.ops.score_combine(base, np.zeros((2, 1)), gamma=1.0)


def test_po_shortcut_unfavored_base_identity():
    sch = _schedule()
    x = np.zeros((2, 2))
    base = ConstantDenoiser(np.full((2, 2), 2.0), model_id="base")
    fav = ConstantDenoiser(np.full((2, 2), 6.0), model_id="po")
    cgg = ContrastiveScoreGuidance(
        ContrastivePair(favored=fav, unfavored="base", gamma=0.75, name="po")
    )
    out = cgg.apply(base, x, 1, None, sch)
    expected = (1.0 - 0.75) * 2.0 + 0.75 * 6.0
    np.testing.assert_allclose(out.guided_score, np.full((2, 2), expected))
    assert base.calls == 1


def test_multi_preference_sums_scaled_deltas():
    sch = _schedule()
    x = np.zeros((1, 4))
    base = ConstantDenoiser(np.ones((1, 4)))
    fav_a = ConstantDenoiser(np.full((1, 4), 3.0))
    fav_b = ConstantDenoiser(np.full((1, 4), -2.0))
    cgg = ContrastiveScoreGuidance((
        ContrastivePair(favored=fav_a, unfavored="base", gamma=0.5, name="a"),
        ContrastivePair(favored=fav_b, unfavored="base", gamma=0.25, name="b"),
    ))
    out = cgg.apply(base, x, 2, None, sch)
    expected = 1.0 + 0.5 * (3.0 - 1.0) + 0.25 * (-2.0 - 1.0)
    np.testing.assert_allclose(out.guided_score, np.full((1, 4), expected))


def test_noise_and_x0_outputs_normalize_to_score():
    sch = _schedule()
    x = np.full((2,), 0.4)
    noise = np.full((2,), 0.2)
    score = DenoiseOutput(noise_pred=noise).as_score(x, 2, sch)
    np.testing.assert_allclose(score, -noise / sch.sigma(2))
    x0 = np.full((2,), 0.1)
    score2 = DenoiseOutput(x0_pred=x0).as_score(x, 2, sch)
    np.testing.assert_allclose(score2, sch.score_from_x0(x, x0, 2))


def test_shape_mismatch_raises():
    sch = _schedule()
    x = np.zeros((2, 3))
    base = ConstantDenoiser(np.zeros((2, 3)))
    fav = ConstantDenoiser(np.zeros((2, 2)))
    cgg = ContrastiveScoreGuidance(ContrastivePair(favored=fav, unfavored="base"))
    with pytest.raises(GuidanceError, match="shape"):
        cgg.apply(base, x, 1, None, sch)


def test_clipping_and_timestep_gate_are_reported():
    sch = _schedule()
    x = np.zeros((2,))
    base = ConstantDenoiser(np.ones((2,)))
    fav = ConstantDenoiser(np.full((2,), 101.0))
    cgg = ContrastiveScoreGuidance(
        (
            ContrastivePair(favored=fav, unfavored="base", gamma=9.0, name="clipped"),
            ContrastivePair(favored=fav, unfavored="base", gamma=1.0, name="disabled",
                            enabled_timestep_range=(0, 0)),
        ),
        safety=GuidanceSafety(gamma_max=2.0, max_delta_norm=1.0),
    )
    out = cgg.apply(base, x, 2, None, sch)
    assert out.scales["clipped"] == 2.0
    assert "clipped" in out.metadata["clipped"]
    assert out.scales["disabled"] == 0.0
    assert out.metadata["disabled"] == ("disabled",)
    assert np.linalg.norm(out.deltas["clipped"]) <= 1.0 + 1e-12


def test_guided_sampler_is_deterministic_and_guidance_changes_path():
    sch = _schedule()
    x = np.array([0.5, -0.25])
    base = ConstantDenoiser(np.array([0.1, 0.1]))
    fav = ConstantDenoiser(np.array([0.7, -0.4]))
    cgg = ContrastiveScoreGuidance(
        ContrastivePair(favored=fav, unfavored="base", gamma=0.5, name="move")
    )
    sampler = GuidedDiffusionSampler(sch)
    a = sampler.sample(x, base, guidance=cgg)
    b = sampler.sample(x, base, guidance=cgg)
    unguided = sampler.sample(x, base)
    np.testing.assert_allclose(a.final, b.final)
    assert not np.allclose(a.final, unguided.final)
    assert len(a.trajectory) == sch.num_steps - 1


def test_objective_gradient_guidance_is_explicitly_deferred():
    with pytest.raises(NotImplementedError, match="look-ahead"):
        ObjectiveGradientGuidance()


def test_diffusion_gemma_cgg_example_runs():
    path = Path(__file__).resolve().parents[2] / "examples" / "diffusion_guidance" / "cgg_diffusion_gemma.py"
    spec = importlib.util.spec_from_file_location("cgg_diffusion_gemma_example", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    summary = mod.run_demo()
    assert summary["canvas_shape"] == [12, 64]
    assert summary["steps"] == 5
    assert summary["guidance_scales"] == {"quality": 0.75, "safety": 0.25}
    assert summary["canvas_l2_delta"] > 0.0
    assert len(summary["guided_tokens"]) == 12
    assert summary["guided_accepted"] == 12


def test_cgg_benchmark_smoke_runs():
    path = Path(__file__).resolve().parents[2] / "examples" / "diffusion_guidance" / "cgg_benchmark.py"
    spec = importlib.util.spec_from_file_location("cgg_benchmark_example", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    report = mod.run_benchmark(smoke=True)
    assert report["benchmark"] == "cgg_adapter_guidance"
    assert report["mode"] == "smoke"
    assert report["num_runs"] == 2
    assert "mean_canvas_l2_delta" in report["aggregate"]
    assert all("per_step" in run for run in report["runs"])
