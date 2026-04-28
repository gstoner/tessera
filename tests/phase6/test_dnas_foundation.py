"""
Phase 6 - Differentiable NAS guide and Python foundation.
"""

from pathlib import Path

import pytest

import tessera
from tessera import arch


ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "guides" / "Tessera_Differentiable_NAS_Guide.md"


def test_dnas_guide_maps_feature_to_compiler_layers():
    text = GUIDE.read_text(encoding="utf-8")
    required = [
        "Differentiable Neural Architecture Search",
        "Graph IR",
        "Schedule IR",
        "MixedOp",
        "arch.Parameter",
        "GumbelSoftmax",
        "HardConcrete",
        "STEOneHot",
        "Bilevel Optimization",
        "Hardware-Aware Objective",
        "Freeze And Specialize",
        "Implementation Map",
    ]
    for term in required:
        assert term in text


def test_docs_map_links_dnas_guide():
    text = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    assert "docs/guides/Tessera_Differentiable_NAS_Guide.md" in text


def test_arch_namespace_is_exported():
    assert tessera.arch is arch


def test_arch_parameter_is_fp32_and_argmaxable():
    alpha = arch.Parameter(3, name="edge.alpha")
    alpha.set([0.1, 2.0, -1.0])
    alpha.set_grad([0.0, 10.0, 0.0])
    alpha.step(0.1, clip_norm=1.0)
    assert alpha.argmax() == 1
    with pytest.raises(arch.ArchitectureSearchError, match="fp32"):
        arch.Parameter(2, dtype="bf16")


def test_softmax_and_gumbel_relaxations_are_normalized():
    alpha = arch.Parameter(3)
    alpha.set([0.0, 1.0, 2.0])
    gate = arch.Softmax(alpha, temperature=2.0)()
    assert pytest.approx(sum(gate), rel=1e-7) == 1.0
    assert gate[2] > gate[1] > gate[0]

    gumbel = arch.GumbelSoftmax(alpha, temperature=1.0, seed=123)()
    assert pytest.approx(sum(gumbel), rel=1e-7) == 1.0
    assert gumbel == arch.GumbelSoftmax(alpha, temperature=1.0, seed=123)()


def test_mixed_op_gates_and_argmax():
    mixed = arch.MixedOp(["flash", "performer", "gmlp"], relax="softmax")
    mixed.alpha.set([0.0, -1.0, 3.0])
    gates = mixed.gates()
    assert pytest.approx(sum(gates), rel=1e-7) == 1.0
    assert mixed.choice() == 2
    assert arch.argmax({"attn": mixed}) == {"attn": 2}


def test_weighted_sum_switch_and_schedule_space():
    assert arch.weighted_sum([10.0, 20.0], [0.25, 0.75]) == 17.5
    assert arch.switch(["a", "b"], [0.1, 0.9], hard=True) == "b"

    sched = arch.ScheduleSpace({"tile_m": [64, 128], "layout": ["row", "tiled"]})
    sched.alpha["tile_m"].set([0.0, 2.0])
    sched.alpha["layout"].set([3.0, 0.0])
    current = sched.current(hard=True)
    assert current == {"tile_m": 128, "layout": "row"}


def test_analytical_cost_model_predicts_positive_costs():
    model = arch.AnalyticalCostModel()
    cost = model.predict(
        arch.CostFeatures(
            flops=2.0e12,
            bytes_moved=1.0e9,
            params=1.0e6,
            bandwidth_gbps=1000.0,
            peak_tflops=100.0,
        )
    )
    assert cost.latency_ms > 0.0
    assert cost.energy > 0.0
    assert cost.memory_bytes > 1.0e9


def test_autodiff_partition_helpers_and_deterministic_alpha_reduce():
    assert arch.validate_backward_wrt("arch") == "arch"
    with pytest.raises(arch.ArchitectureSearchError, match="backward"):
        arch.validate_backward_wrt("banana")

    mixed = arch.MixedOp(["a", "b"], relax="softmax")
    assert arch.arch_parameters(mixed) == [mixed.alpha]

    reduced = arch.deterministic_alpha_all_reduce(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        op="mean",
    )
    assert reduced == [3.0, 4.0]


def test_learned_surrogate_updates_from_autotuner_results():
    from tessera.compiler.autotune_v2 import BayesianAutotuner, GEMMWorkload

    tuner = BayesianAutotuner(GEMMWorkload(M=128, N=128, K=128), seed=1)
    tuner.tune(max_trials=3)
    assert len(tuner.cost_measurements()) == 3

    surrogate = arch.LearnedSurrogateCostModel(lr=1e-4)
    count = surrogate.update_from_autotuner(tuner, arch_name="sm90")
    assert count == 3
    assert surrogate.num_updates == 3

    pred = surrogate.predict(
        arch.CostFeatures(
            flops=float(tuner.workload.flops()),
            bytes_moved=1.0e6,
        )
    )
    assert pred.latency_ms >= 0.0


def test_dnas_ods_and_schedule_knob_ops_are_declared():
    graph_td = (ROOT / "src" / "compiler" / "ir" / "TesseraOps.td").read_text(
        encoding="utf-8"
    )
    for term in (
        "arch.parameter",
        "arch.gumbel_softmax",
        "arch.hard_concrete",
        "arch.ste_one_hot",
        "arch.weighted_sum",
        "arch.switch",
        "arch.mixed",
    ):
        assert term in graph_td

    schedule_td = (
        ROOT
        / "src"
        / "compiler"
        / "programming_model"
        / "ir"
        / "schedule"
        / "ScheduleMeshPipelineOps.td"
    ).read_text(encoding="utf-8")
    assert "schedule-search knob" in schedule_td
    assert "knob" in schedule_td
