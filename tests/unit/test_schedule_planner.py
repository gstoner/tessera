from __future__ import annotations

from tessera.compiler.autotune_v2 import GEMMWorkload
from tessera.compiler.schedule_planner import SchedulePlanner, schedule_cache_key


def test_schedule_planner_selects_deterministic_gemm_schedule():
    planner = SchedulePlanner()

    first = planner.plan_gemm(m=256, n=256, k=128, target="sm90")
    second = planner.plan_gemm(m=256, n=256, k=128, target="cuda")

    assert first.to_dict() == second.to_dict()
    assert first.target == "nvidia_sm90"
    assert first.config.tile_m in {32, 64, 128, 256}
    assert first.cache_key == schedule_cache_key("tessera.matmul", (256, 256, 128), dtype="bf16", target="nvidia_sm90")


def test_schedule_planner_exposes_legality_rejections():
    planner = SchedulePlanner()
    candidates = planner.gemm_candidates(GEMMWorkload(64, 64, 64), target="cpu")

    assert any(candidate.legal for candidate in candidates)
    assert any(candidate.reason == "tile_exceeds_problem_shape" for candidate in candidates)
