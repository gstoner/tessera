"""First-class schedule planning contracts for legality, cost, and selection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from .autotune_v2 import GEMMWorkload, LegalGEMMCandidateGenerator, TuningConfig
from .capabilities import CAPABILITY_REGISTRY_VERSION, normalize_target


@dataclass(frozen=True)
class ScheduleCandidate:
    op_name: str
    target: str
    config: TuningConfig
    legal: bool
    reason: str = ""
    estimated_latency_ms: float = 0.0
    estimated_tflops: float = 0.0

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["config"] = self.config.to_dict()
        return data


@dataclass(frozen=True)
class SelectedSchedule:
    op_name: str
    target: str
    config: TuningConfig
    cache_key: str
    method: str = "roofline"
    capability_version: str = CAPABILITY_REGISTRY_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "op_name": self.op_name,
            "target": self.target,
            "config": self.config.to_dict(),
            "cache_key": self.cache_key,
            "method": self.method,
            "capability_version": self.capability_version,
        }


class SchedulePlanner:
    """Deterministic legality -> cost -> selected schedule pipeline."""

    def __init__(self, *, peak_tflops: float = 312.0, smem_budget_bytes: int = 98_304) -> None:
        self.peak_tflops = peak_tflops
        self.smem_budget_bytes = smem_budget_bytes

    def plan_gemm(
        self,
        *,
        m: int,
        n: int,
        k: int,
        dtype: str = "bf16",
        target: object = "cpu",
        method: str = "roofline",
    ) -> SelectedSchedule:
        target_name = normalize_target(target)
        workload = GEMMWorkload(m, n, k, dtype=dtype, arch=target_name)
        candidates = self.gemm_candidates(workload, target=target_name)
        legal = [candidate for candidate in candidates if candidate.legal]
        if not legal:
            cfg = TuningConfig(32, 32, 32, 1, 1)
        else:
            cfg = max(legal, key=lambda c: (c.estimated_tflops, c.config.tile_m * c.config.tile_n, -c.config.smem_bytes())).config
        return SelectedSchedule(
            op_name="tessera.matmul",
            target=target_name,
            config=cfg,
            cache_key=schedule_cache_key("tessera.matmul", (m, n, k), dtype=dtype, target=target_name),
            method=method,
        )

    def gemm_candidates(self, workload: GEMMWorkload, *, target: object = "cpu") -> list[ScheduleCandidate]:
        target_name = normalize_target(target)
        generator = LegalGEMMCandidateGenerator(
            workload,
            tile_choices=(32, 64, 128, 256),
            warp_choices=(1, 2, 4, 8),
            stage_choices=(1, 2, 3, 4),
            smem_budget_bytes=self.smem_budget_bytes,
        )
        legal_configs = generator.candidates()
        legal_by_config = {cfg: "" for cfg in legal_configs}
        for rejection in generator.rejections:
            legal_by_config[rejection.config] = rejection.reason
        candidates: list[ScheduleCandidate] = []
        for cfg, reason in legal_by_config.items():
            legal = not reason
            latency = _estimate_latency_ms(workload, cfg, self.peak_tflops) if legal else 0.0
            candidates.append(ScheduleCandidate(
                op_name="tessera.matmul",
                target=target_name,
                config=cfg,
                legal=legal,
                reason=reason,
                estimated_latency_ms=latency,
                estimated_tflops=workload.tflops_at(latency) if latency > 0 else 0.0,
            ))
        candidates.sort(key=lambda c: (not c.legal, -c.estimated_tflops, c.config.tile_m, c.config.tile_n, c.config.tile_k))
        return candidates


def schedule_cache_key(
    op_name: str,
    shape: Sequence[int],
    *,
    dtype: str,
    target: object,
    layout: str = "row_major",
    memory_policy: Mapping[str, object] | None = None,
) -> str:
    payload = {
        "op_name": op_name,
        "shape": list(shape),
        "dtype": dtype,
        "target": normalize_target(target),
        "layout": layout,
        "memory_policy": dict(memory_policy or {}),
        "capability_version": CAPABILITY_REGISTRY_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def _estimate_latency_ms(workload: GEMMWorkload, cfg: TuningConfig, peak_tflops: float) -> float:
    occupancy = min(1.0, (cfg.tile_m * cfg.tile_n) / (128 * 128))
    stage_bonus = min(1.15, 1.0 + 0.04 * max(0, cfg.num_stages - 1))
    warp_bonus = min(1.1, 0.85 + 0.05 * cfg.num_warps)
    effective_peak = max(1e-6, peak_tflops * occupancy * stage_bonus * warp_bonus)
    return workload.flops() / (effective_peak * 1e12) * 1_000.0


__all__ = [
    "ScheduleCandidate",
    "SchedulePlanner",
    "SelectedSchedule",
    "schedule_cache_key",
]
