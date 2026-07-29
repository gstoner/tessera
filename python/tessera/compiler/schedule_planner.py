"""First-class schedule planning contracts for legality, cost, and selection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

from .autotune_v2 import GEMMWorkload, LegalGEMMCandidateGenerator, TuningConfig
from .capabilities import CAPABILITY_REGISTRY_VERSION, normalize_target

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .target_perf import TargetPerf


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
    """Deterministic legality -> cost -> selected schedule pipeline.

    .. warning::
       The **legality** half is real; the **cost** half is not yet. See
       :func:`_estimate_latency_ms` — it has no memory term, so it cannot tell a
       compute-bound shape from a memory-bound one and must not be treated as a
       ranking oracle. Prefer :meth:`for_target` over the bare constructor so at
       least the peak is the right hardware's, and treat measured arbitration
       (``emit/autotune.py``) as the only load-bearing scorer.
       See ``docs/audit/compiler/TILESIGHT_ASSESSMENT.md`` §2.
    """

    def __init__(self, *, peak_tflops: float = 312.0, smem_budget_bytes: int = 98_304,
                 perf: "TargetPerf | None" = None) -> None:
        self.peak_tflops = peak_tflops
        self.smem_budget_bytes = smem_budget_bytes
        #: The device profile these numbers came from, when constructed via
        #: :meth:`for_target`. ``None`` means the caller supplied bare numbers
        #: (or accepted the default, which is A100 bf16 — see the class note).
        self.perf = perf

    @classmethod
    def for_target(cls, target: object, *, dtype: str = "bf16",
                   device: str | None = None,
                   smem_budget_bytes: int | None = None) -> "SchedulePlanner":
        """Build a planner from a target's **calibrated** performance profile
        rather than the A100-shaped default.

        The bare ``SchedulePlanner()`` default is ``peak_tflops=312.0`` — A100
        dense bf16 — applied to every target including CPUs and Apple GPUs. This
        constructor sources the peak and the shared-memory budget from
        :mod:`target_perf` instead.

        Raises :class:`TargetPerfError` when the target has no registered device,
        has one but no peak for ``dtype``, or has one with no shared-memory
        capacity and no explicit ``smem_budget_bytes`` — every gap is named so it
        can be closed by a calibration sweep, rather than papered over with a
        default that would be wrong by an order of magnitude (Decision #21).

        ``smem_budget_bytes`` is the escape hatch for targets where the concept
        does not apply as a per-CU capacity (a CPU lane), so the caller states
        the budget instead of inheriting an unrelated GPU's.
        """
        from .target_perf import (
            UNIT_MATRIX, UNIT_VECTOR, TargetPerfError, perf_for_target)

        target_name = normalize_target(target)
        profile = perf_for_target(target_name, device=device)
        if profile is None:
            raise TargetPerfError(
                f"no performance profile registered for target {target_name!r}; "
                f"add one to target_perf.py before planning against it")
        peak = profile.peak(dtype, UNIT_MATRIX)
        if peak is None:
            peak = profile.peak(dtype, UNIT_VECTOR)
        if peak is None:
            raise TargetPerfError(
                f"{profile.device} ({target_name}) has no {dtype!r} peak "
                f"(matrix or vector). Run a calibration sweep and merge it via "
                f"target_perf.apply_corpus(), or add a cited value.")
        if smem_budget_bytes is None:
            smem = profile.value("smem_bytes_per_cu")
            if smem is None:
                raise TargetPerfError(
                    f"{profile.device} ({target_name}) has no "
                    f"'smem_bytes_per_cu'. Pass smem_budget_bytes= explicitly, "
                    f"or add a cited value — silently inheriting another "
                    f"device's shared-memory budget would make every legality "
                    f"verdict wrong.")
            smem_budget_bytes = int(smem)
        return cls(
            peak_tflops=peak,
            smem_budget_bytes=smem_budget_bytes,
            perf=profile,
        )

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
    """Compute-only latency estimate. **Known-weak — do not extend, replace.**

    There is no memory term here: latency is FLOPs over a fudge-factored peak, so
    every memory-bound shape is scored as if bandwidth were infinite, and the
    ``occupancy`` factor below has the sign of the real effect backwards (more
    resident blocks *deepens* the pipeline rather than merely raising
    utilisation). The replacement is tile-reuse-distance cache modelling plus a
    prologue/steady/epilogue envelope — ``TILESIGHT_ASSESSMENT.md`` §3.1 T1/T4.
    ``TargetPerf.roofline_ms`` is the interim honest bound where a byte count is
    available.
    """
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
