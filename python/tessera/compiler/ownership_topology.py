"""Backend-neutral ownership-topology selection.

Ownership is a schedule axis, not an implementation detail.  This module
captures whether one thread, one wave, several waves, or a whole workgroup owns
an output unit and records why the choice was made.  The initial calibration is
the gfx1151 ROCM-7 sparse-selection A/B crossover; other kernels can use the
same contract without copying the sparse heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OwnershipTopology(str, Enum):
    THREAD = "thread"
    WAVE = "wave"
    MULTI_WAVE = "multi_wave"
    WORKGROUP = "workgroup"


@dataclass(frozen=True)
class OwnershipProblem:
    independent_units: int
    work_per_unit: int
    reduction_width: int = 1
    outputs_per_unit: int = 1

    def __post_init__(self) -> None:
        for name, value in (
            ("independent_units", self.independent_units),
            ("work_per_unit", self.work_per_unit),
            ("reduction_width", self.reduction_width),
            ("outputs_per_unit", self.outputs_per_unit),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")


@dataclass(frozen=True)
class OwnershipDecision:
    topology: OwnershipTopology
    reason: str
    calibration: str

    def as_metadata_dict(self) -> dict[str, str]:
        return {
            "topology": self.topology.value,
            "reason": self.reason,
            "calibration": self.calibration,
        }


def select_ownership_topology(
    problem: OwnershipProblem,
    *,
    target: str,
    operation: str = "generic",
) -> OwnershipDecision:
    """Choose an ownership topology from reusable workload dimensions.

    The only measured specialization is gfx1151 sparse top-k.  Small resident
    row batches with at least 2048 candidates amortize a cooperative wave scan;
    abundant rows expose enough independent parallelism that thread ownership
    remains faster.  Uncalibrated targets deliberately choose the conservative
    thread owner and say so rather than laundering a heuristic into proof.
    """
    chip = target.lower()
    if operation == "selection" and chip == "gfx1151":
        cooperative = (
            problem.work_per_unit >= 2048
            and problem.independent_units <= 256
            and problem.outputs_per_unit <= 8
        )
        if cooperative:
            return OwnershipDecision(
                OwnershipTopology.WAVE,
                "few independent rows and a long candidate scan amortize "
                "wave-cooperative reduction",
                "gfx1151 ROCM-7 sparse-topk A/B",
            )
        return OwnershipDecision(
            OwnershipTopology.THREAD,
            "independent row parallelism outweighs cooperative scan overhead",
            "gfx1151 ROCM-7 sparse-topk A/B",
        )
    return OwnershipDecision(
        OwnershipTopology.THREAD,
        "no target-specific cooperative crossover has been measured",
        "conservative default",
    )


__all__ = [
    "OwnershipTopology",
    "OwnershipProblem",
    "OwnershipDecision",
    "select_ownership_topology",
]
