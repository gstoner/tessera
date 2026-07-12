"""Phase-6 backward promotion gates.

Forward support never implies training support. These helpers gate only
backward execution rows and keep mock collectives/reference execution below the
device-verification bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .execution_matrix import ExecutionRow

_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class BackwardPromotion:
    eligible: bool
    status: str
    reason: str


def collective_promotion(
    provider: str,
    *,
    exact_target: str = "",
    numerical_fixture: str = "",
) -> BackwardPromotion:
    """Classify an F5 collective backward provider.

    The in-process mock proves transpose semantics, not distributed device
    execution. NCCL/RCCL promotion requires exact multi-device evidence.
    """
    normalized = provider.lower()
    if normalized in {"mock", "mock_collective", "single_rank"}:
        return BackwardPromotion(
            False, "reference_only",
            "mock_collective proves paired-IR semantics, not multi-device execution",
        )
    if normalized not in {"nccl", "rccl"}:
        return BackwardPromotion(False, "unsupported", f"unknown provider {provider!r}")
    fixture = _REPO_ROOT / numerical_fixture if numerical_fixture else None
    if not exact_target or fixture is None or not fixture.is_file():
        return BackwardPromotion(
            False, "hardware_gated",
            f"{normalized} needs an exact multi-device target and numerical fixture",
        )
    return BackwardPromotion(True, "device_verified_abi", "exact collective proof present")


def accelerator_backward_promotion(
    row: ExecutionRow,
    *,
    fresh_process: bool = False,
) -> BackwardPromotion:
    """Apply Phase-6 provenance requirements to one backward execution row."""
    if row.direction != "backward":
        return BackwardPromotion(False, "forward_only", "forward rows cannot prove training")
    if not row.device_proof:
        return BackwardPromotion(False, "runtime_bound", "no canonical device proof")
    fixture = _REPO_ROOT / row.numerical_fixture if row.numerical_fixture else None
    if not row.evidence_target or fixture is None or not fixture.is_file() or not row.proof_build:
        return BackwardPromotion(False, "invalid_evidence", "incomplete exact-target proof")
    if row.target == "apple_gpu":
        if row.execution_mode != "metal_runtime" or not fresh_process:
            return BackwardPromotion(
                False, "provenance_incomplete",
                "Apple GPU backward requires metal_runtime in a fresh process",
            )
    if row.target == "rocm" and row.execution_mode != "hip_runtime":
        return BackwardPromotion(False, "provenance_incomplete", "ROCm requires hip_runtime")
    if row.target.startswith("nvidia") and row.execution_mode not in {"cuda_runtime", "cupti_runtime"}:
        return BackwardPromotion(False, "provenance_incomplete", "NVIDIA requires CUDA execution")
    return BackwardPromotion(True, row.device_proof, "exact backward proof satisfies Phase 6")


__all__ = [
    "BackwardPromotion", "accelerator_backward_promotion", "collective_promotion",
]
