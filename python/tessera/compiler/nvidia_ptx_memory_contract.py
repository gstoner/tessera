"""PTX 9.3 memory-model boundaries relevant to NVIDIA native artifacts."""
from __future__ import annotations

from dataclasses import dataclass


PTX_MEMORY_SCOPES = ("cta", "cluster", "gpu", "sys")
PTX_ATOMIC_SEMANTICS = ("relaxed", "acquire", "release", "acq_rel")


@dataclass(frozen=True)
class PtxMemoryModelContract:
    minimum_sm: int = 70
    vector_elements_atomic_together: bool = False
    packed_elements_atomic_together: bool = False
    mixed_size_data_races_defined: bool = False
    reductions_form_acquire_patterns: bool = False
    texture_and_global_nc_covered: bool = False
    ordered_submission_implies_intra_kernel_order: bool = False


PTX_MEMORY_MODEL = PtxMemoryModelContract()


def validate_ptx_memory_claim(
    *, packed_atomic_as_unit: bool = False,
    mixed_size_race_defined: bool = False,
    reduction_acquires: bool = False,
    texture_coherent: bool = False,
) -> None:
    """Fail closed for claims excluded by the PTX memory model."""
    claims = {
        "packed/vector access is atomic as one unit": packed_atomic_as_unit,
        "mixed-size data-race behavior is defined": mixed_size_race_defined,
        "red operation forms an acquire pattern": reduction_acquires,
        "texture/ld.global.nc participates in coherent global ordering": texture_coherent,
    }
    invalid = [description for description, enabled in claims.items() if enabled]
    if invalid:
        raise ValueError("invalid PTX memory-model claim: " + "; ".join(invalid))


__all__ = [
    "PTX_ATOMIC_SEMANTICS",
    "PTX_MEMORY_MODEL",
    "PTX_MEMORY_SCOPES",
    "PtxMemoryModelContract",
    "validate_ptx_memory_claim",
]
