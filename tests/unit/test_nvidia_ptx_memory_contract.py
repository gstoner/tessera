"""Host-free truth guards for PTX 9.3 memory-model boundaries."""
from __future__ import annotations

import pytest

from tessera.compiler.nvidia_ptx_memory_contract import (
    PTX_ATOMIC_SEMANTICS,
    PTX_MEMORY_MODEL,
    PTX_MEMORY_SCOPES,
    validate_ptx_memory_claim,
)


def test_sm120_ptx_memory_scopes_and_semantics_are_explicit() -> None:
    assert PTX_MEMORY_MODEL.minimum_sm == 70
    assert PTX_MEMORY_SCOPES == ("cta", "cluster", "gpu", "sys")
    assert PTX_ATOMIC_SEMANTICS == ("relaxed", "acquire", "release", "acq_rel")


def test_packed_vector_and_submission_do_not_overclaim_ordering() -> None:
    assert PTX_MEMORY_MODEL.vector_elements_atomic_together is False
    assert PTX_MEMORY_MODEL.packed_elements_atomic_together is False
    assert PTX_MEMORY_MODEL.ordered_submission_implies_intra_kernel_order is False


@pytest.mark.parametrize(
    "claim",
    [
        {"packed_atomic_as_unit": True},
        {"mixed_size_race_defined": True},
        {"reduction_acquires": True},
        {"texture_coherent": True},
    ],
)
def test_invalid_ptx_memory_claims_fail_closed(claim: dict[str, bool]) -> None:
    with pytest.raises(ValueError, match="invalid PTX memory-model claim"):
        validate_ptx_memory_claim(**claim)


def test_empty_ptx_memory_claim_is_valid() -> None:
    validate_ptx_memory_claim()
